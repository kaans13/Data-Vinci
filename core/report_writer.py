"""
report_writer.py — Data-Autopsy v4

Bilirkişi Kalite Raporu:
  • Markdown dışa aktarım (kullanıcı adı + tarih + Güvenilirlik Sertifikası)
  • Before/After istatistik tablosu
  • Varyans DİKKAT uyarısı (>%20 değişim)
  • Spot-Check tablosu (rastgele 5-10 değişim örneği)
  • Korelasyon matrisi özeti
  • Analitik yol haritası (Regresyon / Sınıflandırma hazırlığı)
  • Güven Skoru (0-100, işlemlerle artar)
"""
from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Any

from core.audit_logger import _sanitize, safe_dumps

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Güven Skoru — işlemler tamamlandıkça artar
# ═══════════════════════════════════════════════════════════════════════════

class TrustScore:
    """
    Başlangıç: 60.
    Normalize  +5  (max +15)
    Impute     +8  (max +16)
    Outlier    +4  (max +12)
    Benford OK +3
    Max: 100
    """
    def __init__(self, initial: float = 60.0) -> None:
        self._score = initial
        self._events: list[dict] = []

    def update(self, event: str, delta: float, reason: str = "") -> None:
        self._score = min(100.0, max(0.0, self._score + delta))
        self._events.append({
            "event": event, "delta": delta,
            "score": round(self._score, 1), "reason": reason,
        })
        logger.debug("TrustScore %s: %+.1f → %.1f", event, delta, self._score)

    @property
    def score(self) -> float:
        return round(self._score, 1)

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def label(self) -> str:
        s = self._score
        if s >= 90: return "Mükemmel ✓"
        if s >= 75: return "İyi"
        if s >= 60: return "Orta"
        return "Zayıf ⚠"


# ═══════════════════════════════════════════════════════════════════════════
# Markdown Rapor Üretici
# ═══════════════════════════════════════════════════════════════════════════

def generate_markdown_report(
    user_name:       str,
    audit_records:   list[dict],
    before_after:    dict,       # {col: {mean_before, mean_after, …}}
    spot_checks:     list[dict], # [{row, col, before, after}, …]
    correlation:     dict,       # {pairs: […], columns: […]}
    trust_score:     TrustScore,
    file_meta:       dict | None = None,
    col_profiles:    dict | None = None,
    output_path:     Path | None = None,
) -> str:
    """
    Profesyonel Bilirkişi Raporu (Markdown).
    Returns: rapor metni (str)
    """
    now       = time.strftime("%d.%m.%Y %H:%M:%S")
    date_only = time.strftime("%d.%m.%Y")
    ts        = trust_score

    # Radikal değişim uyarısı
    radical_cols = [
        col for col, d in before_after.items()
        if d.get("radical_change")
    ]

    lines: list[str] = []

    # ── Başlık ──────────────────────────────────────────────────────────
    lines += [
        "---",
        "# 🔬 Veri Güvenilirlik Sertifikası",
        f"**Hazırlayan:** {user_name}  ",
        f"**Otopsi Tarihi:** {date_only}  ",
        f"**Rapor Saati:** {now}  ",
        f"**Araç:** Data-Autopsy v4.0  ",
        "---",
        "",
    ]

    # ── Radikal değişim uyarısı ─────────────────────────────────────────
    if radical_cols:
        lines += [
            "> ## ⚠️ DİKKAT: Veri Yapısı Radikal Şekilde Değişti!",
            ">",
            "> Aşağıdaki sütunlarda varyans **%20'den fazla** değişti. "
            "İşlemlerinizi gözden geçirin:",
            ">",
        ]
        for col in radical_cols:
            d  = before_after[col]
            vp = d.get("var_change_pct", "?")
            lines.append(f"> - **{col}**: Varyans değişimi `{vp:+.1f}%`")
        lines += [">", "---", ""]

    # ── Güven Skoru ─────────────────────────────────────────────────────
    bar_filled = int(ts.score / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    lines += [
        "## 📊 Veri Güven Skoru",
        "",
        f"```",
        f"[{bar}] {ts.score:.1f}/100  ({ts.label()})",
        f"```",
        "",
    ]
    if ts.events:
        lines.append("| İşlem | Δ Puan | Gerekçe |")
        lines.append("|---|---|---|")
        for ev in ts.events:
            sign = "+" if ev["delta"] >= 0 else ""
            lines.append(f"| {ev['event']} | {sign}{ev['delta']:.1f} | {ev['reason']} |")
        lines.append("")

    # ── Dosya Meta ──────────────────────────────────────────────────────
    if file_meta:
        lines += [
            "## 📁 Veri Seti Bilgisi",
            "",
            f"| Özellik | Değer |",
            f"|---|---|",
            f"| Dosya | `{file_meta.get('file_name', '—')}` |",
            f"| Satır | {file_meta.get('rows', '—'):,} |",
            f"| Sütun | {len(file_meta.get('columns', []))} |",
            f"| Encoding | `{file_meta.get('encoding', '—')}` |",
            f"| Boyut | {file_meta.get('file_size_mb', 0):.2f} MB |",
            f"| Kalite Skoru | {file_meta.get('quality_score', '—')}/100 |",
            "",
        ]
        if file_meta.get("format_warnings"):
            lines.append("**⚠️ Yükleme Uyarıları:**")
            for w in file_meta["format_warnings"]:
                lines.append(f"- {w}")
            lines.append("")

    # ── İşlem Özeti ─────────────────────────────────────────────────────
    ops_by_type: dict[str, list] = {}
    for r in audit_records:
        op = r.get("operation", "?")
        ops_by_type.setdefault(op, []).append(r)

    total = len(audit_records)
    ok    = sum(1 for r in audit_records if r.get("status") == "SUCCESS")
    err   = sum(1 for r in audit_records if r.get("status") == "ERROR")
    warn  = sum(1 for r in audit_records if r.get("status") == "WARNING")

    lines += [
        "## 🔧 İşlem Özeti",
        "",
        f"| Toplam | Başarılı | Uyarı | Hata |",
        f"|---|---|---|---|",
        f"| {total} | {ok} ✓ | {warn} ⚠ | {err} ✗ |",
        "",
    ]

    # ── Before / After İstatistik ────────────────────────────────────────
    if before_after:
        lines += [
            "## 📈 İstatistiksel Karşılaştırma (Önce / Sonra)",
            "",
            "| Sütun | Mean Önce | Mean Sonra | Std Önce | Std Sonra | Skew Önce | Skew Sonra | Varyans Δ% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for col, d in before_after.items():
            vcp = d.get("var_change_pct")
            flag = " ⚠️" if d.get("radical_change") else ""
            lines.append(
                f"| **{col}** | {d.get('mean_before','—')} | {d.get('mean_after','—')} "
                f"| {d.get('std_before','—')} | {d.get('std_after','—')} "
                f"| {d.get('skew_before','—')} | {d.get('skew_after','—')} "
                f"| {f'{vcp:+.1f}%' if vcp is not None else '—'}{flag} |"
            )
        lines.append("")

    # ── Spot Check ──────────────────────────────────────────────────────
    if spot_checks:
        lines += [
            "## 🔍 Değişim Kanıtı (Rastgele Örnekler)",
            "",
            "| Satır No | Sütun | Eski Değer | Yeni Değer |",
            "|---|---|---|---|",
        ]
        for sc in spot_checks[:10]:
            old = str(sc.get("before", "—"))[:60]
            new = str(sc.get("after",  "—"))[:60]
            lines.append(
                f"| {sc.get('row','—')} | {sc.get('col','—')} | `{old}` | `{new}` |"
            )
        lines.append("")

    # ── Korelasyon ──────────────────────────────────────────────────────
    if correlation.get("pairs"):
        lines += [
            "## 🔗 Korelasyon Matrisi (Pearson)",
            "",
            "| Sütun A | Sütun B | Pearson r | Spearman r | Güç | Anlamlı |",
            "|---|---|---|---|---|---|",
        ]
        for p in correlation["pairs"][:8]:
            sig = "✓" if p.get("significant") else "✗"
            lines.append(
                f"| {p['col_a']} | {p['col_b']} "
                f"| {p['pearson_r']:.4f} | {p['spearman_r']:.4f} "
                f"| {p['strength']} | {sig} |"
            )
        lines.append("")

    # ── Sütun Profili ────────────────────────────────────────────────────
    if col_profiles:
        lines += [
            "## 🏥 Sütun Sağlık Haritası",
            "",
            "| Sütun | Eksik % | Tip Uyumsuzluk % | Kalite |",
            "|---|---|---|---|",
        ]
        for col, p in col_profiles.items():
            qs  = p.get("quality_score", 0)
            ico = "🟢" if qs >= 80 else "🟡" if qs >= 50 else "🔴"
            lines.append(
                f"| {col} | {p.get('miss_pct', 0):.1f}% "
                f"| {p.get('type_mismatch_pct', 0):.1f}% "
                f"| {ico} {qs}/100 |"
            )
        lines.append("")

    # ── Analitik Yol Haritası ───────────────────────────────────────────
    lines += [
        "## 🗺️ Analitik Yol Haritası",
        "",
    ]

    # Normallik & regresyon hazırlığı
    norm_ops    = ops_by_type.get("NORMALIZE", [])
    impute_ops  = ops_by_type.get("IMPUTE",    [])
    outlier_ops = ops_by_type.get("AUDIT",     [])

    if norm_ops or outlier_ops:
        lines += [
            "### 📉 Regresyon Hazırlığı",
            "",
            "Normalizasyon ve/veya aykırı değer temizliği uygulandı.",
        ]
        if not radical_cols:
            lines.append(
                "✅ Veri yapısı korundu. **Lineer / Polinom Regresyon** "
                "için uygundur."
            )
        else:
            lines.append(
                "⚠️ Bazı sütunlarda önemli yapısal değişim var. "
                "Regresyon öncesi hipotez testleri yapmanız önerilir."
            )
        lines.append("")

    if impute_ops:
        miss_cols = [r["parameters"].get("column", "—")
                     for r in impute_ops if r.get("parameters")]
        lines += [
            "### 🌲 Sınıflandırma Hazırlığı",
            "",
            f"Eksik veri doldurma uygulandı: {', '.join(set(miss_cols))}",
            "",
            "✅ **Random Forest / XGBoost** için veri bütünlüğü sağlandı. "
            "One-hot encoding adımını unutmayın.",
            "",
        ]

    # ── Veri Kaybı ──────────────────────────────────────────────────────
    if before_after:
        total_var_loss = sum(
            abs(d.get("var_change_pct", 0) or 0)
            for d in before_after.values()
        ) / max(len(before_after), 1)
        lines += [
            "### 📉 Veri Kaybı Analizi",
            "",
            f"Ortalama varyans değişimi: `{total_var_loss:.1f}%`",
        ]
        if total_var_loss > 20:
            lines.append(
                "\n⚠️ **Yüksek varyans kaybı.** Silme yerine "
                "Winsorize / Robust scaling kullanmayı değerlendirin."
            )
        else:
            lines.append(
                "\n✅ Varyans değişimi kabul edilebilir sınırlar içinde."
            )
        lines.append("")

    # ── İşlem Logu ──────────────────────────────────────────────────────
    lines += [
        "## 📋 İşlem Logu",
        "",
        "| Zaman | İşlem | Modül | Durum | Mesaj |",
        "|---|---|---|---|---|",
    ]
    for r in audit_records[-30:]:
        st_ico = "✓" if r.get("status") == "SUCCESS" else "✗" if r.get("status") == "ERROR" else "⚠"
        msg = str(r.get("message", ""))[:60]
        lines.append(
            f"| {r.get('timestamp','')[:19]} | {r.get('operation','?')} "
            f"| {r.get('module','?')} | {st_ico} | {msg} |"
        )
    lines.append("")

    # ── İmza ────────────────────────────────────────────────────────────
    lines += [
        "---",
        f"*Bu rapor Data-Autopsy v4.0 tarafından {now} tarihinde otomatik olarak oluşturulmuştur.*  ",
        f"*Hazırlayan: **{user_name}***  ",
        "*Bu belge metodolojik bir denetim çıktısıdır; nihai karar kullanıcıya aittir.*",
        "",
    ]

    text = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("Rapor yazıldı: %s", output_path)

    return text
