"""
quality_scorer.py — Data-Autopsy v5

Deterministik Ceza Modeli (Penalty-Based Trust Score)
======================================================

Problem: Varsayılan 60 puanı istatistiksel olarak temelsiz.

Çözüm: 100'den başlayan ve veri kusurlarının şiddetine/ağırlığına göre
düşen bir ceza modeli. Her bileşen bağımsız, izlenebilir ve tekrarlanabilir.

Matematiksel Formülasyon:
─────────────────────────
  S = 100 × ∏ (1 - wᵢ × pᵢ)     — multiplicative (cezalar birbirini azaltır)

  Bileşenler:
    pNull    = mean(null_pct_per_col / 100)          Eksik veri oranı
    pType    = mean(type_mismatch_pct / 100)         Tip uyumsuzluğu
    pAnomal  = anomaly_col_count / total_col_count   Anomali sütun oranı
    pBenford = benford_mad / 0.15                    Benford sapması (max 0.15 normalize)
    pDup     = duplicate_row_pct / 100               Yinelenen satır oranı

  Ağırlıklar (w): veri kalitesi literatüründen (ISO 25012 / DQ-MOSAD):
    wNull    = 0.35   (en kritik — model sonuçlarını direkt etkiler)
    wType    = 0.20   (aritmetik hatası verir ama ayrıştırılabilir)
    wAnomal  = 0.20   (aykırı değer gürültüsü)
    wBenford = 0.15   (finansal veri için kritik, diğerleri için orta)
    wDup     = 0.10   (veri şişmesi, ama analize az zarar verir)

  Neden multiplicative, additive değil?
    Additive: S = 100 - 35×p1 - 20×p2... → negatif puan alabiliriz.
    Multiplicative: Her faktör [0,1]'e sıkıştırıldığı için sonuç her zaman
    [0,100] aralığında kalır ve cezalar birbirini ezer (gerçekçi davranış).

  Sütun düzeyinde profil ayrıca hesaplanır → hangi sütun sorumlu görülür.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Ağırlıklar (ISO 25012 / DQ-MOSAD referanslı) ────────────────────────────
WEIGHTS = {
    "null":    0.35,
    "type":    0.20,
    "anomaly": 0.20,
    "benford": 0.15,
    "dup":     0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Ağırlıklar 1.0'a toplanmalı"


# ── Veri sınıfları ───────────────────────────────────────────────────────────

@dataclass
class ColumnScore:
    name:              str
    null_pct:          float       # 0–100
    type_mismatch_pct: float       # 0–100
    health:            str         # "clean" | "missing" | "anomaly"
    quality_score:     float       # 0–100 bu sütuna özgü
    penalty_contrib:   float       # bu sütunun toplam cezaya katkısı


@dataclass
class TrustScoreResult:
    score:             float       # 0–100
    grade:             str         # A/B/C/D/F
    components: dict[str, float]   # her bileşenin ceza değeri (0–1)
    penalties:  dict[str, float]   # ağırlıklı ceza (wᵢ × pᵢ)
    col_scores: list[ColumnScore]
    top_issues: list[str]          # kullanıcıya gösterilecek açıklamalar
    benford_used: bool             # Benford bileşeni hesaplandı mı?

    @property
    def label(self) -> str:
        if self.score >= 90: return "Mükemmel"
        if self.score >= 75: return "İyi"
        if self.score >= 60: return "Orta"
        if self.score >= 40: return "Zayıf"
        return "Kritik"


# ── Ana sınıf ────────────────────────────────────────────────────────────────

class DataQualityScorer:
    """
    Deterministik, tekrarlanabilir veri kalitesi skorlayıcı.

    Kullanım:
        scorer = DataQualityScorer()
        result = scorer.score(df)
        print(result.score, result.grade, result.top_issues)

    Benford testi isteğe bağlı — büyük finansal veri setleri için çalıştır:
        result = scorer.score(df, benford_col="tutar")
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or WEIGHTS

    def score(
        self,
        df: pd.DataFrame,
        benford_col: str | None = None,
    ) -> TrustScoreResult:
        n_rows = max(len(df), 1)
        n_cols = max(len(df.columns), 1)

        # ── Bileşen 1: Null oranı ─────────────────────────────────────────
        null_per_col = [float(df[c].isna().mean()) for c in df.columns]
        # Ortalama + maksimum ağırlıklı: tek sütun %80 null olsa bile cezalanır
        p_null = float(0.6 * np.mean(null_per_col) + 0.4 * max(null_per_col))

        # ── Bileşen 2: Tip uyumsuzluğu ────────────────────────────────────
        type_mismatch_rates = []
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                non_na = df[col].dropna().astype(str)
                if len(non_na) == 0:
                    type_mismatch_rates.append(0.0)
                    continue
                # Sayısal görünen ama string olarak saklanan hücreler
                numeric_like = non_na.apply(
                    lambda x: _is_numeric_like(x)
                ).mean()
                # 5–95% arası sayısal → karışık tip = uyumsuzluk
                if 0.05 < numeric_like < 0.95:
                    type_mismatch_rates.append(float(min(numeric_like, 1 - numeric_like) * 2))
                else:
                    type_mismatch_rates.append(0.0)
            else:
                type_mismatch_rates.append(0.0)
        p_type = float(np.mean(type_mismatch_rates)) if type_mismatch_rates else 0.0

        # ── Bileşen 3: Anomali sütun oranı ───────────────────────────────
        anomaly_count = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                clean = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(clean) > 10:
                    mean_, std_ = float(clean.mean()), float(clean.std())
                    if mean_ != 0 and std_ / abs(mean_) > 2.0:
                        anomaly_count += 1
        p_anomaly = anomaly_count / n_cols                 # 0–1

        # ── Bileşen 4: Yinelenen satır oranı ─────────────────────────────
        dup_count = int(df.duplicated().sum())
        p_dup = dup_count / n_rows                         # 0–1

        # ── Bileşen 5: Benford sapması (opsiyonel) ────────────────────────
        p_benford   = 0.0
        benford_used = False
        if benford_col and benford_col in df.columns:
            try:
                p_benford, benford_used = _benford_penalty(df[benford_col])
            except Exception as exc:
                logger.debug("Benford hesaplanamadı: %s", exc)

        components = {
            "null":    p_null,
            "type":    p_type,
            "anomaly": p_anomaly,
            "dup":     p_dup,
            "benford": p_benford,
        }

        # ── Skor hesaplama: S = 100 × ∏(1 - wᵢ × pᵢ) ────────────────────
        penalties = {k: self.weights[k] * v for k, v in components.items()}
        product   = 1.0
        for k, pen in penalties.items():
            product *= (1.0 - pen)
        final_score = round(max(0.0, min(100.0, product * 100)), 1)

        # ── Sütun düzeyinde skorlar ────────────────────────────────────────
        col_scores = []
        for i, col in enumerate(df.columns):
            np_pct   = float(null_per_col[i] * 100)
            tm_pct   = float(type_mismatch_rates[i] * 100) if i < len(type_mismatch_rates) else 0.0
            h        = _col_health(df[col], np_pct)
            # Sütuna özgü ceza: ağırlıklı null + tip
            col_pen  = 0.35 * null_per_col[i] + 0.20 * (type_mismatch_rates[i] if i < len(type_mismatch_rates) else 0)
            col_qs   = round(max(0.0, (1.0 - col_pen) * 100), 1)
            col_scores.append(ColumnScore(
                name=col, null_pct=round(np_pct, 2),
                type_mismatch_pct=round(tm_pct, 2),
                health=h, quality_score=col_qs,
                penalty_contrib=round(col_pen, 4),
            ))
        col_scores.sort(key=lambda c: c.penalty_contrib, reverse=True)

        # ── İnsan okunabilir açıklamalar ──────────────────────────────────
        issues = _build_issues(components, penalties, dup_count, anomaly_count)

        return TrustScoreResult(
            score=final_score,
            grade=_grade(final_score),
            components=_round_dict(components),
            penalties=_round_dict(penalties),
            col_scores=col_scores,
            top_issues=issues[:5],
            benford_used=benford_used,
        )


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _is_numeric_like(s: str) -> bool:
    try:
        float(s.replace(",", ".").replace(" ", ""))
        return True
    except ValueError:
        return False


def _benford_penalty(series: pd.Series) -> tuple[float, bool]:
    """Benford sapması → 0-1 arası normalleştirilmiş ceza."""
    import math
    pos = pd.to_numeric(series, errors="coerce").dropna()
    pos = pos[pos > 0]
    if len(pos) < 100:
        return 0.0, False

    expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

    def first_digit(x: float) -> int:
        s = f"{abs(x):.10e}"
        for c in s:
            if c.isdigit() and c != "0":
                return int(c)
        return 0

    digits = pos.apply(first_digit)
    n      = len(pos)
    obs    = {d: digits.value_counts().get(d, 0) / n for d in range(1, 10)}
    mad    = float(np.mean([abs(obs[d] - expected[d]) for d in range(1, 10)]))
    # Normalize: MAD > 0.15 → tam ceza; 0 → sıfır ceza
    penalty = min(1.0, mad / 0.15)
    return penalty, True


def _col_health(col: pd.Series, null_pct: float) -> str:
    if null_pct > 5:
        return "missing"
    if pd.api.types.is_numeric_dtype(col):
        clean = pd.to_numeric(col, errors="coerce").dropna()
        if len(clean) > 10:
            mean_ = float(clean.mean())
            std_  = float(clean.std())
            if mean_ != 0 and std_ / abs(mean_) > 2.0:
                return "anomaly"
    return "clean"


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _build_issues(
    components: dict, penalties: dict,
    dup_count: int, anomaly_count: int,
) -> list[str]:
    issues = []
    p = components

    if p["null"] > 0.20:
        issues.append(f"Yüksek eksik veri oranı: ortalama %{p['null']*100:.1f} — "
                       f"imputation veya sütun eleme gerekiyor.")
    elif p["null"] > 0.05:
        issues.append(f"Orta düzey eksik veri: %{p['null']*100:.1f} — "
                       f"doldurma işlemi uygulanabilir.")

    if p["type"] > 0.10:
        issues.append(f"Tip uyumsuzluğu: sütunların %{p['type']*100:.0f}'inde "
                       f"karışık veri türü tespit edildi.")

    if anomaly_count > 0:
        issues.append(f"{anomaly_count} sütunda yüksek varyasyon katsayısı "
                       f"(std/mean > 2) — aykırı değer analizi önerilir.")

    if dup_count > 0:
        issues.append(f"{dup_count} yinelenen satır tespit edildi "
                       f"— tekilleştirme uygulanabilir.")

    if p["benford"] > 0.5:
        issues.append("Benford Yasası'ndan ciddi sapma — finansal manipülasyon "
                       "veya örnekleme hatası riski.")

    if not issues:
        issues.append("Belirgin bir kalite sorunu tespit edilmedi.")

    return issues


def _round_dict(d: dict) -> dict:
    return {k: round(float(v), 4) for k, v in d.items()}
