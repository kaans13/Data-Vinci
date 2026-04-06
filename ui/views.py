"""
views.py — Data-Autopsy v4.1  (Optimizasyon & Hata Düzeltme)

Değişiklikler:
  • Tema: hardcoded renk sabitleri kaldırıldı → theme.py'den dinamik okunuyor
  • city_normalize seçeneği NormalizePanel'den kaldırıldı
  • TrustScore: veri yokken gösterilmiyor
  • Büyük veri optimizasyonu: before_after_card sadece özet tutar
  • Tüm panel _work() fonksiyonları gereksiz kopyalardan temizlendi
"""
from __future__ import annotations

import logging
import random
import threading
import time

import flet as ft
import pandas as pd
import numpy as np

from core.i18n_manager import I18nManager
from core.database import DataAutopsyDB
from core.audit_logger import AuditLogger
import ui.theme as T

logger = logging.getLogger(__name__)

# Sabit aksanlar (temadan bağımsız)
P    = T.P
S    = T.S
W    = T.W
E    = T.E
IC   = T.IC
GOLD = T.GOLD

HEALTH_ICO = {"clean": "🟢", "missing": "🟡", "anomaly": "🔴"}


# ── Dinamik renk erişimi ─────────────────────────────────────────────────────
# Her widget oluşturulduğunda T.CARD() çağrılır → o anki temayı yansıtır

def _c(key: str) -> str:
    """Tema rengini döner."""
    return T.get(key)


# ── Widget Fabrikası ─────────────────────────────────────────────────────────

def card(content: ft.Control, pad: int = 16) -> ft.Container:
    return ft.Container(
        content=content,
        bgcolor=T.CARD(),
        border_radius=12,
        padding=pad,
        border=ft.border.all(1, T.BDR()),
    )

def title(t: str, sub: str = "") -> ft.Column:
    rows: list[ft.Control] = [
        ft.Text(t, size=19, weight=ft.FontWeight.BOLD, color=T.TEXT())
    ]
    if sub:
        rows.append(ft.Text(sub, size=12, color=T.SUB()))
    return ft.Column(rows, spacing=3)

def tag(txt: str, clr: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(txt, size=11, color=clr, weight=ft.FontWeight.W_600),
        bgcolor=clr + "22",
        border_radius=6,
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
    )

def hr() -> ft.Divider:
    return ft.Divider(color=T.BDR(), height=1)

def btn(text: str, ico, on_click, color: str = None, height: int = 36) -> ft.ElevatedButton:
    return ft.ElevatedButton(
        text=text, icon=ico, on_click=on_click,
        style=ft.ButtonStyle(bgcolor=color or P, color=ft.Colors.WHITE),
        height=height,
    )

def outbtn(text: str, ico, on_click) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        text=text, icon=ico, on_click=on_click,
        style=ft.ButtonStyle(color=P),
    )

def info_box(text: str, clr: str = None) -> ft.Container:
    clr = clr or IC
    return ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.INFO_OUTLINE, color=clr, size=15),
            ft.Text(text, size=12, color=clr, expand=True),
        ], spacing=8),
        bgcolor=clr + "15",
        border_radius=8,
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        border=ft.border.only(left=ft.BorderSide(3, clr)),
    )

def expert_box(note: str, rec_method: str) -> ft.Container:
    method_labels = {
        "zscore":        ("Z-Score (Klasik)",     IC),
        "iqr":           ("IQR",                  W),
        "robust_zscore": ("Robust Z-Score (MAD)", S),
    }
    lbl, clr = method_labels.get(rec_method, ("—", T.SUB()))
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.LIGHTBULB_OUTLINE, color=GOLD, size=15),
                ft.Text("Bilirkişi Notu", size=12, color=GOLD,
                        weight=ft.FontWeight.W_700),
                ft.Container(
                    content=ft.Text(f"Öneri: {lbl}", size=11, color=clr),
                    bgcolor=clr + "22", border_radius=6,
                    padding=ft.padding.symmetric(horizontal=8, vertical=3),
                ),
            ], spacing=8),
            ft.Text(note, size=12, color=T.TEXT()),
        ], spacing=6),
        bgcolor="#1E1A00" if T._T.get("BG","#09") < "#8" else "#FFFDE7",
        border_radius=8,
        padding=ft.padding.symmetric(horizontal=14, vertical=12),
        border=ft.border.only(left=ft.BorderSide(3, GOLD)),
    )

def effect_box(eff: dict) -> ft.Container:
    if not eff.get("effect_available"):
        return ft.Container()
    cd   = eff.get("cohen_d", 0)
    intr = eff.get("cohen_interpretation", "—")
    clr  = S if abs(cd) < 0.2 else W if abs(cd) < 0.5 else E
    mp   = eff.get("mean_delta_pct")
    vp   = eff.get("var_delta_pct")
    return ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.SHOW_CHART, color=clr, size=15),
            ft.Text("Etki:", size=12, color=T.SUB()),
            tag(f"d={cd:+.3f}", clr),
            tag(intr, clr),
            ft.Text(f"Ort {mp:+.1f}%" if mp is not None else "", size=11, color=T.SUB()),
            ft.Text(f"Var {vp:+.1f}%" if vp is not None else "", size=11, color=T.SUB()),
        ], spacing=6, wrap=True),
        bgcolor=clr + "10", border_radius=8,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        border=ft.border.only(left=ft.BorderSide(3, clr)),
    )

def health_bar(col: str, health: str, miss_pct: float,
               quality: float | None = None) -> ft.Row:
    ico = HEALTH_ICO.get(health, "🟢")
    clr = E if health == "anomaly" else W if health == "missing" else S
    qs_txt = f"  {quality:.0f}" if quality is not None else ""
    return ft.Row([
        ft.Text(f"{ico} {col}", size=12, color=T.TEXT(), width=170,
                overflow=ft.TextOverflow.ELLIPSIS),
        ft.Container(
            content=ft.ProgressBar(
                value=min(1.0, miss_pct / 100),
                color=clr, bgcolor=T.BDR(), height=5, border_radius=3,
            ),
            width=120,
        ),
        ft.Text(f"{miss_pct:.1f}%{qs_txt}", size=11, color=clr,
                width=55, text_align=ft.TextAlign.RIGHT),
    ], spacing=10)

def mini_hist(bins: list, counts: list) -> ft.Container:
    if not bins or not counts:
        return ft.Container()
    max_c = max(counts) or 1
    rows  = []
    for i, cnt in enumerate(counts[:20]):   # max 20 bin göster
        w   = max(2, int(cnt / max_c * 68))
        mid = round((bins[i] + bins[i + 1]) / 2, 2) if i + 1 < len(bins) else round(bins[i], 2)
        rows.append(ft.Row([
            ft.Text(f"{mid:>10.2f}", size=9, color=T.SUB(),
                    font_family="Courier New"),
            ft.Container(
                bgcolor=P if cnt / max_c > 0.6 else IC,
                border_radius=2, width=w, height=8,
            ),
            ft.Text(str(cnt), size=9, color=T.SUB()),
        ], spacing=3))
    return ft.Container(
        content=ft.Column(rows, spacing=1),
        bgcolor=T.CARD2(), border_radius=8, padding=10,
    )

def benford_bars(bar_data: list[dict]) -> ft.Container:
    rows = []
    for b in bar_data:
        obs, exp  = b["obs_pct"], b["exp_pct"]
        diff, susp = b["diff_pct"], b.get("suspicious", False)
        clr   = E if susp else S
        obs_w = max(2, int(obs * 4.0))
        exp_w = max(2, int(exp * 4.0))
        rows.append(ft.Row([
            ft.Text(str(b["digit"]), size=12, color=T.TEXT(),
                    width=16, weight=ft.FontWeight.BOLD),
            ft.Column([
                ft.Row([
                    ft.Container(bgcolor=clr, width=obs_w, height=8, border_radius=2),
                    ft.Text(f"{obs:.1f}%", size=10, color=clr),
                ], spacing=4),
                ft.Row([
                    ft.Container(bgcolor=T.SUB(), width=exp_w, height=8, border_radius=2),
                    ft.Text(f"{exp:.1f}%", size=10, color=T.SUB()),
                ], spacing=4),
            ], spacing=2, expand=True),
            ft.Text(f"{diff:+.1f}%", size=10,
                    color=E if susp else T.SUB(),
                    width=44, text_align=ft.TextAlign.RIGHT),
        ], spacing=6))
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Container(bgcolor=S, width=10, height=8, border_radius=2),
                ft.Text("Gözlenen", size=10, color=S),
                ft.Container(bgcolor=T.SUB(), width=10, height=8, border_radius=2),
                ft.Text("Beklenen", size=10, color=T.SUB()),
            ], spacing=6),
            *rows,
        ], spacing=4),
        bgcolor=T.CARD2(), border_radius=8, padding=10,
    )

def comparison_table(outlier_vals: list, normal_vals: list) -> ft.Container:
    n = min(12, max(len(outlier_vals), len(normal_vals)))
    rows = []
    for i in range(n):
        ov = f"{outlier_vals[i]:.4f}" if i < len(outlier_vals) else "—"
        nv = f"{normal_vals[i]:.4f}"  if i < len(normal_vals)  else "—"
        rows.append(ft.DataRow(cells=[
            ft.DataCell(ft.Text(str(i + 1), size=11, color=T.SUB())),
            ft.DataCell(ft.Text(ov, size=12,
                                color=E if i < len(outlier_vals) else T.SUB())),
            ft.DataCell(ft.Text(nv, size=12, color=T.TEXT())),
        ]))
    return card(ft.Column([
        ft.Text("Aykırı vs. Normal", size=13,
                weight=ft.FontWeight.W_600, color=T.TEXT()),
        ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("#",         size=11, color=T.SUB())),
                ft.DataColumn(ft.Text("🔴 Aykırı", size=11, color=E)),
                ft.DataColumn(ft.Text("🟢 Normal", size=11, color=S)),
            ],
            rows=rows,
            border=ft.border.all(1, T.BDR()), border_radius=8,
            heading_row_height=28, data_row_min_height=24,
        ),
    ]))


def before_after_card(
    operation: str,
    col_summary: dict,
    before_after_stats: dict | None,
) -> ft.Container:
    """İşlem sonrası Before/After özet kartı."""
    all_changes = sum(v.get("changes", 0) for v in col_summary.values())
    all_total   = max(
        next(iter(col_summary.values()), {}).get("total", 1), 1
    )
    pct_changed = round(all_changes / all_total * 100, 1)

    # Spot-check: değişen satırlardan rastgele max 5 örnek
    all_samples: list[dict] = []
    for col, info in col_summary.items():
        for s in info.get("change_samples", []):
            all_samples.append({"col": col, **s})

    sample_rows_data = random.sample(all_samples, min(5, len(all_samples)))

    spot_rows = [
        ft.DataRow(cells=[
            ft.DataCell(ft.Text(str(s.get("row", "—")), size=11, color=T.SUB())),
            ft.DataCell(ft.Text(str(s.get("col",  "—")), size=11, color=T.TEXT())),
            ft.DataCell(ft.Text(str(s.get("before", "—"))[:40], size=11, color=W)),
            ft.DataCell(ft.Text(str(s.get("after",  "—"))[:40], size=11, color=S)),
        ])
        for s in sample_rows_data
    ]

    # Before/After istatistik — sadece değişen sütunlar, max 5
    stat_rows = []
    if before_after_stats:
        for col, d in list(before_after_stats.items())[:5]:
            vcp  = d.get("var_change_pct")
            flag = "⚠️" if d.get("radical_change") else ""
            stat_rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(col, size=11, color=T.TEXT())),
                ft.DataCell(ft.Text(str(d.get("mean_before", "—")), size=11, color=T.SUB())),
                ft.DataCell(ft.Text(str(d.get("mean_after",  "—")), size=11, color=T.TEXT())),
                ft.DataCell(ft.Text(str(d.get("std_before",  "—")), size=11, color=T.SUB())),
                ft.DataCell(ft.Text(str(d.get("std_after",   "—")), size=11, color=T.TEXT())),
                ft.DataCell(ft.Text(
                    f"{vcp:+.1f}% {flag}" if vcp is not None else "—",
                    size=11,
                    color=E if d.get("radical_change") else S,
                )),
            ]))

    radical_alert = any(
        d.get("radical_change") for d in (before_after_stats or {}).values()
    )

    controls: list[ft.Control] = [
        ft.Row([
            ft.Icon(ft.Icons.COMPARE_ARROWS, color=P, size=18),
            ft.Text(f"Etki Özeti — {operation}", size=14,
                    weight=ft.FontWeight.BOLD, color=T.TEXT()),
        ], spacing=8),
        ft.Row([
            tag(f"{all_changes:,} değişiklik", W if all_changes else S),
            tag(f"%{pct_changed} etkilendi", IC),
            tag(f"{len(col_summary)} sütun", T.SUB()),
        ], spacing=8, wrap=True),
    ]

    if radical_alert:
        controls.append(ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=E, size=16),
                ft.Text(
                    "DİKKAT: Veri yapısı önemli ölçüde değişti (varyans >%20)!",
                    size=12, color=E,
                ),
            ], spacing=8),
            bgcolor=E + "18", border_radius=8,
            padding=ft.padding.symmetric(horizontal=12, vertical=10),
            border=ft.border.only(left=ft.BorderSide(3, E)),
        ))

    if spot_rows:
        controls += [
            ft.Text("🔍 Değişim Örnekleri (Rastgele)",
                    size=13, weight=ft.FontWeight.W_600, color=T.TEXT()),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Satır",      size=11, color=T.SUB())),
                    ft.DataColumn(ft.Text("Sütun",      size=11, color=T.SUB())),
                    ft.DataColumn(ft.Text("Eski Değer", size=11, color=W)),
                    ft.DataColumn(ft.Text("Yeni Değer", size=11, color=S)),
                ],
                rows=spot_rows,
                border=ft.border.all(1, T.BDR()), border_radius=8,
                heading_row_height=26, data_row_min_height=22,
            ),
        ]

    if stat_rows:
        controls += [
            ft.Text("📈 İstatistiksel Karşılaştırma",
                    size=13, weight=ft.FontWeight.W_600, color=T.TEXT()),
            ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Sütun",      size=11, color=T.SUB())),
                    ft.DataColumn(ft.Text("Mean Önce",  size=11, color=T.SUB())),
                    ft.DataColumn(ft.Text("Mean Sonra", size=11, color=T.TEXT())),
                    ft.DataColumn(ft.Text("Std Önce",   size=11, color=T.SUB())),
                    ft.DataColumn(ft.Text("Std Sonra",  size=11, color=T.TEXT())),
                    ft.DataColumn(ft.Text("Var Δ%",     size=11, color=E)),
                ],
                rows=stat_rows,
                border=ft.border.all(1, T.BDR()), border_radius=8,
                heading_row_height=26, data_row_min_height=22,
            ),
        ]

    return card(ft.Column(controls, spacing=10))


def revert_confirm_row(
    on_revert, on_confirm,
    result_label: str = "", lang: str = "tr",
) -> ft.Container:
    rv_lbl = "Orijinale Dön"  if lang == "tr" else "Revert to Original"
    cf_lbl = "Sonucu Onayla" if lang == "tr" else "Confirm Result"
    return ft.Container(
        content=ft.Row([
            ft.Text(result_label, size=11, color=T.SUB(), expand=True,
                    overflow=ft.TextOverflow.ELLIPSIS),
            ft.OutlinedButton(
                rv_lbl, icon=ft.Icons.HISTORY, on_click=on_revert,
                style=ft.ButtonStyle(color=W),
            ),
            btn(cf_lbl, ft.Icons.CHECK_CIRCLE_OUTLINE, on_confirm, color=S),
        ], spacing=10),
        bgcolor=T.CARD2(), border_radius=10,
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        border=ft.border.all(1, T.BDR()),
    )


# ── BasePanel ────────────────────────────────────────────────────────────────

class BasePanel:
    def __init__(self, i18n: I18nManager, db: DataAutopsyDB,
                 audit: AuditLogger, page: ft.Page,
                 loading=None, trust_score=None):
        self.i18n        = i18n
        self.db          = db
        self.audit       = audit
        self.page        = page
        self.loading     = loading
        self.trust_score = trust_score

    def t(self, k: str, **kw) -> str:
        return self.i18n.t(k, **kw)

    def build(self) -> ft.Control:
        raise NotImplementedError

    def refresh_columns(self, *a, **kw): pass

    def hard_reset(self) -> None:
        """
        Amnesia protokolü — paneldeki tüm sonuç container'larını temizler.
        Yeni dosya yüklendiğinde main.py çağırır.
        """
        for ctrl in self._get_result_containers():
            try:
                if hasattr(ctrl, "controls"):
                    ctrl.controls = []
                if hasattr(ctrl, "value") and isinstance(getattr(ctrl, "value", None), str):
                    ctrl.value = ""
            except Exception:
                pass
        try:
            self.page.update()
        except Exception:
            pass

    def _get_result_containers(self) -> list:
        """Alt sınıflar temizlenecek container listesini döner."""
        return []

    def _status(self, ctrl: ft.Text, msg: str, clr: str):
        ctrl.value = msg
        ctrl.color = clr
        try: self.page.update()
        except Exception: pass

    def _run(self, title_: str, fn, on_done, on_error):
        if self.loading:
            self.loading.run(title_, fn, on_done, on_error)
        else:
            def _w():
                try:    on_done(fn())
                except Exception as ex: on_error(ex)
            threading.Thread(target=_w, daemon=True).start()

    def _export(self, tname: str):
        def _saved(e: ft.FilePickerResultEvent):
            if not e.path: return
            try:
                fmt = "xlsx" if e.path.endswith(".xlsx") else "csv"
                self.db.export_table(tname, e.path, fmt=fmt)
                self._snack(f"Kaydedildi: {e.path}", S)
            except Exception as ex:
                self._snack(f"Hata: {ex}", E)
        picker = ft.FilePicker(on_result=_saved)
        self.page.overlay.append(picker)
        self.page.update()
        picker.save_file(
            dialog_title="Sonucu Kaydet",
            file_name="data_autopsy_result.csv",
            allowed_extensions=["csv", "xlsx"],
        )

    def _snack(self, msg: str, clr: str):
        self.page.snack_bar = ft.SnackBar(
            ft.Text(msg, color=ft.Colors.WHITE), bgcolor=clr
        )
        self.page.snack_bar.open = True
        try: self.page.update()
        except Exception: pass

    def _lang(self) -> str:
        return self.i18n.language


# ── NormalizePanel ───────────────────────────────────────────────────────────

class NormalizePanel(BasePanel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._status_t  = ft.Text("", size=13)
        self._result_c  = ft.Column([], spacing=10)
        self._col_list  = ft.Column(
            [ft.Text("Veri yükleyiniz.", color=T.SUB(), size=12)],
            scroll=ft.ScrollMode.AUTO, height=210,
        )
        self._col_health: dict = {}
        self._case_dd   = ft.Dropdown(
            options=[], width=210,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._op_cbs: dict[str, ft.Checkbox] = {}
        self._pending_result_table: str | None = None
        self._pending_original_df = None
        self._footer = ft.Column([], spacing=0)

    def _get_result_containers(self) -> list:
        return [self._result_c, self._footer, self._status_t]

    def build(self) -> ft.Control: # <--- EKLENMESİ GEREKEN SATIR
        self._rebuild_opts()
        return ft.Column([
            title(self.t("normalize_title"), self.t("normalize_desc")),
            hr(),] + [
                ft.Row([
                    card(ft.Column([
                        ft.Row([
                            ft.Text(self.t("select_columns"), size=13,
                                    weight=ft.FontWeight.W_600, color=T.TEXT()),
                            ft.TextButton(self.t("select_all"),
                                          on_click=lambda e: self._sel(True)),
                            ft.TextButton(self.t("deselect_all"),
                                          on_click=lambda e: self._sel(False)),
                        ], spacing=6),
                        self._col_list,
                    ], spacing=8)),
                    card(ft.Column([
                        ft.Text(self.t("case_normalization"), size=13,
                                weight=ft.FontWeight.W_600, color=T.TEXT()),
                        self._case_dd,
                        hr(),
                        ft.Text("İşlemler", size=13,
                                weight=ft.FontWeight.W_600, color=T.TEXT()),
                        *list(self._op_cbs.values()),
                    ], spacing=8)),
                ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
                ft.Row([
                    btn(self.t("normalize_run"), ft.Icons.AUTO_FIX_HIGH,
                        self._run_normalize),
                    outbtn(self.t("export"), ft.Icons.DOWNLOAD_OUTLINED,
                           self._do_export),
                ], spacing=10),
                self._status_t,
                self._footer,
                self._result_c,
            ], spacing=14, scroll=ft.ScrollMode.AUTO)


        self._rebuild_opts()
        return ft.Column([
            title(self.t("normalize_title"), self.t("normalize_desc")),
            hr(),
            ft.Row([
                card(ft.Column([
                    ft.Row([
                        ft.Text(self.t("select_columns"), size=13,
                                weight=ft.FontWeight.W_600, color=T.TEXT()),
                        ft.TextButton(self.t("select_all"),
                                      on_click=lambda e: self._sel(True)),
                        ft.TextButton(self.t("deselect_all"),
                                      on_click=lambda e: self._sel(False)),
                    ], spacing=6),
                    self._col_list,
                ], spacing=8)),
                card(ft.Column([
                    ft.Text(self.t("case_normalization"), size=13,
                            weight=ft.FontWeight.W_600, color=T.TEXT()),
                    self._case_dd,
                    hr(),
                    ft.Text("İşlemler", size=13,
                            weight=ft.FontWeight.W_600, color=T.TEXT()),
                    *list(self._op_cbs.values()),
                ], spacing=8)),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
            ft.Row([
                btn(self.t("normalize_run"), ft.Icons.AUTO_FIX_HIGH,
                    self._run_normalize),
                outbtn(self.t("export"), ft.Icons.DOWNLOAD_OUTLINED,
                       self._do_export),
            ], spacing=10),
            self._status_t,
            self._footer,
            self._result_c,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

    def refresh_columns(self, cols: list[str], health: dict | None = None):
        self._col_health = health or {}
        new_cbs = []
        for c in cols:
            ico = HEALTH_ICO.get(self._col_health.get(c, "clean"), "🟢")
            new_cbs.append(
                ft.Checkbox(label=f"{ico} {c}", value=False, active_color=P)
            )
        self._col_list.controls = new_cbs
        try: self.page.update()
        except Exception: pass

    def _rebuild_opts(self):
        self._case_dd.options = [
            ft.dropdown.Option("none",  self.t("no_change")),
            ft.dropdown.Option("lower", self.t("lowercase_mode")),
            ft.dropdown.Option("upper", self.t("uppercase_mode")),
            ft.dropdown.Option("title", self.t("titlecase_mode")),
        ]
        self._case_dd.value = "none"

        # city_normalize KALDIRILDI — tarih normalize kaldı
        ops_list = [
            ("fix_encoding",     self.t("encoding_detection")),
            ("strip_whitespace", self.t("whitespace_fix")),
            ("unicode_normalize",self.t("turkish_fix")),
            ("date_normalize",   "Tarih Standartlaştırma"),
        ]
        for k, lbl in ops_list:
            if k not in self._op_cbs:
                self._op_cbs[k] = ft.Checkbox(
                    label=lbl,
                    value=(k in ("fix_encoding", "strip_whitespace",
                                 "unicode_normalize")),
                    active_color=P,
                )
            else:
                self._op_cbs[k].label = lbl

    def _sel(self, val: bool):
        for cb in self._col_list.controls:
            if isinstance(cb, ft.Checkbox):
                cb.value = val
        self.page.update()

    def _run_normalize(self, e):
        selected = []
        for cb in self._col_list.controls:
            if isinstance(cb, ft.Checkbox) and cb.value:
                lbl = cb.label or ""
                col = lbl.split(" ", 1)[1] if lbl and lbl[0] in "🟢🟡🔴" else lbl
                selected.append(col)

        if not selected:
            self._status(self._status_t, "Lütfen sütun seçiniz.", W)
            return
        if not self.db.table_exists("ds_primary"):
            self._status(self._status_t, "Önce veri yükleyiniz.", W)
            return

        ops  = [k for k, cb in self._op_cbs.items() if cb.value]
        case = self._case_dd.value
        if case and case != "none":
            ops.append(f"turkish_{case}")

        self._pending_original_df = self.db.get_df("ds_primary").copy()
        self._status(self._status_t, "Normalleştiriliyor...", T.SUB())

        def _work():
            from modules.normalizer import normalize_dataframe
            t0 = time.perf_counter()
            df = self.db.get_df("ds_primary")
            result_df, summary = normalize_dataframe(
                df, columns=selected, operations=ops
            )
            tname = self.db.write_result(
                result_df, "normalize",
                source_table="ds_primary", operation="NORMALIZE",
            )
            self.db.update_working_copy(result_df)
            dur = (time.perf_counter() - t0) * 1000
            # BA stats sadece sayısal sütunlar için, büyük veri dostu
            ba_stats = self.db.compute_before_after_stats(
                self._pending_original_df, result_df
            )
            return result_df, summary, tname, dur, ba_stats

        def _done(res):
            result_df, summary, tname, dur, ba_stats = res
            self._pending_result_table = tname
            total = sum(v.get("changes", 0) for v in summary.values())
            self._status(
                self._status_t,
                f"✓ {total} değişiklik | {dur:.0f}ms → {tname}", S,
            )
            self._footer.controls = [
                revert_confirm_row(
                    self._do_revert, self._do_confirm, tname, self._lang()
                )
            ]
            self._result_c.controls = [
                before_after_card("Normalizasyon", summary, ba_stats)
            ]
            self.page.update()

            orig = self._pending_original_df
            for col, info in summary.items():
                self.audit.log_normalize(
                    col, ", ".join(ops),
                    info.get("changes", 0), info.get("total", 0),
                    dur / max(len(summary), 1),
                    before=orig[col]      if orig is not None and col in orig.columns      else None,
                    after=result_df[col]  if col in result_df.columns else None,
                )
            if self.trust_score:
                self.trust_score.update("Normalize", +5, f"{total} değişiklik")

        def _err(exc):
            self._status(self._status_t, f"Hata: {exc}", E)

        self._run("Normalleştiriliyor...", _work, _done, _err)

    def _do_revert(self, e=None):
        if self._pending_original_df is not None:
            self.db.update_working_copy(self._pending_original_df)
            self._footer.controls = []
            self._result_c.controls = []
            self._status(self._status_t, "Orijinal veriye dönüldü.", W)
            self.page.update()
        else:
            self._snack("Geri dönülecek veri yok.", W)

    def _do_confirm(self, e=None):
        self._footer.controls = []
        self._status(
            self._status_t, f"✓ Onaylandı → {self._pending_result_table}", S
        )
        self.page.update()
        self._snack("Normalizasyon sonucu onaylandı.", S)

    def _do_export(self, e=None):
        tname = self._pending_result_table
        if tname and self.db.table_exists(tname):
            self._export(tname)
        else:
            self._status(self._status_t, "Önce normalleştirme çalıştırın.", W)

    def refresh_texts(self):
        self._rebuild_opts()


# ── AuditPanel ───────────────────────────────────────────────────────────────

class AuditPanel(BasePanel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Widget'lar build() içinde oluşturulur — hard_reset için placeholder
        self._status_t = ft.Text("", size=13)
        self._area     = ft.Column([], spacing=12, scroll=ft.ScrollMode.AUTO)

    def _get_result_containers(self) -> list:
        return [self._area, self._status_t]

    def build(self) -> ft.Control:
        self._num_dd = ft.Dropdown(
            label=self.t("select_numeric_col"), width=240,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._grp_dd = ft.Dropdown(
            label="Grup Sütunu", width=240,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._iqr_f = ft.TextField(
            label=self.t("iqr_threshold"), value="1.5", width=130,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._z_f = ft.TextField(
            label=self.t("zscore_threshold"), value="3.0", width=130,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        # _status_t ve _area __init__ içinde oluşturuldu

        return ft.Column([
            title(self.t("audit_title"), self.t("audit_desc")),
            hr(),
            card(ft.Column([
                ft.Row([self._num_dd, self._grp_dd], spacing=12, wrap=True),
                ft.Row([self._iqr_f, self._z_f], spacing=12),
                ft.Row([
                    btn(self.t("normality_test"),
                        ft.Icons.SCIENCE_OUTLINED, self._normality, "#00897B"),
                    btn(self.t("benford_test"),
                        ft.Icons.ANALYTICS,         self._benford,   P),
                    btn(self.t("outlier_iqr"),
                        ft.Icons.SCATTER_PLOT,       self._iqr,       "#2196F3"),
                    btn("Robust Z-Score",
                        ft.Icons.SHOW_CHART,         self._zscore,    "#9C27B0"),
                    btn("Korelasyon",
                        ft.Icons.HUB_OUTLINED,       self._correlation,"#FF5722"),
                    btn(self.t("variance_analysis"),
                        ft.Icons.STACKED_BAR_CHART,  self._variance,  "#FF9800"),
                ], wrap=True, spacing=8),
            ], spacing=10)),
            self._status_t,
            self._area,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

    def refresh_columns(self, num_cols: list[str], all_cols: list[str]):
        self._num_dd.options = [ft.dropdown.Option(c) for c in num_cols]
        self._grp_dd.options = [ft.dropdown.Option(c) for c in all_cols]
        if num_cols: self._num_dd.value = num_cols[0]
        if all_cols: self._grp_dd.value = all_cols[0]
        try: self.page.update()
        except Exception: pass

    def _series(self, col: str):
        if not self.db.table_exists("ds_primary"): return None
        df = self.db.get_df("ds_primary")
        return df[col] if col in df.columns else None

    def _push(self, ctrl: ft.Control):
        self._area.controls.insert(0, ctrl)
        # Bellek tasarrufu: max 8 sonuç tut
        self._area.controls = self._area.controls[:8]
        try: self.page.update()
        except Exception: pass

    def _normality(self, e):
        col = self._num_dd.value
        if not col: return
        s = self._series(col)
        if s is None: return

        def _work():
            from modules.statistical_auditor import (
                run_normality_test, compute_distribution,
            )
            t0   = time.perf_counter()
            nr   = run_normality_test(s, lang=self._lang())
            dist = compute_distribution(s, col, lang=self._lang())
            return nr, dist, (time.perf_counter() - t0) * 1000

        def _done(res):
            nr, dist, dur = res
            clr = S if nr.is_normal else W
            self._push(ft.Column([
                card(ft.Column([
                    ft.Text(f"Normallik Testi — '{col}'",
                            weight=ft.FontWeight.BOLD, color=T.TEXT()),
                    ft.Row([
                        tag(nr.test_name, IC),
                        tag(f"p={nr.p_value}", clr),
                        tag(f"n={nr.sample_size}", T.SUB()),
                        tag(nr.skew_label,
                            W if abs(nr.skewness) > 1 else S),
                    ], spacing=6, wrap=True),
                    ft.Text(
                        f"Skew={nr.skewness}  Kurt={nr.kurtosis}  "
                        f"Ort={dist.descriptive.get('mean')}  "
                        f"Std={dist.descriptive.get('std')}",
                        size=12, color=T.SUB(),
                    ),
                    expert_box(nr.expert_note,
                               nr.recommended_outlier_method),
                ], spacing=8)),
                card(ft.Column([
                    ft.Text("Dağılım", size=13,
                            weight=ft.FontWeight.W_600, color=T.TEXT()),
                    mini_hist(dist.hist_bins, dist.hist_counts),
                ], spacing=6)),
            ], spacing=8))
            self._status(self._status_t,
                         f"Normallik testi tamamlandı ({dur:.0f}ms)", S)

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run(f"Normallik: {col}", _work, _done, _err)

    def _benford(self, e):
        col = self._num_dd.value
        if not col: return
        s = self._series(col)
        if s is None: return

        def _work():
            from modules.statistical_auditor import run_benford_test
            t0 = time.perf_counter()
            r  = run_benford_test(s, column_name=col, lang=self._lang())
            return r, (time.perf_counter() - t0) * 1000

        def _done(res):
            r, dur = res
            # is_suspicious → incelenmeli (uyarı rengi), değil hata rengi
            clr = W if r.is_suspicious else S
            lbl = self.t("suspicious") if r.is_suspicious else self.t("normal_dist")
            norm_ctrl = ft.Container()
            if r.normality:
                norm_ctrl = expert_box(
                    r.normality.expert_note,
                    r.normality.recommended_outlier_method,
                )
            self._push(ft.Column([
                card(ft.Column([
                    ft.Text(f"Benford Yasası — '{col}'",
                            weight=ft.FontWeight.BOLD, color=T.TEXT()),
                    ft.Row([
                        tag(lbl, clr),
                        tag(r.conformity, IC),
                        tag(f"n={r.sample_size}", T.SUB()),
                        tag(f"MAD={r.mad:.4f}", T.SUB()),
                    ], spacing=6, wrap=True),
                    ft.Text(f"χ²={r.chi_square}  p={r.p_value}  ({dur:.0f}ms)",
                            size=12, color=T.SUB()),
                    info_box(r.expert_note, clr),
                    norm_ctrl,
                ], spacing=8)),
                card(ft.Column([
                    ft.Text("Gözlenen vs. Beklenen",
                            size=13, weight=ft.FontWeight.W_600, color=T.TEXT()),
                    benford_bars(r.bar_data),
                ], spacing=6)),
            ], spacing=8))
            self._status(self._status_t, f"Benford tamamlandı ({dur:.0f}ms)", S)
            self.audit.log_benford(col, r.chi_square, r.p_value,
                                   r.is_suspicious, dur)
            if self.trust_score and not r.is_suspicious:
                self.trust_score.update("Benford", +3, "Benford uyumlu")

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run(f"Benford: {col}", _work, _done, _err)

    def _iqr(self, e):
        col = self._num_dd.value
        if not col: return
        s = self._series(col)
        if s is None: return

        def _work():
            from modules.statistical_auditor import run_iqr_outlier_detection
            t0   = time.perf_counter()
            mult = float(self._iqr_f.value or 1.5)
            r    = run_iqr_outlier_detection(
                s, column_name=col, multiplier=mult, lang=self._lang()
            )
            return r, (time.perf_counter() - t0) * 1000

        def _done(res):
            r, dur = res
            self._push(ft.Column([
                card(ft.Column([
                    ft.Text(f"IQR — '{col}'",
                            weight=ft.FontWeight.BOLD, color=T.TEXT()),
                    ft.Row([
                        tag(f"{r.outlier_count} aykırı", W),
                        tag(f"%{r.outlier_rate}",
                            E if r.outlier_rate > 10 else W),
                        tag(f"[{r.lower_bound}, {r.upper_bound}]", IC),
                    ], spacing=6, wrap=True),
                    ft.Text(
                        f"Ort={r.descriptive.get('mean')}  "
                        f"Med={r.descriptive.get('median')}  "
                        f"Std={r.descriptive.get('std')}",
                        size=12, color=T.SUB(),
                    ),
                    expert_box(
                        r.expert_note,
                        r.normality.recommended_outlier_method
                        if r.normality else "iqr",
                    ),
                    effect_box(r.effect_size),
                ], spacing=8)),
                comparison_table(r.outlier_values, r.normal_sample),
            ], spacing=8))
            self._status(self._status_t, f"IQR tamamlandı ({dur:.0f}ms)", S)
            self.audit.log_outlier(
                col, "IQR", float(self._iqr_f.value or 1.5),
                r.outlier_count, r.total_count, dur,
                normality_p=r.normality.p_value if r.normality else None,
            )
            if self.trust_score:
                self.trust_score.update("IQR", +4, f"{r.outlier_count} aykırı")

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run(f"IQR: {col}", _work, _done, _err)

    def _zscore(self, e):
        col = self._num_dd.value
        if not col: return
        s = self._series(col)
        if s is None: return

        def _work():
            from modules.statistical_auditor import run_zscore_outlier_detection
            t0  = time.perf_counter()
            thr = float(self._z_f.value or 3.0)
            r   = run_zscore_outlier_detection(
                s, column_name=col,
                threshold=thr, robust=True, lang=self._lang(),
            )
            return r, (time.perf_counter() - t0) * 1000

        def _done(res):
            r, dur = res
            self._push(ft.Column([
                card(ft.Column([
                    ft.Text(f"{r.method} — '{col}'",
                            weight=ft.FontWeight.BOLD, color=T.TEXT()),
                    ft.Row([
                        tag(r.method, IC),
                        tag(f"{r.outlier_count} aykırı", W),
                        tag(f"%{r.outlier_rate}",
                            E if r.outlier_rate > 10 else W),
                    ], spacing=6, wrap=True),
                    expert_box(
                        r.expert_note,
                        r.normality.recommended_outlier_method
                        if r.normality else "robust_zscore",
                    ),
                    effect_box(r.effect_size),
                ], spacing=8)),
                comparison_table(r.outlier_values, r.normal_sample),
            ], spacing=8))
            self._status(self._status_t,
                         f"Z-Score tamamlandı ({dur:.0f}ms)", S)
            self.audit.log_outlier(
                col, r.method, float(self._z_f.value or 3.0),
                r.outlier_count, r.total_count, dur,
                normality_p=r.normality.p_value if r.normality else None,
            )

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run(f"Z-Score: {col}", _work, _done, _err)

    def _correlation(self, e):
        if not self.db.table_exists("ds_primary"):
            self._status(self._status_t, "Önce veri yükleyiniz.", W)
            return

        def _work():
            t0  = time.perf_counter()
            cor = self.db.compute_correlation_matrix("ds_primary")
            return cor, (time.perf_counter() - t0) * 1000

        def _done(res):
            cor, dur = res
            if not cor.get("pairs"):
                self._status(
                    self._status_t,
                    "Korelasyon için yeterli sayısal sütun yok.", W,
                )
                return
            rows = []
            for p in cor["pairs"][:10]:
                sig_ico = "✓" if p.get("significant") else "✗"
                r_val   = abs(p["pearson_r"])
                s_clr   = S if r_val > 0.6 else W if r_val > 0.3 else T.SUB()
                rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(p["col_a"], size=12, color=T.TEXT())),
                    ft.DataCell(ft.Text(p["col_b"], size=12, color=T.TEXT())),
                    ft.DataCell(ft.Container(
                        content=ft.Text(f"{p['pearson_r']:.4f}",
                                        size=12, color=s_clr),
                        bgcolor=s_clr + "22", border_radius=4,
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    )),
                    ft.DataCell(ft.Text(f"{p['spearman_r']:.4f}",
                                        size=12, color=T.SUB())),
                    ft.DataCell(tag(p["strength"], s_clr)),
                    ft.DataCell(ft.Text(sig_ico, size=14,
                                        color=S if sig_ico == "✓" else E)),
                ]))
            self._push(card(ft.Column([
                ft.Text("Korelasyon Matrisi",
                        weight=ft.FontWeight.BOLD, size=14, color=T.TEXT()),
                ft.Text(f"Sütunlar: {', '.join(cor['columns'])}",
                        size=12, color=T.SUB()),
                ft.DataTable(
                    columns=[
                        ft.DataColumn(ft.Text("Sütun A",    size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("Sütun B",    size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("Pearson r",  size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("Spearman r", size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("Güç",        size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("p<0.05",     size=11, color=T.SUB())),
                    ],
                    rows=rows,
                    border=ft.border.all(1, T.BDR()), border_radius=8,
                    heading_row_height=28, data_row_min_height=24,
                ),
            ], spacing=8)))
            self._status(self._status_t,
                         f"Korelasyon tamamlandı ({dur:.0f}ms)", S)

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run("Korelasyon hesaplanıyor...", _work, _done, _err)

    def _variance(self, e):
        target = self._num_dd.value
        group  = self._grp_dd.value
        if not target or not group: return

        def _work():
            from modules.statistical_auditor import run_variance_impact_analysis
            t0 = time.perf_counter()
            df = self.db.get_df("ds_primary")
            r  = run_variance_impact_analysis(
                df, target, group, lang=self._lang()
            )
            return r, (time.perf_counter() - t0) * 1000

        def _done(res):
            r, dur = res
            clr = S if r.is_significant else T.SUB()
            self._push(card(ft.Column([
                ft.Text(f"Varyans — '{target}' ~ '{group}'",
                        weight=ft.FontWeight.BOLD, color=T.TEXT()),
                ft.Row([
                    tag(r.effect, clr),
                    tag(f"η²={r.eta_squared}", IC),
                    tag(f"F={r.f_statistic}", T.SUB()),
                    tag(f"p={r.p_value}", clr),
                ], spacing=6, wrap=True),
                info_box(r.expert_note, clr),
            ], spacing=8)))
            self._status(self._status_t,
                         f"Varyans analizi tamamlandı ({dur:.0f}ms)", S)

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run(f"Varyans: {target}~{group}", _work, _done, _err)


# ── MatchPanel ───────────────────────────────────────────────────────────────

class MatchPanel(BasePanel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._status_t = ft.Text("", size=13)
        self._result_c = ft.Column([], spacing=8)

    def _get_result_containers(self) -> list:
        return [self._result_c, self._status_t]

    def build(self) -> ft.Control:
        self._col_p = ft.Dropdown(
            label=self.t("primary_dataset"), width=240,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._col_s = ft.Dropdown(
            label=self.t("secondary_dataset"), width=240,
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._algo = ft.Dropdown(
            label=self.t("algorithm"), width=230,
            options=[
                ft.dropdown.Option("jaro_winkler", self.t("algo_jaro_winkler")),
                ft.dropdown.Option("levenshtein",  self.t("algo_levenshtein")),
                ft.dropdown.Option("token_sort",   self.t("algo_token_sort")),
            ],
            value="jaro_winkler",
            border_color=T.BDR(), focused_border_color=P,
            text_style=ft.TextStyle(color=T.TEXT()),
        )
        self._thr_lbl = ft.Text("Eşik: 80", size=12, color=T.SUB())
        self._thr = ft.Slider(
            min=50, max=100, value=80, divisions=50,
            label="{value}", active_color=P,
        )
        self._thr.on_change = lambda ev: (
            setattr(self._thr_lbl, "value",
                    f"Eşik: {ev.control.value:.0f}"),
            self.page.update(),
        )
        # _status_t ve _result_c __init__ içinde oluşturuldu

        return ft.Column([
            title(self.t("match_title"), self.t("match_desc")),
            hr(),
            card(ft.Column([
                ft.Row([self._col_p, self._col_s], spacing=12, wrap=True),
                self._algo,
                ft.Row([
                    ft.Text(self.t("similarity_threshold"),
                            size=13, color=T.TEXT()),
                    self._thr_lbl,
                ], spacing=8),
                self._thr,
                btn(self.t("run_matching"), ft.Icons.COMPARE_ARROWS,
                    self._run_match),
            ], spacing=10)),
            self._status_t,
            self._result_c,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

    def refresh_columns(self, cols: list[str]):
        opts = [ft.dropdown.Option(c) for c in cols]
        self._col_p.options = opts
        self._col_s.options = opts
        try: self.page.update()
        except Exception: pass

    def _run_match(self, e):
        col_p = self._col_p.value
        col_s = self._col_s.value
        if not col_p or not col_s:
            self._status(self._status_t, "Sütun seçiniz.", W)
            return
        if not (self.db.table_exists("ds_primary") and
                self.db.table_exists("ds_secondary")):
            self._status(self._status_t,
                         "Her iki veri seti de yüklenmelidir.", W)
            return

        thr  = float(self._thr.value)
        algo = self._algo.value

        def _work():
            from modules.fuzzy_matcher import (
                run_fuzzy_match, merge_on_match_result, MatchAlgorithm,
            )
            t0   = time.perf_counter()
            df_p = self.db.get_df("ds_primary")
            df_s = self.db.get_df("ds_secondary")
            result = run_fuzzy_match(
                df_p[col_p].astype(str),
                df_s[col_s].astype(str),
                algorithm=MatchAlgorithm(algo),
                threshold=thr,
            )
            merged = merge_on_match_result(df_p, df_s, result)
            tname  = self.db.write_result(
                merged, "match",
                source_table="ds_primary", operation="MATCH",
            )
            return result, (time.perf_counter() - t0) * 1000, tname

        def _done(res):
            result, dur, tname = res
            self._status(
                self._status_t,
                f"✓ {result.matched_count} eşleşme | "
                f"%{result.match_rate_pct} | {dur:.0f}ms → {tname}",
                S,
            )
            rows = [
                ft.DataRow(cells=[
                    ft.DataCell(
                        ft.Text(m.primary_value[:40], size=12, color=T.TEXT())
                    ),
                    ft.DataCell(
                        ft.Text(m.secondary_value[:40], size=12, color=T.TEXT())
                    ),
                    ft.DataCell(ft.Container(
                        content=ft.Text(
                            f"{m.score:.1f}",
                            color=S if m.score >= thr else W,
                        ),
                        bgcolor=(S if m.score >= thr else W) + "22",
                        border_radius=4,
                        padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    )),
                    ft.DataCell(tag(
                        m.status,
                        S if m.status == "matched" else W,
                    )),
                ])
                for m in result.matched[:25]
            ]
            self._result_c.controls = [
                card(ft.Column([
                    ft.Text("Eşleştirme Özeti",
                            weight=ft.FontWeight.BOLD, size=14, color=T.TEXT()),
                    ft.Row([
                        tag(f"Eşleşti: {result.matched_count}", S),
                        tag(f"Eşleşmedi (B): {len(result.unmatched_primary)}", W),
                        tag(f"Eşleşmedi (İ): {len(result.unmatched_secondary)}", W),
                        tag(f"%{result.match_rate_pct}", P),
                    ], spacing=6, wrap=True),
                    outbtn(self.t("export"), ft.Icons.DOWNLOAD_OUTLINED,
                           lambda ev, tn=tname: self._export(tn)),
                ], spacing=8)),
                card(ft.Column([
                    ft.Text(
                        "Eşleşen Çiftler"
                        + (" (ilk 25)" if len(result.matched) > 25 else ""),
                        weight=ft.FontWeight.W_600, size=13, color=T.TEXT(),
                    ),
                    ft.DataTable(
                        columns=[
                            ft.DataColumn(ft.Text("Birincil", size=11,
                                                  color=T.SUB())),
                            ft.DataColumn(ft.Text("İkincil",  size=11,
                                                  color=T.SUB())),
                            ft.DataColumn(ft.Text("Skor",     size=11,
                                                  color=T.SUB())),
                            ft.DataColumn(ft.Text("Durum",    size=11,
                                                  color=T.SUB())),
                        ],
                        rows=rows,
                        border=ft.border.all(1, T.BDR()), border_radius=8,
                        heading_row_height=28, data_row_min_height=24,
                    ) if rows else ft.Text("Eşleşen kayıt yok.", color=T.SUB()),
                ], spacing=8)),
            ]
            self.page.update()
            self.audit.log_match(
                col_p, col_s, algo, thr,
                result.matched_count,
                len(result.unmatched_primary),
                len(result.unmatched_secondary),
                dur,
            )

        def _err(ex):
            self._status(self._status_t, f"Hata: {ex}", E)

        self._run("Eşleştirme yapılıyor...", _work, _done, _err)


# ── ImputePanel ──────────────────────────────────────────────────────────────

class ImputePanel(BasePanel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pending_result_table: str | None = None
        self._pending_original_df = None
        self._footer = ft.Column([], spacing=0)

    def _get_result_containers(self) -> list:
        # _area ve _status_t build() içinde atanır; hard_reset öncesi build
        # çağrılmış olmayabilir, hasattr ile güvenle eriş
        containers = [self._footer]
        if hasattr(self, "_area"):
            containers.append(self._area)
        if hasattr(self, "_status_t"):
            containers.append(self._status_t)
        return containers

    def build(self) -> ft.Control:
        self._area     = ft.Column([], spacing=8)
        self._status_t = ft.Text("", size=13)
        return ft.Column([
            title(self.t("impute_title"), self.t("impute_desc")),
            hr(),
            card(ft.Column([
                ft.Text(self.t("missing_analysis"),
                        weight=ft.FontWeight.W_600, color=T.TEXT()),
                info_box(
                    "MCAR/MAR/MNAR deseni tespit eder. "
                    "İşlem sonrası Onayla veya Orijinale Dön.",
                    IC,
                ),
                ft.Row([
                    btn(self.t("missing_analysis"),
                        ft.Icons.SEARCH, self._analyze),
                    outbtn(self.t("export"), ft.Icons.DOWNLOAD_OUTLINED,
                           self._do_export),
                ], spacing=10),
            ], spacing=10)),
            self._status_t,
            self._footer,
            self._area,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

    def _analyze(self, e):
        if not self.db.table_exists("ds_primary"):
            self._status(self._status_t, "Önce veri yükleyiniz.", W)
            return
        try:
            from modules.smart_imputer import (
                analyze_missing_patterns, impute_column,
            )
            df  = self.db.get_df("ds_primary")
            
            analysis = analyze_missing_patterns(df)
            self._area.controls.clear()
            found = False

            for col, info in analysis.items():
                if info.missing_count == 0:
                    continue
                found = True
                miss_clr = (E  if info.missing_pct > 30
                             else W if info.missing_pct > 10
                             else S)

                def _make_fill(
                    column=col,
                    method=info.recommended_method,
                    pat=info.pattern,
                ):
                    def _do(ev):
                        def _work():
                            df2    = self.db.get_df("ds_primary")
                            before = df2[column].copy()
                            filled, res = impute_column(df2[column], method)
                            df2[column] = filled
                            tname = self.db.write_result(
                                df2, "impute",
                                source_table="ds_primary",
                                operation="IMPUTE",
                            )
                            self.db.update_working_copy(df2)
                            ba = self.db.compute_before_after_stats(
                                self._pending_original_df, df2
                            )
                            return res, tname, before, filled, ba

                        def _done(r):
                            res, tname, bef, aft, ba_stats = r
                            self._pending_result_table = tname
                            self._status(
                                self._status_t,
                                f"✓ '{column}': {res.cells_filled} hücre"
                                f" → {tname}",
                                S,
                            )
                            self._footer.controls = [
                                revert_confirm_row(
                                    self._do_revert, self._do_confirm,
                                    tname, self._lang(),
                                )
                            ]
                            if ba_stats:
                                dummy_summary = {
                                    column: {
                                        "changes": res.cells_filled,
                                        "total":   len(aft),
                                        "change_samples": [],
                                    }
                                }
                                self._area.controls.insert(
                                    0,
                                    before_after_card(
                                        "Doldurma", dummy_summary, ba_stats
                                    ),
                                )
                            self.page.update()
                            self.audit.log_impute(
                                column, pat.value, method.value,
                                res.cells_filled,
                                before=bef, after=aft,
                            )
                            if self.trust_score:
                                self.trust_score.update(
                                    "Impute", +8, f"'{column}' dolduruldu"
                                )

                        def _err(ex):
                            self._status(self._status_t, f"Hata: {ex}", E)

                        self._run(
                            f"'{column}' dolduruluyor...",
                            _work, _done, _err,
                        )
                    return _do

                self._area.controls.append(card(ft.Column([
                    ft.Row([
                        ft.Text(col, weight=ft.FontWeight.BOLD, color=T.TEXT()),
                        tag(info.pattern.value, miss_clr),
                        tag(f"%{info.missing_pct:.1f}", miss_clr),
                        tag(f"{info.missing_count} eksik", T.SUB()),
                    ], spacing=6, wrap=True),
                    health_bar(
                        col,
                        "missing" if info.missing_pct > 5 else "clean",
                        info.missing_pct,
                    ),
                    ft.Text(info.method_reason, size=11, color=T.SUB()),
                    ft.Row([
                        tag(f"Öneri: {info.recommended_method.value}", P),
                        btn("Doldur", ft.Icons.HEALING,
                            _make_fill(), height=32),
                    ], spacing=10),
                ], spacing=8)))

            if not found:
                self._area.controls.append(
                    info_box("Eksik değer bulunamadı. Veri seti temiz! 🎉", S)
                )
            self._status(self._status_t, "Analiz tamamlandı.", S)
        except Exception as ex:
            self._status(self._status_t, f"Hata: {ex}", E)
        self.page.update()

    def _do_revert(self, e=None):
        if self._pending_original_df is not None:
            self.db.update_working_copy(self._pending_original_df)
            self._footer.controls = []
            self._status(self._status_t, "Orijinal veriye dönüldü.", W)
            self.page.update()
        else:
            self._snack("Geri dönülecek veri yok.", W)

    def _do_confirm(self, e=None):
        self._footer.controls = []
        self._status(
            self._status_t,
            f"✓ Onaylandı → {self._pending_result_table}", S,
        )
        self.page.update()
        self._snack("Doldurma sonucu onaylandı.", S)

    def _do_export(self, e=None):
        tname = self._pending_result_table
        if tname and self.db.table_exists(tname):
            self._export(tname)
        else:
            self._status(self._status_t, "Önce doldurma çalıştırın.", W)


# ── ReportPanel ──────────────────────────────────────────────────────────────

class ReportPanel(BasePanel):
    def __init__(self, *a, user_name: str = "", **kw):
        super().__init__(*a, **kw)
        self.user_name = user_name

    def _get_result_containers(self) -> list:
        containers = []
        if hasattr(self, "_area"):
            containers.append(self._area)
        if hasattr(self, "_status_t"):
            containers.append(self._status_t)
        return containers

    def build(self) -> ft.Control:
        self._area     = ft.Column([], spacing=8)
        self._status_t = ft.Text("", size=13)
        return ft.Column([
            title(self.t("report_title"), self.t("report_desc")),
            hr(),
            card(ft.Column([
                ft.Text(
                    f"Oturum: "
                    f"{self.audit.session_start.strftime('%d.%m.%Y %H:%M:%S')}",
                    color=T.TEXT(), size=13,
                ),
                info_box(
                    "İşlemler JSONL olarak kaydediliyor. "
                    "Markdown rapor dışa aktarılabilir.",
                    IC,
                ),
                ft.Row([
                    btn(self.t("generate_report"),
                        ft.Icons.DESCRIPTION, self._generate),
                    btn("Markdown İndir",
                        ft.Icons.DOWNLOAD, self._export_md, color="#2196F3"),
                ], spacing=10),
            ], spacing=10)),
            self._status_t,
            self._area,
        ], spacing=14, scroll=ft.ScrollMode.AUTO)

    def _generate(self, e):
        try:
            report  = self.audit.generate_report()
            summary = report["summary"]
            qs      = summary["quality_score"]
            qs_clr  = S if qs >= 80 else W if qs >= 50 else E

            ts       = self.trust_score
            ts_score = ts.score if ts else 0
            ts_clr   = S if ts_score >= 80 else W if ts_score >= 60 else E

            # Lineage tablosu
            lineage  = self.db.get_lineage()
            lin_rows = [
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(l.get("timestamp", ""), size=11,
                                        color=T.SUB())),
                    ft.DataCell(ft.Text(l.get("operation", ""), size=11,
                                        color=P)),
                    ft.DataCell(ft.Text(l.get("table", ""), size=11,
                                        color=T.TEXT(), width=200)),
                    ft.DataCell(ft.Text(str(l.get("rows", "")), size=11,
                                        color=T.SUB())),
                ])
                for l in lineage[-15:]
            ]

            # Before/After (orijinal vs şu anki)
            ba_ctrl = ft.Container()
            if (self.db.table_exists("ds_primary") and
                    self.db.table_exists("ds_original_p")):
                orig = self.db.get_original("p")
                curr = self.db.get_df("ds_primary")
                if orig is not None:
                    ba      = self.db.compute_before_after_stats(orig, curr)
                    radical = [c for c, d in ba.items()
                               if d.get("radical_change")]
                    if ba:
                        stat_rows = []
                        for col, d in list(ba.items())[:6]:
                            vcp  = d.get("var_change_pct")
                            flag = " ⚠️" if d.get("radical_change") else ""
                            stat_rows.append(ft.DataRow(cells=[
                                ft.DataCell(ft.Text(col, size=11, color=T.TEXT())),
                                ft.DataCell(ft.Text(
                                    str(d.get("mean_before", "")), size=11, color=T.SUB())),
                                ft.DataCell(ft.Text(
                                    str(d.get("mean_after",  "")), size=11, color=T.TEXT())),
                                ft.DataCell(ft.Text(
                                    str(d.get("std_before",  "")), size=11, color=T.SUB())),
                                ft.DataCell(ft.Text(
                                    str(d.get("std_after",   "")), size=11, color=T.TEXT())),
                                ft.DataCell(ft.Text(
                                    f"{vcp:+.1f}%{flag}" if vcp is not None else "—",
                                    size=11,
                                    color=E if d.get("radical_change") else S,
                                )),
                            ]))

                        rad_warn = ft.Container()
                        if radical:
                            rad_warn = ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED,
                                            color=E, size=18),
                                    ft.Text(
                                        f"DİKKAT: {', '.join(radical)} "
                                        f"sütunlarında varyans %20+ değişti!",
                                        size=13, color=E,
                                    ),
                                ], spacing=8),
                                bgcolor=E + "18", border_radius=8,
                                padding=ft.padding.symmetric(
                                    horizontal=14, vertical=12
                                ),
                                border=ft.border.only(
                                    left=ft.BorderSide(3, E)
                                ),
                            )

                        ba_ctrl = card(ft.Column([
                            ft.Text("Önce / Sonra İstatistik", size=13,
                                    weight=ft.FontWeight.W_600, color=T.TEXT()),
                            rad_warn,
                            ft.DataTable(
                                columns=[
                                    ft.DataColumn(ft.Text("Sütun",      size=11, color=T.SUB())),
                                    ft.DataColumn(ft.Text("Mean Önce",  size=11, color=T.SUB())),
                                    ft.DataColumn(ft.Text("Mean Sonra", size=11, color=T.TEXT())),
                                    ft.DataColumn(ft.Text("Std Önce",   size=11, color=T.SUB())),
                                    ft.DataColumn(ft.Text("Std Sonra",  size=11, color=T.TEXT())),
                                    ft.DataColumn(ft.Text("Var Δ%",     size=11, color=E)),
                                ],
                                rows=stat_rows,
                                border=ft.border.all(1, T.BDR()),
                                border_radius=8,
                                heading_row_height=26,
                                data_row_min_height=22,
                            ),
                        ], spacing=8))

            eff_sum  = summary.get("effect_size_summary", {})
            eff_rows = [
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(op, size=12, color=T.TEXT())),
                    ft.DataCell(ft.Text(
                        str(eff.get("avg_cohen_d", "—")), size=12, color=IC
                    )),
                    ft.DataCell(tag(eff.get("interpretation", "—"), IC)),
                ])
                for op, eff in eff_sum.items()
            ]

            self._area.controls = [
                # Skor kartı
                card(ft.Column([
                    ft.Row([
                        ft.Column([
                            ft.Text("Veri Güven Skoru", size=11, color=T.SUB()),
                            ft.Text(f"{ts_score:.0f}/100", size=36,
                                    weight=ft.FontWeight.BOLD, color=ts_clr),
                            ft.Text(ts.label() if ts else "", size=11,
                                    color=ts_clr),
                        ], spacing=2),
                        ft.ProgressBar(
                            value=ts_score / 100, color=ts_clr,
                            bgcolor=T.BDR(), height=10,
                            border_radius=5, width=180,
                        ),
                        ft.Column([
                            ft.Text("Audit Kalitesi", size=11, color=T.SUB()),
                            ft.Text(f"{qs}/100", size=36,
                                    weight=ft.FontWeight.BOLD, color=qs_clr),
                        ], spacing=2),
                    ], spacing=20,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    hr(),
                    ft.Row([
                        tag(f"Toplam: {summary['total_operations']}", IC),
                        tag(f"Başarılı: {summary['success_count']}", S),
                        tag(f"Uyarı: {summary['warning_count']}", W),
                        tag(f"Hata: {summary['error_count']}", E),
                    ], spacing=6, wrap=True),
                ], spacing=10)),
                ba_ctrl,
                # Effect Size
                card(ft.Column([
                    ft.Text("Etki Büyüklüğü (Cohen d)",
                            size=13, weight=ft.FontWeight.W_600, color=T.TEXT()),
                    ft.DataTable(
                        columns=[
                            ft.DataColumn(ft.Text("İşlem",    size=11, color=T.SUB())),
                            ft.DataColumn(ft.Text("Ort. |d|", size=11, color=T.SUB())),
                            ft.DataColumn(ft.Text("Yorum",    size=11, color=T.SUB())),
                        ],
                        rows=eff_rows if eff_rows else [
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text(
                                    "Henüz işlem yok.", size=12, color=T.SUB()
                                )),
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                            ])
                        ],
                        border=ft.border.all(1, T.BDR()), border_radius=8,
                        heading_row_height=28, data_row_min_height=24,
                    ),
                ], spacing=8)),
                # Lineage
                card(ft.Column([
                    ft.Text("Veri Soy Ağacı",
                            size=13, weight=ft.FontWeight.W_600, color=T.TEXT()),
                    ft.DataTable(
                        columns=[
                            ft.DataColumn(ft.Text("Saat",   size=11, color=T.SUB())),
                            ft.DataColumn(ft.Text("İşlem",  size=11, color=T.SUB())),
                            ft.DataColumn(ft.Text("Tablo",  size=11, color=T.SUB())),
                            ft.DataColumn(ft.Text("Satır",  size=11, color=T.SUB())),
                        ],
                        rows=lin_rows if lin_rows else [
                            ft.DataRow(cells=[
                                ft.DataCell(ft.Text("Log yok.", size=12, color=T.SUB())),
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                                ft.DataCell(ft.Text("")),
                            ])
                        ],
                        border=ft.border.all(1, T.BDR()), border_radius=8,
                        heading_row_height=28, data_row_min_height=24,
                    ),
                ], spacing=8)),
            ]
            self._status(
                self._status_t,
                f"Rapor oluşturuldu. Güven: {ts_score:.0f}/100", S,
            )
        except Exception as ex:
            self._status(self._status_t, f"Hata: {ex}", E)
        self.page.update()

    def _export_md(self, e):
        from core.report_writer import generate_markdown_report, TrustScore

        def _save(ev: ft.FilePickerResultEvent):
            if not ev.path: return
            try:
                records = self.audit.get_records()
                ba_stats = {}
                if (self.db.table_exists("ds_primary") and
                        self.db.table_exists("ds_original_p")):
                    orig = self.db.get_original("p")
                    curr = self.db.get_df("ds_primary")
                    if orig is not None:
                        ba_stats = self.db.compute_before_after_stats(
                            orig, curr
                        )
                corr = {}
                if self.db.table_exists("ds_primary"):
                    try:
                        corr = self.db.compute_correlation_matrix("ds_primary")
                    except Exception:
                        pass

                p_meta      = self.db.get_loaded_tables().get("ds_primary")
                col_profiles = (p_meta or {}).get("col_profiles")
                ts = self.trust_score or TrustScore()

                generate_markdown_report(
                    user_name     = self.user_name or "Kullanıcı",
                    audit_records = records,
                    before_after  = ba_stats,
                    spot_checks   = [],
                    correlation   = corr,
                    trust_score   = ts,
                    file_meta     = p_meta,
                    col_profiles  = col_profiles,
                    output_path   = ev.path,
                )
                self._snack(f"Rapor kaydedildi: {ev.path}", S)
            except Exception as ex:
                self._snack(f"Hata: {ex}", E)

        picker = ft.FilePicker(on_result=_save)
        self.page.overlay.append(picker)
        self.page.update()
        picker.save_file(
            dialog_title="Raporu Kaydet",
            file_name=f"data_autopsy_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
            allowed_extensions=["md"],
        )
