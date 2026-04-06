"""
main.py — Data-Autopsy v4.1

Düzeltmeler:
  • Tema: T.set_theme(dark) ile tüm renk sabitlerini günceller,
    sonra page.controls.clear() + _build_main() → gerçek tema değişimi
  • TrustScore: veri yüklenmeden önce sidebar'da gösterilmez;
    ilk dosya yüklendiğinde başlangıç skoru atanır
  • Büyük veri: column profile hesabı yükleme sonrasına defer edilir
  • Home sütun health listesi max 12 ile sınırlandı
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import flet as ft

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import ui.theme as T
T.set_theme(True)   # uygulama dark modda başlar

from core.i18n_manager import I18nManager
from core.database import DataAutopsyDB
from core.audit_logger import AuditLogger
from core.report_writer import TrustScore
from ui.views import (
    NormalizePanel, AuditPanel, MatchPanel, ImputePanel, ReportPanel,
    health_bar, tag, hr, card, info_box,
    P, S, W, E, IC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_autopsy")

APP_W, APP_H = 1400, 880
NAV_W        = 236


# ═══════════════════════════════════════════════════════════════════════════
# LoadingOverlay
# ═══════════════════════════════════════════════════════════════════════════

class LoadingOverlay:
    def __init__(self, page: ft.Page) -> None:
        self.page    = page
        self._title  = ft.Text(
            "İşleniyor...", size=15, weight=ft.FontWeight.W_600,
            color=ft.Colors.WHITE, text_align=ft.TextAlign.CENTER,
        )
        self._bar    = ft.ProgressBar(
            width=320, color=P, bgcolor="#2A2A44", value=None,
        )
        self._pct    = ft.Text(
            "", size=13, color=P, weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER,
        )
        self.control = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.55, "#000000"),
            visible=False,
            content=ft.Container(
                content=ft.Column([
                    ft.ProgressRing(color=P, width=48, height=48,
                                    stroke_width=4),
                    self._title, self._pct, self._bar,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                   spacing=12),
                bgcolor="#1A1A2E", border_radius=20,
                padding=ft.padding.symmetric(horizontal=52, vertical=40),
                border=ft.border.all(1, "#2A2A44"),
            ),
            alignment=ft.alignment.center,
        )

    def show(self, title_: str = "İşleniyor...") -> None:
        self._title.value = title_
        self._bar.value   = None
        self._pct.value   = ""
        self.control.visible = True
        self._safe_update()

    def set_progress(self, pct: float, msg: str = "") -> None:
        self._bar.value = max(0.0, min(1.0, pct / 100))
        self._pct.value = f"%{pct:.0f}"
        self._safe_update()

    def hide(self) -> None:
        self.control.visible = False
        self._safe_update()

    def run(self, title_: str, fn, on_done=None, on_error=None) -> None:
        self.show(title_)
        def _worker():
            try:
                result = fn(); self.hide()
                if on_done: on_done(result)
            except Exception as exc:
                logger.exception("Worker hatası")
                self.hide()
                if on_error: on_error(exc)
        threading.Thread(target=_worker, daemon=True).start()

    def _safe_update(self) -> None:
        try: self.page.update()
        except Exception: pass


# ═══════════════════════════════════════════════════════════════════════════
# Welcome Screen
# ═══════════════════════════════════════════════════════════════════════════

def build_welcome(page: ft.Page, on_submit) -> ft.Control:
    name_field = ft.TextField(
        hint_text="Adınızı girin...",
        border=ft.InputBorder.UNDERLINE,
        border_color="#3A3A66", focused_border_color=P,
        text_style=ft.TextStyle(color=ft.Colors.WHITE, size=22),
        hint_style=ft.TextStyle(color="#444466", size=22),
        text_align=ft.TextAlign.CENTER,
        cursor_color=P, width=340, autofocus=True,
    )
    err = ft.Text("", color=E, size=13, text_align=ft.TextAlign.CENTER)

    def _go(e=None):
        name = (name_field.value or "").strip()
        if not name:
            err.value = "Devam etmek için adınızı girin."
            page.update()
            return
        on_submit(name)
    name_field.on_submit = _go

    go_btn = ft.Container(
        content=ft.Text(
            "Otopsiye Başla →", size=15,
            color=ft.Colors.WHITE, weight=ft.FontWeight.W_600,
        ),
        on_click=_go, bgcolor=P, border_radius=50, ink=True,
        padding=ft.padding.symmetric(horizontal=48, vertical=16),
    )
    return ft.Container(
        expand=True,
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_center,
            end=ft.alignment.bottom_center,
            colors=["#06060E", "#0E0E20"],
        ),
        content=ft.Column([
            ft.Container(height=80),
            ft.Container(
                content=ft.Icon(ft.Icons.BIOTECH_OUTLINED, size=60, color=P),
                bgcolor="#13132A", border_radius=24, padding=20,
            ),
            ft.Container(height=28),
            ft.Text("Data-Vinci", size=44, weight=ft.FontWeight.BOLD,
                    color=ft.Colors.WHITE, text_align=ft.TextAlign.CENTER),
            ft.Text(
                "Veri Kalitesi & İstatistiksel Denetim",
                size=14, color="#6666AA", text_align=ft.TextAlign.CENTER,
            ),
            ft.Container(height=48),
            ft.Text("Adınızı girerek başlayın.",
                    size=13, color="#444466", text_align=ft.TextAlign.CENTER),
            ft.Container(height=12),
            name_field,
            ft.Container(height=4),
            err,
            ft.Container(height=24),
            go_btn,
            ft.Container(height=18),
            ft.Text("v4.1 — Optimizasyon", size=10, color="#1E1E33",
                    text_align=ft.TextAlign.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
        alignment=ft.alignment.center,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Home Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def build_home(
    user_name: str,
    p_meta: dict | None,
    s_meta: dict | None,
    navigate,
    on_remove,
    db: DataAutopsyDB,
    trust_score: TrustScore | None,
) -> ft.Control:
    hr_val   = time.localtime().tm_hour
    greeting = (
        "Günaydın"  if hr_val < 12 else
        "İyi günler" if hr_val < 18 else
        "İyi akşamlar"
    )
    date_str = time.strftime("%d %B %Y")

    def stat_card(val, lbl, ico, clr) -> ft.Container:
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Icon(ico, size=20, color=clr),
                    bgcolor=clr + "22", border_radius=11,
                    padding=11, width=44, height=44,
                    alignment=ft.alignment.center,
                ),
                ft.Column([
                    ft.Text(str(val), size=22, weight=ft.FontWeight.BOLD,
                            color=T.TEXT()),
                    ft.Text(lbl, size=11, color=T.SUB()),
                ], spacing=0),
            ], spacing=14),
            bgcolor=T.CARD(), border_radius=14,
            padding=ft.padding.symmetric(18, 16),
            border=ft.border.all(1, T.BDR()), expand=True,
        )

    def meta_mini(v, l, i):
        return ft.Column([
            ft.Icon(i, size=13, color=T.SUB()),
            ft.Text(str(v), size=13, weight=ft.FontWeight.W_600,
                    color=T.TEXT()),
            ft.Text(l, size=10, color=T.SUB()),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER,
           spacing=1, expand=True)

    def ds_card(label: str, meta: dict | None, which: str) -> ft.Container:
        if meta:
            miss = sum(
                1 for v in meta.get("missing_summary", {}).values()
                if v["missing_count"] > 0
            )
            anom = sum(
                1 for v in meta.get("col_health", {}).values()
                if v == "anomaly"
            )
            body = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.TABLE_CHART_OUTLINED, color=P, size=17),
                    ft.Text(
                        meta["file_name"], size=13,
                        weight=ft.FontWeight.W_600, color=T.TEXT(),
                        expand=True, overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Container(
                        content=ft.Text("Yüklendi", size=10, color=S),
                        bgcolor=S + "22", border_radius=4,
                        padding=ft.padding.symmetric(horizontal=8, vertical=3),
                    ),
                    ft.IconButton(
                        ft.Icons.CLOSE, icon_size=14, icon_color=E,
                        tooltip="Kaldır",
                        on_click=lambda e, w=which: on_remove(w),
                    ),
                ], spacing=6),
                hr(),
                ft.Row([
                    meta_mini(f"{meta['rows']:,}", "satır",
                              ft.Icons.STORAGE),
                    meta_mini(len(meta["columns"]), "sütun",
                              ft.Icons.VIEW_COLUMN),
                    meta_mini(f"{meta.get('file_size_mb', 0):.1f}MB", "boyut",
                              ft.Icons.FOLDER_OUTLINED),
                    meta_mini(meta.get("encoding", "—"), "encoding",
                              ft.Icons.CODE),
                ], spacing=4),
                ft.Row([
                    ft.Text(
                        f"{'🟡 ' + str(miss) + ' eksik' if miss else '🟢 Eksik yok'}",
                        size=11, color=W if miss else S,
                    ),
                    ft.Text("  |  ", size=11, color=T.SUB()),
                    ft.Text(
                        f"{'🔴 ' + str(anom) + ' anomali' if anom else '🟢 Temiz'}",
                        size=11, color=E if anom else S,
                    ),
                    ft.Text("  |  ", size=11, color=T.SUB()),
                    ft.Text(
                        f"Kalite: {meta.get('quality_score', 0):.0f}/100",
                        size=11,
                        color=S if meta.get("quality_score", 0) >= 70 else W,
                    ),
                ], spacing=0),
                *(
                    [ft.Text(
                        "  ".join(meta["format_warnings"][:2]),
                        size=10, color=W,
                    )]
                    if meta.get("format_warnings") else []
                ),
            ], spacing=6)
        else:
            body = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.UPLOAD_FILE_OUTLINED,
                            color=T.SUB(), size=17),
                    ft.Text(label, size=13, color=T.SUB()),
                ], spacing=8),
                ft.Text("Henüz yüklenmedi", size=12, color=T.BDR()),
            ], spacing=8)

        return ft.Container(
            content=body, bgcolor=T.CARD(), border_radius=14,
            padding=16, border=ft.border.all(1, T.BDR()), expand=True,
        )

    def action_card(t_, desc, ico, clr, pid) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(ico, size=20, color=clr),
                    bgcolor=clr + "22", border_radius=11,
                    padding=11, width=44, height=44,
                    alignment=ft.alignment.center,
                ),
                ft.Container(height=4),
                ft.Text(t_, size=13, weight=ft.FontWeight.W_600,
                        color=T.TEXT()),
                ft.Text(desc, size=11, color=T.SUB()),
            ], spacing=2),
            bgcolor=T.CARD(), border_radius=14, padding=16,
            border=ft.border.all(1, T.BDR()),
            on_click=lambda e, p=pid: navigate(p),
            ink=True, expand=True,
        )

    # Güven Skoru — sadece veri yüklendiğinde göster
    ts_widget = ft.Container()
    if trust_score is not None and p_meta is not None:
        ts_score = trust_score.score
        ts_clr   = S if ts_score >= 80 else W if ts_score >= 60 else E
        ts_widget = ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text("Veri Güven Skoru", size=11, color=T.SUB()),
                    ft.Text(f"{ts_score:.0f}/100", size=22,
                            weight=ft.FontWeight.BOLD, color=ts_clr),
                    ft.Text(trust_score.label(), size=11, color=ts_clr),
                ], spacing=1, expand=True),
                ft.ProgressBar(
                    value=ts_score / 100, color=ts_clr,
                    bgcolor=T.BDR(), height=8, border_radius=4, width=160,
                ),
            ], spacing=14,
               vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=T.CARD(), border_radius=12, padding=14,
            border=ft.border.only(left=ft.BorderSide(3, ts_clr)),
        )

    # Sütun sağlık haritası
    health_items: list[ft.Control] = []
    if p_meta:
        profiles   = p_meta.get("col_profiles", {})
        col_health = p_meta.get("col_health", {})
        items = sorted(
            p_meta.get("missing_summary", {}).items(),
            key=lambda x: x[1]["missing_pct"],
            reverse=True,
        )[:12]
        for col, info in items:
            pct     = info["missing_pct"]
            h       = col_health.get(col, "clean")
            qs      = profiles.get(col, {}).get("quality_score")
            profile = profiles.get(col, {})
            health_items.append(
                ft.Container(
                    content=health_bar(col, h, pct, qs),
                    on_click=lambda e, c=col, pr=profile: _col_popup(
                        c, pr, e.page
                    ),
                    border_radius=6, ink=True,
                    padding=ft.padding.symmetric(vertical=2, horizontal=4),
                )
            )

    total_rows = (
        (p_meta["rows"] if p_meta else 0) +
        (s_meta["rows"] if s_meta else 0)
    )
    n_ds = (1 if p_meta else 0) + (1 if s_meta else 0)

    header_row_items: list[ft.Control] = [
        ft.Column([
            ft.Text(f"{greeting}, {user_name} 👋", size=26,
                    weight=ft.FontWeight.BOLD, color=T.TEXT()),
            ft.Text(date_str, size=12, color=T.SUB()),
        ], spacing=2, expand=True),
    ]
    # Güven skoru sadece veri varsa sağ üste eklenir
    if p_meta is not None:
        header_row_items.append(ts_widget)

    return ft.Container(
        expand=True,
        content=ft.Column([
            ft.Row(header_row_items, spacing=16),
            ft.Container(height=20),
            ft.Row([
                stat_card(f"{total_rows:,}", "Toplam Satır",
                          ft.Icons.STORAGE, P),
                stat_card(str(n_ds), "Veri Seti",
                          ft.Icons.DATASET, "#4FC3F7"),
                stat_card(time.strftime("%H:%M"), "Oturum",
                          ft.Icons.ACCESS_TIME, "#81C784"),
            ], spacing=12),
            ft.Container(height=20),
            ft.Text("Veri Setleri", size=13,
                    weight=ft.FontWeight.W_600, color=T.SUB()),
            ft.Container(height=6),
            ft.Row([
                ds_card("Birincil Veri Seti", p_meta, "ds_primary"),
                ds_card("İkincil Veri Seti",  s_meta, "ds_secondary"),
            ], spacing=12),
            ft.Container(height=20),
            ft.Text("Hızlı Eylemler", size=13,
                    weight=ft.FontWeight.W_600, color=T.SUB()),
            ft.Container(height=6),
            ft.Row([
                action_card("Normalleştir", "Encoding & Türkçe",
                            ft.Icons.AUTO_FIX_HIGH, P, "normalize"),
                action_card("Denetle", "Benford & Normallik",
                            ft.Icons.ANALYTICS, "#4FC3F7", "audit"),
                action_card("Eşleştir", "Bulanık eşleştirme",
                            ft.Icons.COMPARE_ARROWS, "#FFB74D", "match"),
                action_card("Doldur", "Eksik veri",
                            ft.Icons.HEALING, "#81C784", "impute"),
                action_card("Rapor", "Kalite & Lineage",
                            ft.Icons.DESCRIPTION, "#CE93D8", "report"),
            ], spacing=10),
            ft.Container(height=20),
            ft.Container(
                content=ft.Column([
                    ft.Text(
                        "Sütun Sağlık Haritası"
                        + (" — Birincil Veri (tıkla → profil)"
                           if p_meta else ""),
                        size=13, weight=ft.FontWeight.W_600, color=T.SUB(),
                    ),
                    ft.Container(height=6),
                    *(
                        health_items if health_items else
                        [ft.Text("Veri yüklendikten sonra gösterilir.",
                                 size=12, color=T.SUB())]
                    ),
                ], spacing=6),
                bgcolor=T.CARD(), border_radius=14, padding=18,
                border=ft.border.all(1, T.BDR()),
            ),
        ], spacing=0, scroll=ft.ScrollMode.AUTO),
        padding=ft.padding.symmetric(horizontal=26, vertical=22),
    )


def _col_popup(col: str, profile: dict, page: ft.Page) -> None:
    """Sütun profil dialog."""
    stats = profile.get("stats", {})
    top5  = profile.get("top5",  [])
    miss  = profile.get("miss_pct", 0)
    qs    = profile.get("quality_score", 0)
    qs_clr = S if qs >= 80 else W if qs >= 50 else E

    top5_rows = [
        ft.DataRow(cells=[
            ft.DataCell(ft.Text(str(t["value"])[:40], size=11)),
            ft.DataCell(ft.Text(str(t["count"]), size=11, color=IC)),
            ft.DataCell(ft.Text(f"{t['pct']:.1f}%", size=11, color=T.SUB())),
        ])
        for t in top5
    ]
    stat_items = [
        ft.Row([
            ft.Text(k.replace("_", " ").title() + ":",
                    size=12, color=T.SUB(), width=80),
            ft.Text(str(v), size=12, color=T.TEXT()),
        ], spacing=6)
        for k, v in stats.items()
    ]

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text(f"Sütun: {col}", color=T.TEXT(), size=15,
                      weight=ft.FontWeight.BOLD),
        bgcolor=T.CARD(),
        content=ft.Container(
            content=ft.Column([
                ft.Row([
                    tag(profile.get("dtype", "—"), IC),
                    tag(f"%{miss:.1f} eksik", W if miss > 5 else S),
                    tag(f"{qs}/100", qs_clr),
                ], spacing=8, wrap=True),
                hr(),
                *(stat_items if stat_items else
                  [ft.Text("Sayısal sütun değil.", size=12, color=T.SUB())]),
                hr(),
                ft.Text("Top 5 Değer", size=12,
                        weight=ft.FontWeight.W_600, color=T.TEXT()),
                ft.DataTable(
                    columns=[
                        ft.DataColumn(ft.Text("Değer", size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("Adet",  size=11, color=T.SUB())),
                        ft.DataColumn(ft.Text("%",      size=11, color=T.SUB())),
                    ],
                    rows=top5_rows if top5_rows else [
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text("Veri yok", size=11,
                                                color=T.SUB())),
                            ft.DataCell(ft.Text("")),
                            ft.DataCell(ft.Text("")),
                        ])
                    ],
                    border=ft.border.all(1, T.BDR()), border_radius=8,
                    heading_row_height=26, data_row_min_height=22,
                ),
            ], spacing=8, width=460, scroll=ft.ScrollMode.AUTO),
            height=340,
        ),
        actions=[
            ft.TextButton(
                "Kapat",
                on_click=lambda e: (
                    setattr(dlg, "open", False), page.update()
                ),
            )
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.dialog = dlg
    dlg.open   = True
    page.update()


# ═══════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════

def main(page: ft.Page) -> None:
    i18n        = I18nManager(initial_language="tr")
    db          = DataAutopsyDB()
    audit       = AuditLogger(audit_dir=str(ROOT / "audits"))
    trust_score = TrustScore(initial=60.0)   # başlangıç skoru, veri yokken gösterilmez

    page.title   = "Data-Vinci"
    page.bgcolor = T.BG()
    page.padding = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width      = APP_W
    page.window.height     = APP_H
    page.window.min_width  = 1000
    page.window.min_height = 680

    loading = LoadingOverlay(page)

    state = {
        "user":       "",
        "panel":      "home",
        "p_meta":     None,
        "s_meta":     None,
        "dark":       True,
        "data_ready": False,  # veri yüklenene kadar trust skoru gizlenir
    }

    panels:      dict = {}
    pcontent:    dict = {}
    nav_btns:    dict = {}
    content_area      = ft.Container(expand=True, bgcolor=T.BG())
    load_status       = ft.Text("", size=10, color=T.SUB())

    # Sidebar trust score widget — sadece data_ready=True sonrası gösterilir
    _ts_container = ft.Container()   # placeholder, sonra doldurulur

    fp_prim: ft.FilePicker | None = None
    fp_sec:  ft.FilePicker | None = None

    # ── launch ──────────────────────────────────────────────────────────

    def _launch(name: str) -> None:
        state["user"] = name
        kw = dict(
            i18n=i18n, db=db, audit=audit, page=page,
            loading=loading, trust_score=trust_score,
        )
        panels["normalize"] = NormalizePanel(**kw)
        panels["audit"]     = AuditPanel    (**kw)
        panels["match"]     = MatchPanel    (**kw)
        panels["impute"]    = ImputePanel   (**kw)
        panels["report"]    = ReportPanel   (**kw, user_name=name)

        for pid, p in panels.items():
            pcontent[pid] = p.build()

        page.controls.clear()
        _build_main()
        _switch("home")
        page.update()

    # ── build main ──────────────────────────────────────────────────────

    def _build_main() -> None:
        nonlocal fp_prim, fp_sec, nav_btns, _ts_container

        nav_items = [
            ("home",      "Ana Sayfa",              ft.Icons.HOME_OUTLINED),
            ("normalize", i18n.t("menu_normalize"),  ft.Icons.AUTO_FIX_HIGH),
            ("audit",     i18n.t("menu_audit"),      ft.Icons.ANALYTICS_OUTLINED),
            ("match",     i18n.t("menu_match"),      ft.Icons.COMPARE_ARROWS),
            ("impute",    i18n.t("menu_impute"),     ft.Icons.HEALING),
            ("report",    i18n.t("menu_report"),     ft.Icons.DESCRIPTION_OUTLINED),
        ]
        nav_btns = {}
        nav_col  = []
        for pid, lbl, ico in nav_items:
            inner = ft.TextButton(
                content=ft.Row([
                    ft.Icon(ico, size=15),
                    ft.Text(lbl, size=13),
                ], spacing=10),
                on_click=lambda e, p=pid: _switch(p),
                style=ft.ButtonStyle(
                    color={
                        ft.ControlState.DEFAULT: T.SUB(),
                        ft.ControlState.HOVERED: T.TEXT(),
                    },
                    overlay_color=ft.Colors.with_opacity(0.07, P),
                    padding=ft.padding.symmetric(horizontal=14, vertical=11),
                    shape=ft.RoundedRectangleBorder(radius=8),
                ),
                expand=True,
            )
            c = ft.Container(content=inner, border_radius=8, data=pid)
            nav_btns[pid] = c
            nav_col.append(c)

        fp_prim = ft.FilePicker(on_result=_on_file(True))
        fp_sec  = ft.FilePicker(on_result=_on_file(False))
        page.overlay.extend([fp_prim, fp_sec])

        init      = state["user"][0].upper() if state["user"] else "?"
        theme_ico = ft.Icons.LIGHT_MODE if state["dark"] else ft.Icons.DARK_MODE
        theme_btn = ft.IconButton(
            theme_ico,
            icon_color=T.SUB(),
            on_click=lambda e: _toggle_theme(),
            tooltip="Tema",
        )
        lang_btn = ft.IconButton(
            ft.Icons.LANGUAGE,
            icon_color=T.SUB(),
            on_click=lambda e: _toggle_lang(),
            tooltip="TR / EN",
        )

        # Trust score — veri yüklenince visible olur
        _ts_container = ft.Container(visible=state["data_ready"])

        sidebar = ft.Container(
            width=NAV_W, bgcolor=T.NAV(),
            border=ft.border.only(right=ft.BorderSide(1, T.BDR())),
            content=ft.Column([
                # Logo
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.BIOTECH, color=P, size=18),
                        ft.Text("Data-Vinci", size=14,
                                weight=ft.FontWeight.BOLD, color=T.TEXT()),
                    ], spacing=8),
                    padding=ft.padding.only(
                        left=14, top=16, right=14, bottom=12
                    ),
                ),
                ft.Divider(color=T.BDR(), height=1),
                # Kullanıcı
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            content=ft.Text(
                                init, size=11,
                                color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD,
                            ),
                            bgcolor=P, border_radius=50,
                            width=26, height=26,
                            alignment=ft.alignment.center,
                        ),
                        ft.Text(state["user"], size=12, color=T.SUB()),
                    ], spacing=8),
                    padding=ft.padding.symmetric(horizontal=14, vertical=8),
                ),
                ft.Divider(color=T.BDR(), height=1),
                # Trust Score (gizli — veri yüklenince açılır)
                _ts_container,
                # Dosya yükleme
                ft.Container(
                    content=ft.Column([
                        ft.ElevatedButton(
                            "Birincil Veri",
                            icon=ft.Icons.UPLOAD_FILE,
                            on_click=lambda e: fp_prim.pick_files(
                                allowed_extensions=["csv", "xlsx", "xls"],
                                dialog_title="Birincil Veri Seti",
                            ),
                            style=ft.ButtonStyle(
                                bgcolor=P, color=ft.Colors.WHITE
                            ),
                            height=34,
                        ),
                        ft.ElevatedButton(
                            "İkincil Veri",
                            icon=ft.Icons.UPLOAD_FILE_OUTLINED,
                            on_click=lambda e: fp_sec.pick_files(
                                allowed_extensions=["csv", "xlsx", "xls"],
                                dialog_title="İkincil Veri Seti",
                            ),
                            style=ft.ButtonStyle(
                                bgcolor=T.CARD2(), color=T.SUB()
                            ),
                            height=34,
                        ),
                        load_status,
                    ], spacing=6),
                    padding=ft.padding.symmetric(horizontal=10, vertical=10),
                ),
                ft.Divider(color=T.BDR(), height=1),
                # Nav butonları
                ft.Container(
                    content=ft.Column(nav_col, spacing=2),
                    padding=ft.padding.symmetric(horizontal=6, vertical=8),
                    expand=True,
                ),
                ft.Divider(color=T.BDR(), height=1),
                ft.Container(
                    content=ft.Row(
                        [lang_btn, theme_btn],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    padding=ft.padding.symmetric(vertical=6),
                ),
                ft.Container(
                    content=ft.Text("v4.1", size=9, color=T.BDR(),
                                    text_align=ft.TextAlign.CENTER),
                    padding=ft.padding.only(bottom=8),
                    alignment=ft.alignment.center,
                ),
            ], spacing=0, expand=True),
        )

        content_area.bgcolor = T.BG()
        page.add(ft.Stack([
            ft.Row([sidebar, content_area], expand=True, spacing=0),
            loading.control,
        ], expand=True))

    # ── trust score sidebar güncelle ─────────────────────────────────────

    def _update_trust_widget() -> None:
        if not state["data_ready"]:
            return
        sc   = trust_score.score
        clr  = S if sc >= 80 else W if sc >= 60 else E
        _ts_container.visible = True
        _ts_container.content = ft.Column([
            ft.Divider(color=T.BDR(), height=1),
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("Güven Skoru", size=10, color=T.SUB(),
                                expand=True),
                        ft.Text(f"{sc:.0f}/100", size=10, color=clr),
                    ], spacing=4),
                    ft.ProgressBar(
                        value=sc / 100, color=clr,
                        bgcolor=T.BDR(), height=4, border_radius=2,
                    ),
                ], spacing=4),
                padding=ft.padding.symmetric(horizontal=12, vertical=8),
            ),
        ], spacing=0)
        try: page.update()
        except Exception: pass

    # ── dosya yükleme ─────────────────────────────────────────────────────

    def _on_file(is_prim: bool):
        def handler(ev: ft.FilePickerResultEvent) -> None:
            if not ev.files:
                return
            f     = ev.files[0]
            table = "ds_primary"   if is_prim else "ds_secondary"
            which = "Birincil"     if is_prim else "İkincil"

            def _load():
                return db.load_file(f.path, table_name=table)

            def _done(meta: dict) -> None:
                num_cols   = db.get_numeric_columns(table)
                all_cols   = meta["columns"]
                str_cols   = db.get_string_columns(table)
                col_health = meta.get("col_health", {})

                if is_prim:
                    # ── AMNESIA PROTOKOLÜ (Kesin Sıfırlama) ─────────────────
                    # Sadece backend'deki class'ları değil, arayüzdeki UI container'larını 
                    # da Flet motorunda temizlemeye zorlar.
                    for pid, panel in panels.items():
                        if hasattr(panel, 'hard_reset'):
                            panel.hard_reset()
                            # Panelleri yeni veriye göre temizle ve sıfırdan "build" et
                            pcontent[pid] = panel.build()
                            
                    # Pending state sıfırla
                    state["p_meta"]     = None
                    state["data_ready"] = False
                    trust_score._score  = 0.0
                    # ── YENİ META YAZ ──────────────────────────────────────
                    state["p_meta"]    = meta
                    state["data_ready"] = True
                    qs = meta.get("quality_score", 60)
                    trust_score._score = float(qs)
                    panels["normalize"].refresh_columns(
                        str_cols + [c for c in all_cols if c not in str_cols],
                        health=col_health,
                    )
                    panels["audit"].refresh_columns(num_cols, all_cols)
                    panels["match"].refresh_columns(all_cols)
                else:
                    state["s_meta"] = meta
                    panels["match"].refresh_columns(all_cols)

                _update_load_status()
                _update_trust_widget()
                audit.log_load(
                    str(f.path), meta["rows"], len(meta["columns"]),
                    meta["encoding"], meta["file_size_kb"],
                    meta["duration_ms"],
                )
                _refresh_home()

                warns = meta.get("format_warnings", [])
                msg   = (
                    f"✓ {meta['file_name']} yüklendi "
                    f"({meta['rows']:,} satır)"
                )
                if warns:
                    msg += f"  ⚠ {len(warns)} uyarı"
                page.snack_bar = ft.SnackBar(
                    ft.Text(msg, color=ft.Colors.WHITE),
                    bgcolor=W if warns else S,
                )
                page.snack_bar.open = True
                page.update()

            def _err(exc: Exception) -> None:
                _update_load_status()
                page.snack_bar = ft.SnackBar(
                    ft.Text(f"Yükleme hatası: {exc}", color=ft.Colors.WHITE),
                    bgcolor=E,
                )
                page.snack_bar.open = True
                page.update()

            loading.run(f"{which} veri yükleniyor...", _load, _done, _err)

        return handler

    def _update_load_status() -> None:
        p, s = state["p_meta"], state["s_meta"]
        load_status.value = (
            (f"B: {p['file_name']}" if p else "B: —") + "\n"
            + (f"İ: {s['file_name']}" if s else "İ: —")
        )
        load_status.color = S if p else T.SUB()
        try: page.update()
        except Exception: pass

    # ── dosya kaldır ──────────────────────────────────────────────────────

    def _remove_file(table_name: str) -> None:
        db.remove_file(table_name)
        if table_name == "ds_primary":
            state["p_meta"]    = None
            state["data_ready"] = False
            _ts_container.visible = False
            panels["normalize"].refresh_columns([], health={})
            panels["audit"].refresh_columns([], [])
        else:
            state["s_meta"] = None
        _update_load_status()
        _refresh_home()
        page.snack_bar = ft.SnackBar(
            ft.Text(f"{table_name} kaldırıldı.", color=ft.Colors.WHITE),
            bgcolor=W,
        )
        page.snack_bar.open = True
        page.update()

    # ── panel geçiş ───────────────────────────────────────────────────────

    def _refresh_home() -> None:
        pcontent["home"] = build_home(
            state["user"], state["p_meta"], state["s_meta"],
            _switch, _remove_file, db,
            trust_score if state["data_ready"] else None,
        )
        if state["panel"] == "home":
            content_area.content = pcontent["home"]
            try: page.update()
            except Exception: pass

    def _switch(pid: str) -> None:
        state["panel"] = pid
        if pid == "home":
            _refresh_home()
        else:
            content_area.content = pcontent.get(pid)
        for p, c in nav_btns.items():
            c.bgcolor = (
                ft.Colors.with_opacity(0.10, P) if p == pid else None
            )
        try: page.update()
        except Exception: pass

    # ── tema değişimi ─────────────────────────────────────────────────────

    def _toggle_theme() -> None:
        state["dark"] = not state["dark"]
        T.set_theme(state["dark"])           # TÜM renk fonksiyonları güncellenir
        page.theme_mode = (
            ft.ThemeMode.DARK if state["dark"] else ft.ThemeMode.LIGHT
        )
        page.bgcolor = T.BG()

        # Panelleri yeniden oluştur (yeni temayı yansıtmaları için)
        for pid, p in panels.items():
            pcontent[pid] = p.build()

        page.controls.clear()
        page.overlay.clear()
        _build_main()
        _switch(state["panel"])
        page.update()

    # ── dil değişimi ─────────────────────────────────────────────────────

    def _toggle_lang() -> None:
        i18n.toggle_language()
        for pid, p in panels.items():
            pcontent[pid] = p.build()
        page.controls.clear()
        page.overlay.clear()
        _build_main()
        _switch(state["panel"])
        page.update()

    # ── başlat ───────────────────────────────────────────────────────────

    page.add(build_welcome(page, _launch))
    page.update()


if __name__ == "__main__":
    ft.app(target=main, assets_dir=str(ROOT / "assets"))
