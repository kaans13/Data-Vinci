"""
state_manager.py — Data-Autopsy v5  Redux-Benzeri State Yönetimi

Problem
=======
`state = {"user": "", "p_meta": None, ...}` global sözlüğü:
  1. Race condition: iki thread aynı anda `state["p_meta"] = ...` yaparsa
     yarım yazılmış state okunabilir.
  2. Takip edilemez: state nerede, nasıl değişti bilgi yok.
  3. Tutarsızlık: `state["data_ready"]` ve `state["p_meta"]` birbirinden
     bağımsız güncellenebilir → p_meta=None ama data_ready=True kalabilir.

Çözüm: Redux Deseni
===================
  Redux'un 3 kuralı:
  1. Tek doğru kaynak (Single Source of Truth): tüm uygulama state'i tek yerde.
  2. State read-only: direkt değiştirilemez, sadece Action gönderilebilir.
  3. Reducer saf fonksiyon: (state, action) → new_state

  Flet'e adaptasyon:
  - State bir dataclass (tip güvenliği)
  - Reducer'lar önceden tanımlı, başka yerde state değiştirilemiyor
  - Subscriber'lar (UI callback'leri) state değişince otomatik tetiklenir
  - RLock ile thread-safe: `dispatch()` atomik

  Race condition neden ortadan kalkar?
  - `dispatch()` Lock alır → state oluşturulur → listener'lar çağrılır → Lock bırakılır
  - İki thread aynı anda dispatch etse bile sırayla işlenir, yarım yazma olmaz
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Uygulama State'i (Immutable Dataclass) ───────────────────────────────────

@dataclass(frozen=True)
class AppState:
    """
    frozen=True → doğrudan değiştirilemez.
    Değişim için: `replace(state, field=new_value)` kullanılır.
    Bu, Python'da en hafif immutability garantisi.
    """
    # Kullanıcı
    user_name:    str  = ""
    is_launched:  bool = False

    # Veri setleri
    p_meta:       dict | None = None    # birincil veri meta
    s_meta:       dict | None = None    # ikincil veri meta
    data_ready:   bool = False          # en az bir veri yüklendi

    # UI durumu
    active_panel: str  = "home"
    is_dark_theme: bool = True
    language:     str  = "tr"

    # Kalite
    trust_score:  float = 0.0           # 0 → veri yok, >0 → hesaplandı
    quality_grade: str  = ""            # A/B/C/D/F

    # Yükleme durumu
    is_loading:   bool = False
    loading_text: str  = ""

    # Hata
    last_error:   str  = ""


# ── Action Türleri ────────────────────────────────────────────────────────────

class ActionType(Enum):
    # Kullanıcı
    SET_USER         = auto()
    # Veri
    SET_PRIMARY      = auto()
    SET_SECONDARY    = auto()
    REMOVE_PRIMARY   = auto()
    REMOVE_SECONDARY = auto()
    # Tam sıfırlama (yeni dosya yüklendiğinde)
    HARD_RESET       = auto()
    # UI
    SET_PANEL        = auto()
    TOGGLE_THEME     = auto()
    TOGGLE_LANGUAGE  = auto()
    # Kalite
    UPDATE_TRUST     = auto()
    # Yükleme
    SET_LOADING      = auto()
    CLEAR_LOADING    = auto()
    # Hata
    SET_ERROR        = auto()
    CLEAR_ERROR      = auto()


@dataclass(frozen=True)
class Action:
    type:    ActionType
    payload: Any = None


# ── Reducer (Saf Fonksiyon) ───────────────────────────────────────────────────

def _reducer(state: AppState, action: Action) -> AppState:
    """
    (AppState, Action) → AppState

    Saf fonksiyon: dış state'e dokunmaz, her zaman yeni state döner.
    """
    t = action.type
    p = action.payload

    if t == ActionType.SET_USER:
        return replace(state, user_name=str(p), is_launched=True)

    elif t == ActionType.SET_PRIMARY:
        meta  = p or {}
        score = float(meta.get("quality_score", 0.0))
        grade = _quality_grade(score)
        return replace(
            state,
            p_meta=meta,
            data_ready=True,
            trust_score=score,
            quality_grade=grade,
            last_error="",
        )

    elif t == ActionType.SET_SECONDARY:
        return replace(state, s_meta=p)

    elif t == ActionType.REMOVE_PRIMARY:
        still_ready = state.s_meta is not None
        return replace(
            state,
            p_meta=None,
            data_ready=still_ready,
            trust_score=0.0,
            quality_grade="",
        )

    elif t == ActionType.REMOVE_SECONDARY:
        return replace(state, s_meta=None)

    elif t == ActionType.SET_PANEL:
        return replace(state, active_panel=str(p))

    elif t == ActionType.TOGGLE_THEME:
        return replace(state, is_dark_theme=not state.is_dark_theme)

    elif t == ActionType.TOGGLE_LANGUAGE:
        new_lang = "en" if state.language == "tr" else "tr"
        return replace(state, language=new_lang)

    elif t == ActionType.HARD_RESET:
        # Veri ve analiz sonuçlarını sıfırlar, kullanıcı ve UI ayarları korunur
        return replace(
            state,
            p_meta=None,
            s_meta=None,
            data_ready=False,
            trust_score=0.0,
            quality_grade="",
            is_loading=False,
            loading_text="",
            last_error="",
        )

    elif t == ActionType.UPDATE_TRUST:
        # payload: {"delta": float, "reason": str}
        delta    = float((p or {}).get("delta", 0.0))
        new_score = float(max(0.0, min(100.0, state.trust_score + delta)))
        return replace(
            state,
            trust_score=new_score,
            quality_grade=_quality_grade(new_score),
        )

    elif t == ActionType.SET_LOADING:
        text = str(p) if p else "İşleniyor..."
        return replace(state, is_loading=True, loading_text=text)

    elif t == ActionType.CLEAR_LOADING:
        return replace(state, is_loading=False, loading_text="")

    elif t == ActionType.SET_ERROR:
        return replace(state, last_error=str(p or ""))

    elif t == ActionType.CLEAR_ERROR:
        return replace(state, last_error="")

    return state   # bilinmeyen action → state değişmez


# ── StateManager ─────────────────────────────────────────────────────────────

class StateManager:
    """
    Thread-safe, Redux-benzeri merkezi state yöneticisi.

    Kullanım:
        store = StateManager()
        store.subscribe(lambda s: update_ui(s))
        store.dispatch(Action(ActionType.SET_USER, "Metin"))
        print(store.state.user_name)  # "Metin"

    Race condition güvencesi:
        dispatch() bir RLock içinde çalışır.
        İki thread aynı anda dispatch etseydi sırayla işlenirdi.
        State asla yarım güncellenmez.
    """

    def __init__(self, initial: AppState | None = None):
        self._state      = initial or AppState()
        self._lock       = threading.RLock()
        self._listeners: list[Callable[[AppState], None]] = []
        self._history:   list[tuple[Action, AppState]] = []
        self._max_history = 50

    # ── Temel API ────────────────────────────────────────────────────────

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    def dispatch(self, action: Action) -> AppState:
        """
        Action gönder, state güncelle, subscriber'ları bildir.
        Thread-safe — atomik.
        Returns: yeni state
        """
        with self._lock:
            old_state = self._state
            new_state = _reducer(old_state, action)

            if new_state is old_state:
                return old_state   # değişim yok → listener çağrılmaz

            self._state = new_state

            # Geçmiş kaydı (undo için)
            self._history.append((action, old_state))
            if len(self._history) > self._max_history:
                self._history.pop(0)

            logger.debug("[Store] %s → trust=%.1f panel=%s",
                         action.type.name,
                         new_state.trust_score,
                         new_state.active_panel)

        # Listener'lar lock dışında çağrılır → deadlock riski yok
        self._notify(new_state)
        return new_state

    def subscribe(self, listener: Callable[[AppState], None]) -> Callable:
        """
        State değişince çağrılacak callback ekler.
        Returns: unsubscribe fonksiyonu.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

        def unsubscribe():
            if listener in self._listeners:
                self._listeners.remove(listener)
        return unsubscribe

    def undo(self) -> AppState | None:
        """Son action'ı geri alır. History boşsa None döner."""
        with self._lock:
            if not self._history:
                return None
            _, prev_state = self._history.pop()
            self._state   = prev_state

        self._notify(prev_state)
        return prev_state

    # ── Kısayol dispatch'ler ─────────────────────────────────────────────
    # Tip güvenliği + kullanım kolaylığı için

    def hard_reset(self) -> AppState:
        """
        Amnesia protokolü: veri + analiz sonuçları tamamen silinir.
        Yeni dosya yüklendiğinde çağrılır.
        Kullanıcı adı, tema ve dil ayarları korunur.
        """
        return self.dispatch(Action(ActionType.HARD_RESET))

    def set_user(self, name: str) -> AppState:
        return self.dispatch(Action(ActionType.SET_USER, name))

    def set_primary(self, meta: dict) -> AppState:
        return self.dispatch(Action(ActionType.SET_PRIMARY, meta))

    def set_secondary(self, meta: dict) -> AppState:
        return self.dispatch(Action(ActionType.SET_SECONDARY, meta))

    def remove_primary(self) -> AppState:
        return self.dispatch(Action(ActionType.REMOVE_PRIMARY))

    def remove_secondary(self) -> AppState:
        return self.dispatch(Action(ActionType.REMOVE_SECONDARY))

    def set_panel(self, panel: str) -> AppState:
        return self.dispatch(Action(ActionType.SET_PANEL, panel))

    def toggle_theme(self) -> AppState:
        return self.dispatch(Action(ActionType.TOGGLE_THEME))

    def toggle_language(self) -> AppState:
        return self.dispatch(Action(ActionType.TOGGLE_LANGUAGE))

    def update_trust(self, delta: float, reason: str = "") -> AppState:
        return self.dispatch(Action(ActionType.UPDATE_TRUST,
                                    {"delta": delta, "reason": reason}))

    def set_loading(self, text: str = "") -> AppState:
        return self.dispatch(Action(ActionType.SET_LOADING, text))

    def clear_loading(self) -> AppState:
        return self.dispatch(Action(ActionType.CLEAR_LOADING))

    def set_error(self, msg: str) -> AppState:
        return self.dispatch(Action(ActionType.SET_ERROR, msg))

    def clear_error(self) -> AppState:
        return self.dispatch(Action(ActionType.CLEAR_ERROR))

    # ── Debug ────────────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        with self._lock:
            return [
                {"action": a.type.name, "payload": str(a.payload)[:60]}
                for a, _ in self._history
            ]

    # ── Private ──────────────────────────────────────────────────────────

    def _notify(self, state: AppState) -> None:
        for listener in list(self._listeners):
            try:
                listener(state)
            except Exception as exc:
                logger.error("Listener hatası: %s", exc)


def _quality_grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    if score > 0:   return "F"
    return ""   # veri yok
