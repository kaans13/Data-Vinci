"""
event_store.py — Data-Autopsy v5  Event Sourcing & Rollback

Mimari
======
Problem: Şu an her işlem DataFrame'i değiştiriyor. Rollback yok.

Çözüm: Command-Event ayrımı + lazy materializasyon.

  ┌─ Kavramlar ───────────────────────────────────────────────────────────
  │  Command   : "normalize_column('ad', ops=['strip'])" — niyet
  │  Event     : Uygulanan komutun kaydı (zaman damgası, versiyon, meta)
  │  Snapshot  : Belirli bir andaki DataFrame (checkpoint)
  │  Replay    : Event zincirini baştan oynat → o ana ait veriyi üret
  │
  │  Neden Event Sourcing?
  │  • Asıl veri asla değişmez → tutarlılık garantisi
  │  • Herhangi bir önceki adıma "sıfır veri kopyası" ile geri dönülür
  │  • Denetlenebilirlik: "Ne zaman, kim, ne yaptı?" tam izlenebilir
  │  • Lazy: Sadece ihtiyaç duyulduğunda materialize edilir
  └───────────────────────────────────────────────────────────────────────

  Replay maliyeti: O(N_events × row_cost)
  Snapshot cache: Her K=10 event'te otomatik snapshot alır → replay O(K)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL = 10   # Her 10 event'te bir snapshot


# ── Event türleri ────────────────────────────────────────────────────────────

class EventType(str, Enum):
    LOAD       = "LOAD"
    NORMALIZE  = "NORMALIZE"
    IMPUTE     = "IMPUTE"
    DROP_ROWS  = "DROP_ROWS"
    DROP_COLS  = "DROP_COLS"
    TYPE_CAST  = "TYPE_CAST"
    CUSTOM     = "CUSTOM"


@dataclass(frozen=True)
class DataEvent:
    """
    Immutable event kaydı.
    frozen=True → yanlışlıkla değiştirilemez, dict key olarak kullanılabilir.
    """
    event_id:   str
    event_type: EventType
    timestamp:  float
    version:    int         # bu event sonrası versiyon numarası
    params:     dict        # işlem parametreleri (sütun adı, operasyonlar vs.)
    user:       str
    source_hash: str        # önceki verinin md5 hash'i (zincir bütünlüğü)
    description: str        # insan okunabilir açıklama

    def to_dict(self) -> dict:
        return {
            "event_id":    self.event_id,
            "event_type":  self.event_type.value,
            "timestamp":   self.timestamp,
            "version":     self.version,
            "params":      self.params,
            "user":        self.user,
            "source_hash": self.source_hash,
            "description": self.description,
        }


@dataclass
class Snapshot:
    version: int
    df:      pd.DataFrame
    taken_at: float = field(default_factory=time.time)


# ── Transformasyon fonksiyonları (saf fonksiyonlar) ──────────────────────────
# Her fonksiyon: (df, params) → pd.DataFrame
# Saf (pure) — dış state'e dokunmaz, aynı girdi → aynı çıktı

def _apply_normalize(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from modules.normalizer import normalize_dataframe
    result, _ = normalize_dataframe(
        df,
        columns=params.get("columns"),
        operations=params.get("operations", []),
    )
    return result


def _apply_impute(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    from modules.smart_imputer import impute_column, ImputationMethod
    col    = params["column"]
    method = ImputationMethod(params["method"])
    df2    = df.copy()
    filled, _ = impute_column(df2[col], method)
    df2[col]  = filled
    return df2


def _apply_drop_rows(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    indices = params.get("indices", [])
    return df.drop(index=indices, errors="ignore").reset_index(drop=True)


def _apply_drop_cols(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    cols = params.get("columns", [])
    return df.drop(columns=cols, errors="ignore")


def _apply_type_cast(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df2  = df.copy()
    col  = params["column"]
    to   = params["to_type"]
    df2[col] = pd.to_numeric(df2[col], errors="coerce") if to == "numeric" \
               else df2[col].astype(str)
    return df2


def _apply_custom(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    fn: Callable = params.get("fn")
    if fn:
        return fn(df)
    return df


# Dispatch tablosu
_HANDLERS: dict[EventType, Callable] = {
    EventType.NORMALIZE: _apply_normalize,
    EventType.IMPUTE:    _apply_impute,
    EventType.DROP_ROWS: _apply_drop_rows,
    EventType.DROP_COLS: _apply_drop_cols,
    EventType.TYPE_CAST: _apply_type_cast,
    EventType.CUSTOM:    _apply_custom,
}


# ── EventStore ────────────────────────────────────────────────────────────────

class EventStore:
    """
    Merkezi event deposu.

    - Tüm operasyonlar event olarak kaydedilir, asıl veri değişmez.
    - materialize(version) → o versiyondaki DataFrame'i döner.
    - rollback(n) → son n event'i geri alır.
    - Snapshot cache ile tekrar replay'i hızlandırır.
    - Thread-safe (RLock).
    """

    def __init__(self, origin: pd.DataFrame, user: str = "user",
                 snapshot_interval: int = SNAPSHOT_INTERVAL):
        self._lock      = threading.RLock()
        self._origin    = origin.copy()         # asla değişmez
        self._events:  list[DataEvent] = []
        self._snapshots: list[Snapshot] = [
            Snapshot(version=0, df=origin.copy())
        ]
        self._interval  = snapshot_interval
        self._user      = user
        self._version   = 0

    # ── Genel bilgi ──────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def event_count(self) -> int:
        return len(self._events)

    def can_undo(self) -> bool:
        return self._version > 0

    def get_events(self) -> list[DataEvent]:
        with self._lock:
            return list(self._events)

    def get_lineage(self) -> list[dict]:
        """Tüm event zincirini insan okunabilir formatta döner."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    # ── Komut uygula ─────────────────────────────────────────────────────

    def apply(
        self,
        event_type: EventType,
        params: dict,
        description: str = "",
    ) -> tuple[DataEvent, pd.DataFrame]:
        """
        Yeni bir event ekler ve materialized DataFrame'i döner.
        Returns: (event, new_dataframe)
        """
        with self._lock:
            # Mevcut durumun hash'i (zincir bütünlüğü için)
            current_df  = self._materialize_locked(self._version)
            source_hash = _df_hash(current_df)

            new_version = self._version + 1
            event = DataEvent(
                event_id    = str(uuid.uuid4())[:8],
                event_type  = event_type,
                timestamp   = time.time(),
                version     = new_version,
                params      = _safe_params(params),
                user        = self._user,
                source_hash = source_hash,
                description = description or f"{event_type.value} v{new_version}",
            )

            # Transformasyonu uygula
            handler = _HANDLERS.get(event_type)
            if handler is None:
                raise ValueError(f"Bilinmeyen event türü: {event_type}")

            new_df = handler(current_df, params)
            self._events.append(event)
            self._version = new_version

            # Periyodik snapshot
            if new_version % self._interval == 0:
                self._snapshots.append(
                    Snapshot(version=new_version, df=new_df.copy())
                )
                logger.debug("Snapshot alındı: v%d", new_version)

            logger.info("[EventStore] %s v%d — %s",
                        event_type.value, new_version, description[:60])
            return event, new_df

    # ── Rollback ─────────────────────────────────────────────────────────

    def rollback(self, steps: int = 1) -> pd.DataFrame:
        """
        Son `steps` event'i geri alır.
        Returns: hedef versiyondaki DataFrame.
        """
        with self._lock:
            if steps <= 0:
                raise ValueError("steps > 0 olmalı")
            target_version = max(0, self._version - steps)

            # Event'leri kaldır
            self._events = [e for e in self._events
                            if e.version <= target_version]
            # Snapshot'ları temizle
            self._snapshots = [s for s in self._snapshots
                               if s.version <= target_version]
            self._version = target_version

            result = self._materialize_locked(target_version)
            logger.info("[EventStore] Rollback → v%d", target_version)
            return result

    def rollback_to_origin(self) -> pd.DataFrame:
        """Tamamen sıfırla."""
        with self._lock:
            self._events.clear()
            self._snapshots = [Snapshot(version=0, df=self._origin.copy())]
            self._version   = 0
            return self._origin.copy()

    # ── Materializasyon ───────────────────────────────────────────────────

    def materialize(self, version: int | None = None) -> pd.DataFrame:
        with self._lock:
            return self._materialize_locked(version or self._version)

    def _materialize_locked(self, target: int) -> pd.DataFrame:
        """
        En yakın snapshot'tan replay ederek hedef versiyonu üretir.
        Kilit dışarıda alınmış olmalı.
        """
        # En yakın snapshot'u bul
        best_snap = max(
            (s for s in self._snapshots if s.version <= target),
            key=lambda s: s.version,
            default=Snapshot(version=0, df=self._origin.copy()),
        )

        df = best_snap.df.copy()

        # Snapshot'tan sonraki event'leri replay et
        for event in self._events:
            if event.version <= best_snap.version:
                continue
            if event.version > target:
                break
            handler = _HANDLERS.get(event.event_type)
            if handler:
                try:
                    df = handler(df, event.params)
                except Exception as exc:
                    logger.error("Replay hatası event %s: %s",
                                 event.event_id, exc)
        return df

    # ── JSONL dışa aktarım ────────────────────────────────────────────────

    def export_log(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ev in self._events:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Event log yazıldı: %s", path)


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _df_hash(df: pd.DataFrame) -> str:
    """DataFrame'in hızlı md5 hash'i (bütünlük kontrolü için)."""
    try:
        h = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes(),
            usedforsecurity=False,
        ).hexdigest()
        return h[:12]
    except Exception:
        return "hash_err"


def _safe_params(params: dict) -> dict:
    """Params dict'ini JSON-safe hale getirir (callable'lar hariç)."""
    safe = {}
    for k, v in params.items():
        if callable(v):
            safe[k] = f"<callable:{v.__name__}>"
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe
