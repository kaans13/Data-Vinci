"""
audit_logger.py — Data-Autopsy v3

Yenilikler:
  - SafeJSONEncoder entegrasyonu (numpy/bool/NaN hepsi güvenli)
  - Her işlemde effect_size (mean_delta, var_delta, cohen_d) logu
  - Thread-safe JSONL yazımı (threading.Lock)
  - ds_processed immutability: orijinal/işlenmiş tablo farkı loglanır
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Encoder — tüm NumPy/NaN/bool tipleri güvenli
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursive sanitizer: NaN/Inf/numpy → JSON-safe Python."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v or abs(v) == float("inf")) else v
    if isinstance(obj, np.bool_):        return bool(obj)
    if isinstance(obj, np.ndarray):      return [_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):    return obj.isoformat()
    if isinstance(obj, dict):            return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [_sanitize(i) for i in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Drop-in JSONEncoder; delegates to _sanitize for non-standard types."""
    def default(self, obj: Any) -> Any:
        sanitized = _sanitize(obj)
        if sanitized is not obj:
            return sanitized
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(_sanitize(obj), _one_shot)


def safe_dumps(obj: Any, **kw) -> str:
    return json.dumps(_sanitize(obj), cls=NumpyEncoder, ensure_ascii=False, **kw)


# ---------------------------------------------------------------------------
# Effect Size — işlem öncesi/sonrası istatistik karşılaştırması
# ---------------------------------------------------------------------------

def compute_effect_size(
    before: pd.Series,
    after: pd.Series,
) -> dict:
    """
    İki seri arasındaki etki büyüklüğünü hesaplar.
    Kullanım: normalizasyon veya imputation sonrası çağrılır.

    Returns dict:
      mean_before, mean_after, mean_delta_pct
      var_before,  var_after,  var_delta_pct
      cohen_d   — standartlaşmış ortalama farkı
    """
    try:
        b_num = pd.to_numeric(before, errors="coerce").dropna()
        a_num = pd.to_numeric(after,  errors="coerce").dropna()
        if len(b_num) < 2 or len(a_num) < 2:
            return {"effect_available": False}

        mb, ma = float(b_num.mean()), float(a_num.mean())
        vb, va = float(b_num.var()),  float(a_num.var())

        # Pooled std için Welch yaklaşımı
        pooled_std = float(np.sqrt((b_num.std() ** 2 + a_num.std() ** 2) / 2))
        cohen_d    = float((ma - mb) / pooled_std) if pooled_std > 0 else 0.0

        def _pct_delta(b: float, a: float) -> float | None:
            if b == 0: return None
            return round((a - b) / abs(b) * 100, 2)

        return _sanitize({
            "effect_available":  True,
            "mean_before":       round(mb, 4),
            "mean_after":        round(ma, 4),
            "mean_delta_pct":    _pct_delta(mb, ma),
            "var_before":        round(vb, 4),
            "var_after":         round(va, 4),
            "var_delta_pct":     _pct_delta(vb, va),
            "cohen_d":           round(cohen_d, 4),
            "cohen_interpretation": _cohen_label(abs(cohen_d)),
        })
    except Exception as exc:
        return {"effect_available": False, "error": str(exc)}


def _cohen_label(d: float) -> str:
    if d < 0.2:   return "Ihmal Edilebilir"
    if d < 0.5:   return "Küçük"
    if d < 0.8:   return "Orta"
    return "Büyük"


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Thread-safe JSONL audit logger.
    Her işlemi effect size ile birlikte kaydeder.
    """

    def __init__(self, audit_dir: str = "audits") -> None:
        self.audit_dir     = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.session_id    = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc)
        self._records:  list[dict] = []
        self._lock      = threading.Lock()

        ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.audit_dir / f"session_{ts}_{self.session_id[:8]}.jsonl"
        self._write({"type": "SESSION_START", "session_id": self.session_id,
                     "timestamp": self.session_start.isoformat(),
                     "tool": "Data-Autopsy", "version": "3.0"})

    # ------------------------------------------------------------------
    # Core log
    # ------------------------------------------------------------------

    def log(
        self,
        operation:      str,
        module:         str,
        status:         str = "SUCCESS",
        message:        str = "",
        input_summary:  dict | None = None,
        output_summary: dict | None = None,
        parameters:     dict | None = None,
        duration_ms:    float = 0.0,
        effect_size:    dict | None = None,
        data_sample:    Any = None,
    ) -> dict:
        record = _sanitize({
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "session_id":     self.session_id,
            "record_id":      str(uuid.uuid4()),
            "operation":      operation.upper(),
            "module":         module,
            "status":         status.upper(),
            "message":        message,
            "input_summary":  input_summary  or {},
            "output_summary": output_summary or {},
            "parameters":     parameters     or {},
            "duration_ms":    round(float(duration_ms), 3),
            "effect_size":    effect_size    or {},
        })
        if data_sample is not None:
            record["data_integrity_hash"] = self._hash(data_sample)

        with self._lock:
            self._records.append(record)
        self._write(record)

        lvl = logging.ERROR if status == "ERROR" else (
              logging.WARNING if status == "WARNING" else logging.INFO)
        logger.log(lvl, "[%s] %s — %s", operation, module, message[:120])
        return record

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def log_load(self, file_path, rows, columns, encoding, file_size_kb, duration_ms=0) -> dict:
        return self.log(
            operation="LOAD", module="database",
            message=f"Yüklendi: {Path(file_path).name}",
            input_summary={"file_path": str(file_path), "file_size_kb": round(float(file_size_kb), 2)},
            output_summary={"rows": int(rows), "columns": int(columns), "encoding": str(encoding)},
            duration_ms=duration_ms,
        )

    def log_normalize(
        self, column, operation_type, changes_count, total_rows,
        duration_ms=0, before: pd.Series | None = None, after: pd.Series | None = None
    ) -> dict:
        eff = compute_effect_size(before, after) if before is not None and after is not None else {}
        return self.log(
            operation="NORMALIZE", module="normalizer",
            message=f"'{column}' → {operation_type}: {changes_count}/{total_rows} değişti",
            input_summary={"column": str(column), "total_rows": int(total_rows)},
            output_summary={"changes_count": int(changes_count),
                            "change_rate_pct": round(int(changes_count)/max(int(total_rows),1)*100, 2)},
            parameters={"operation_type": str(operation_type)},
            duration_ms=duration_ms,
            effect_size=eff,
        )

    def log_benford(self, column, chi_square, p_value, is_suspicious, duration_ms=0) -> dict:
        return self.log(
            operation="AUDIT", module="statistical_auditor.benford",
            message=f"Benford '{column}': {'ŞÜPHELİ' if is_suspicious else 'Normal'}",
            output_summary={"chi_square": round(float(chi_square), 4),
                            "p_value": round(float(p_value), 4),
                            "is_suspicious": bool(is_suspicious)},
            parameters={"column": str(column), "alpha": 0.05},
            duration_ms=duration_ms,
        )

    def log_outlier(
        self, column, method, threshold, outlier_count, total_rows,
        duration_ms=0, normality_p=None
    ) -> dict:
        return self.log(
            operation="AUDIT", module="statistical_auditor.outlier",
            message=f"Aykırı '{column}' ({method}): {outlier_count}/{total_rows}",
            output_summary={"outlier_count": int(outlier_count),
                            "outlier_rate_pct": round(int(outlier_count)/max(int(total_rows),1)*100, 2),
                            "normality_p": float(normality_p) if normality_p is not None else None},
            parameters={"column": str(column), "method": str(method), "threshold": float(threshold)},
            duration_ms=duration_ms,
        )

    def log_match(
        self, col_p, col_s, algorithm, threshold,
        matched, unmatched_p, unmatched_s, duration_ms=0
    ) -> dict:
        return self.log(
            operation="MATCH", module="fuzzy_matcher",
            message=f"Eşleştirme ({algorithm}): {matched} çift",
            output_summary={"matched_pairs": int(matched),
                            "unmatched_primary": int(unmatched_p),
                            "unmatched_secondary": int(unmatched_s)},
            parameters={"col_primary": str(col_p), "col_secondary": str(col_s),
                        "algorithm": str(algorithm), "threshold": float(threshold)},
            duration_ms=duration_ms,
        )

    def log_impute(
        self, column, missing_pattern, method, cells_filled, duration_ms=0,
        before: pd.Series | None = None, after: pd.Series | None = None
    ) -> dict:
        eff = compute_effect_size(before, after) if before is not None and after is not None else {}
        return self.log(
            operation="IMPUTE", module="smart_imputer",
            message=f"Doldurma '{column}' ({method}): {cells_filled} hücre",
            output_summary={"cells_filled": int(cells_filled)},
            parameters={"column": str(column), "missing_pattern": str(missing_pattern),
                        "method": str(method)},
            duration_ms=duration_ms,
            effect_size=eff,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self) -> dict:
        with self._lock:
            records = list(self._records)

        end     = datetime.now(timezone.utc)
        by_op   = {}
        for r in records:
            by_op.setdefault(r.get("operation", "?"), []).append(r)

        # Effect size aggregation
        eff_summary = {}
        for op, recs in by_op.items():
            effects = [r.get("effect_size", {}) for r in recs if r.get("effect_size", {}).get("effect_available")]
            if effects:
                avg_cd = float(np.mean([abs(e.get("cohen_d", 0)) for e in effects]))
                eff_summary[op] = {"avg_cohen_d": round(avg_cd, 4),
                                   "interpretation": _cohen_label(avg_cd)}

        report = _sanitize({
            "report_metadata": {
                "report_id":              str(uuid.uuid4()),
                "generated_at":           end.isoformat(),
                "session_id":             self.session_id,
                "session_start":          self.session_start.isoformat(),
                "session_duration_sec":   round((end - self.session_start).total_seconds(), 2),
                "tool": "Data-Autopsy", "version": "3.0",
            },
            "summary": {
                "total_operations":    len(records),
                "ops_by_type":         {k: len(v) for k, v in by_op.items()},
                "success_count":       sum(1 for r in records if r.get("status") == "SUCCESS"),
                "warning_count":       sum(1 for r in records if r.get("status") == "WARNING"),
                "error_count":         sum(1 for r in records if r.get("status") == "ERROR"),
                "quality_score":       self._quality_score(records),
                "effect_size_summary": eff_summary,
            },
            "operations":        by_op,
            "audit_trail_file":  str(self.log_file),
        })

        out = self.audit_dir / f"report_{self.session_id[:8]}.json"
        out.write_text(safe_dumps(report, indent=2), encoding="utf-8")
        logger.info("Rapor oluşturuldu: %s", out)
        return report

    def get_records(self, operation: str | None = None) -> list[dict]:
        with self._lock:
            recs = list(self._records)
        if operation:
            return [r for r in recs if r.get("operation") == operation.upper()]
        return recs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(safe_dumps(record) + "\n")
        except OSError as e:
            logger.error("Log yazma hatası: %s", e)

    def _hash(self, data: Any) -> str:
        try:
            return hashlib.sha256(safe_dumps(data).encode()).hexdigest()
        except Exception:
            return "hash_error"

    def _quality_score(self, records: list[dict]) -> float:
        if not records:
            return 100.0
        err  = sum(1 for r in records if r.get("status") == "ERROR")
        warn = sum(1 for r in records if r.get("status") == "WARNING")
        return max(0.0, min(100.0, round(100.0 - err * 10 - warn * 3, 1)))
