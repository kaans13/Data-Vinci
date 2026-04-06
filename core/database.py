"""
database.py — Data-Autopsy v4  "Bulletproof" Veri Motoru

Yenilikler v4:
  ┌─ OKUMA ─────────────────────────────────────────────────────────────────
  │  • charset-normalizer ile encoding tespiti (UTF-8, ISO-8859-9, Windows-1254…)
  │  • Otomatik delimiter tespiti (,  ;  \\t  |) — sniffer + sayım fallback
  │  • Büyük dosyalar (>10 MB) → DuckDB read_csv_auto (bellek dostu)
  │  • on_bad_lines='skip' — bozuk satırlar sessizce atlanır
  │  • Türk usulü sayı formatı ("1.250,50") → float dönüşümü + uyarı log
  │  • 3 aşamalı hata yakalamalı fallback zinciri
  └─────────────────────────────────────────────────────────────────────────
  ┌─ VERİ YÖNETIMI ──────────────────────────────────────────────────────────
  │  • ds_primary / ds_secondary SALT OKUNUR — asla değiştirilmez
  │  • ds_result_<isim>_v<N>  versiyonlama — snapshot zinciri
  │  • Data lineage: her snapshot neyin üstüne yapıldığını bilir
  │  • Sütun profil önbelleği (profil_cache) — tekrar hesaplamayı önler
  └─────────────────────────────────────────────────────────────────────────
  ┌─ JSON GÜVENLİĞİ ────────────────────────────────────────────────────────
  │  • _sanitize() recursive: np.nan/inf → None, int64→int, float64→float
  │  • safe_dumps() wrapper her log çağrısında kullanılır
  └─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Encoding kütüphaneleri ───────────────────────────────────────────────────
try:
    from charset_normalizer import from_bytes as _cn_from_bytes
    _CN = True
except ImportError:
    _CN = False

try:
    import chardet
    _CHARDET = True
except ImportError:
    _CHARDET = False

ENCODING_CHAIN = [
    "utf-8", "utf-8-sig", "windows-1254", "iso-8859-9", "cp857",
    "cp1252", "latin-1",
]

LARGE_FILE_THRESHOLD_MB = 10.0   # Bu boyutun üstü DuckDB ile yüklenir


# ═══════════════════════════════════════════════════════════════════════════
# JSON Güvenliği — tüm np.* tipler + NaN/Inf
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize(obj: Any) -> Any:
    """Recursive sanitizer: herhangi bir nesneyi JSON-safe Python'a çevirir."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (v != v or abs(v) == float("inf")) else v
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return [_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, dict):         return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_sanitize(i) for i in obj]
    return obj


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        return _sanitize(obj)

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(_sanitize(obj), _one_shot)


def safe_dumps(obj: Any, **kw) -> str:
    return json.dumps(_sanitize(obj), cls=_SafeEncoder, ensure_ascii=False, **kw)


# alias — geriye dönük uyumluluk
to_python_native = _sanitize
safe_json_dumps  = safe_dumps


# ═══════════════════════════════════════════════════════════════════════════
# Türk Usulü Sayı Formatı Düzeltici
# ═══════════════════════════════════════════════════════════════════════════

_TR_NUMBER_RE = re.compile(r'^\s*-?\d{1,3}(?:\.\d{3})*,\d+\s*$')
_TR_INT_RE    = re.compile(r'^\s*-?\d{1,3}(?:\.\d{3})+\s*$')


def _parse_tr_number(val: str) -> float | None:
    """
    Türk usulü sayı formatını float'a çevirir.
      "1.250,50"  → 1250.50
      "1.000.000" → 1000000
      Başarısız olursa None döner.
    """
    val = val.strip()
    if _TR_NUMBER_RE.match(val):
        return float(val.replace(".", "").replace(",", "."))
    if _TR_INT_RE.match(val):
        return float(val.replace(".", ""))
    return None


def fix_turkish_numbers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    DataFrame'deki object sütunlarda Türk usulü sayı formatlarını düzeltir.
    Returns (düzeltilmiş_df, toplam_düzeltilen_hücre_sayısı)
    """
    df = df.copy()
    total_fixed = 0
    for col in df.columns:
        if not (df[col].dtype == object
                or pd.api.types.is_string_dtype(df[col])
                or "string" in str(df[col].dtype).lower()):
            continue
        sample = df[col].dropna().astype(str).head(50)
        tr_hits = sample.apply(
            lambda x: _parse_tr_number(x) is not None
        ).sum()
        if tr_hits < len(sample) * 0.3:   # < %30 uyum → bu sütun sayısal değil
            continue
        def _convert(x):
            if pd.isna(x): return x
            v = _parse_tr_number(str(x))
            return v if v is not None else x
        old_strs = df[col].astype(str).copy()
        df[col] = df[col].apply(_convert)
        fixed = (old_strs != df[col].astype(str)).sum()
        if fixed > 0:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
            total_fixed += int(fixed)
            logger.info("Türk sayı formatı düzeltildi: %s (%d hücre)", col, fixed)
    return df, total_fixed


# ═══════════════════════════════════════════════════════════════════════════
# DataAutopsyDB
# ═══════════════════════════════════════════════════════════════════════════

class DataAutopsyDB:
    """
    Bulletproof DuckDB veri motoru.

    Tablo isimlendirme:
      ds_primary              — yüklenen birincil veri (SALT OKUNUR)
      ds_secondary            — yüklenen ikincil veri  (SALT OKUNUR)
      ds_original_p/s         — otomatik backup
      ds_result_<ad>_v<N>     — işlem sonuçları + versiyonlama
    """

    IMMUTABLE = frozenset(["ds_primary", "ds_secondary",
                           "ds_original_p", "ds_original_s"])

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn    = duckdb.connect(db_path)
        try:
            self.conn.execute("PRAGMA threads=4")
        except Exception:
            pass
        self._loaded: dict[str, dict]  = {}
        self._registry: list[dict]     = []
        self._lineage:  list[dict]     = []   # Data lineage zinciri
        self._version:  dict[str, int] = {}   # result_name → versiyon sayacı
        self._profile_cache: dict[str, dict] = {}
        logger.info("DuckDB hazır: %s", db_path)

    # ──────────────────────────────────────────────────────────────────────
    # Dosya Yükleme — Bulletproof 3-aşama
    # ──────────────────────────────────────────────────────────────────────

    def load_file(
        self,
        file_path: str | Path,
        table_name: str = "ds_primary",
        encoding: str | None = None,
        sheet_name: int | str = 0,
    ) -> dict:
        """
        CSV/Excel yükler. 3 aşamalı hata yakalamalı zincir:
          1. DuckDB read_csv_auto (büyük dosyalar)
          2. pd.read_csv (encoding + sniffer)
          3. pd.read_csv (sep=None, engine=python fallback)
        """
        t0   = time.perf_counter()
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        size_mb       = path.stat().st_size / 1_048_576
        suffix        = path.suffix.lower()
        detected_enc  = encoding or (self._detect_encoding(path) if suffix == ".csv" else "utf-8")
        format_warnings: list[str] = []

        # — Oku ————————————————————————————————————————
        if suffix == ".csv":
            df = self._read_csv_bulletproof(path, detected_enc, size_mb,
                                            format_warnings)
        elif suffix in (".xlsx", ".xls", ".xlsm"):
            df = self._read_excel(path, sheet_name)
        else:
            raise ValueError(f"Desteklenmeyen format: {suffix}")

        # — Türk sayı formatı düzelt ————————————————————
        df, tr_fixed = fix_turkish_numbers(df)
        if tr_fixed:
            format_warnings.append(
                f"Format uyuşmazlığı nedeniyle {tr_fixed} adet hücre "
                f"Türk sayı formatından dönüştürüldü."
            )

        # — Sütun adlarını temizle ————————————————————
        df.columns = self._sanitize_cols(df.columns.tolist())

        # — DuckDB'ye yaz ——————————————————————————————
        self._register_table(df, table_name)
        backup = "ds_original_p" if table_name == "ds_primary" else "ds_original_s"
        self._register_table(df, backup)

        dur = (time.perf_counter() - t0) * 1000

        # — Profil oluştur ——————————————————————————————
        col_health      = self._column_health(df)
        missing_summary = self._missing_summary(df)
        col_profiles    = self._build_column_profiles(df)
        quality_score   = self._compute_quality_score(df, missing_summary)

        meta = _sanitize({
            "table_name":       table_name,
            "file_path":        str(path),
            "file_name":        path.name,
            "rows":             int(len(df)),
            "columns":          df.columns.tolist(),
            "dtypes":           {c: str(t) for c, t in df.dtypes.items()},
            "encoding":         detected_enc,
            "file_size_mb":     round(float(size_mb), 3),
            "file_size_kb":     round(float(size_mb * 1024), 2),
            "duration_ms":      round(float(dur), 2),
            "missing_summary":  missing_summary,
            "col_health":       col_health,
            "col_profiles":     col_profiles,
            "quality_score":    quality_score,
            "format_warnings":  format_warnings,
            "large_file":       size_mb > LARGE_FILE_THRESHOLD_MB,
        })

        self._loaded[table_name] = meta
        self._profile_cache[table_name] = col_profiles
        self._registry.append({
            "table":     table_name,
            "path":      str(path),
            "name":      path.name,
            "rows":      int(len(df)),
            "loaded_at": time.strftime("%H:%M:%S"),
        })
        self._add_lineage("LOAD", table_name, source=str(path),
                          rows=int(len(df)), encoding=detected_enc)

        logger.info("Yüklendi: %s → %s (%d satır, %.0f ms)", path.name, table_name, len(df), dur)
        return meta

    # ── CSV okuyucu — 3 aşama ─────────────────────────────────────────────

    def _read_csv_bulletproof(
        self, path: Path, enc: str, size_mb: float,
        warnings_out: list,
    ) -> pd.DataFrame:
        na_vals = ["", "NA", "N/A", "null", "NULL", "NaN", "-", "--", "nan", "none"]

        # AŞAMA 1: Büyük dosya → DuckDB read_csv_auto
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            try:
                logger.info("Büyük dosya (%.1f MB) → DuckDB read_csv_auto", size_mb)
                df = self.conn.execute(
                    f"SELECT * FROM read_csv_auto('{path}', ignore_errors=true)"
                ).df()
                if len(df) > 0:
                    return df
            except Exception as e:
                logger.warning("DuckDB read_csv_auto başarısız: %s — pandas'a geçiliyor", e)

        # AŞAMA 2: charset-normalizer tespiti + delimiter sniffer
        delim = self._detect_delimiter(path, enc)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=delim,
                    on_bad_lines="skip",
                    na_values=na_vals,
                    keep_default_na=True,
                    low_memory=False,
                    dtype_backend="numpy_nullable",
                )
            if len(df.columns) >= 1:
                return df
        except Exception as e:
            logger.warning("pd.read_csv (enc=%s, sep=%r) hata: %s", enc, delim, e)

        # AŞAMA 3: sep=None, engine=python (en güçlü fallback)
        for enc_try in [enc] + [e for e in ENCODING_CHAIN if e != enc]:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc_try,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                    na_values=na_vals,
                    keep_default_na=True,
                )
                warnings_out.append(
                    f"Dosya encoding={enc_try} ve otomatik delimiter ile yüklendi."
                )
                return df
            except Exception as e2:
                logger.warning("Fallback (enc=%s): %s", enc_try, e2)
                continue

        raise RuntimeError(
            f"'{path.name}' dosyası hiçbir yöntemle açılamadı. "
            f"Dosyanın bozuk olmadığından emin olun."
        )

    def _read_excel(self, path: Path, sheet) -> pd.DataFrame:
        na_vals = ["", "NA", "N/A", "null", "NULL", "NaN", "-"]
        return pd.read_excel(path, sheet_name=sheet, na_values=na_vals)

    # ── Encoding & Delimiter ───────────────────────────────────────────────

    def _detect_encoding(self, path: Path, sample_bytes: int = 131072) -> str:
        with open(path, "rb") as f:
            raw = f.read(sample_bytes)

        # BOM kontrolü
        if raw.startswith(b"\xef\xbb\xbf"):   return "utf-8-sig"
        if raw.startswith(b"\xff\xfe"):         return "utf-16-le"
        if raw.startswith(b"\xfe\xff"):         return "utf-16-be"

        # charset-normalizer (en güvenilir)
        if _CN:
            try:
                matches = _cn_from_bytes(raw)
                if matches:
                    best = matches.best()
                    if best and best.encoding:
                        logger.info("charset-normalizer → %s (%.0f%% güven)",
                                    best.encoding, (best.chaos * -100 + 100))
                        return str(best.encoding)
            except Exception as e:
                logger.debug("charset-normalizer hata: %s", e)

        # chardet fallback
        if _CHARDET:
            r = chardet.detect(raw)
            if r.get("confidence", 0) >= 0.70 and r.get("encoding"):
                return r["encoding"]

        # Manuel fallback zinciri
        return self._encoding_fallback(path)

    def _encoding_fallback(self, path: Path) -> str:
        for enc in ENCODING_CHAIN:
            try:
                with open(path, encoding=enc, errors="strict") as f:
                    f.read(8192)
                logger.info("Encoding fallback: %s", enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        return "latin-1"

    def _detect_delimiter(self, path: Path, enc: str) -> str:
        """
        Delimiter tespiti:
          1. csv.Sniffer ile ilk 4KB
          2. Karakter sayımı
        """
        try:
            with open(path, encoding=enc, errors="replace") as f:
                sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            logger.info("Sniffer delimiter: %r", dialect.delimiter)
            return dialect.delimiter
        except csv.Error:
            pass

        # Sayım fallback
        with open(path, encoding=enc, errors="replace") as f:
            head = f.read(2048)
        counts = {d: head.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=counts.get)
        logger.info("Sayım delimiter: %r (count=%d)", best, counts[best])
        return best

    # ──────────────────────────────────────────────────────────────────────
    # Versiyonlama (Data Lineage)
    # ──────────────────────────────────────────────────────────────────────

    def write_result(self, df: pd.DataFrame, result_name: str,
                     source_table: str | None = None,
                     operation: str | None = None) -> str:
        """
        Versiyonlu snapshot yazar: ds_result_<ad>_v<N>

        Returns: tablo adı (ör. "ds_result_normalize_v2")
        """
        v = self._version.get(result_name, 0) + 1
        self._version[result_name] = v
        tname = f"ds_result_{result_name}_v{v}"
        self._register_table(df, tname)

        self._add_lineage(
            operation or "TRANSFORM",
            tname,
            source=source_table,
            rows=int(len(df)),
            version=v,
        )
        logger.info("Snapshot yazıldı: %s (%d satır)", tname, len(df))
        return tname

    def get_latest_result(self, result_name: str) -> str | None:
        """En son versiyonlu tablo adını döner."""
        v = self._version.get(result_name, 0)
        if v == 0:
            return None
        return f"ds_result_{result_name}_v{v}"

    def get_version_history(self, result_name: str) -> list[dict]:
        """Belirli bir sonucun tüm versiyon geçmişini döner."""
        prefix = f"ds_result_{result_name}_v"
        return [e for e in self._lineage if
                e.get("table", "").startswith(prefix)]

    def get_lineage(self) -> list[dict]:
        return list(self._lineage)

    def undo_last(self, result_name: str) -> str | None:
        """
        Son versiyonu geri alır.
        Önceki versiyonun adını döner (veya None).
        """
        v = self._version.get(result_name, 0)
        if v <= 1:
            logger.warning("Geri alınacak versiyon yok: %s", result_name)
            return None
        old = f"ds_result_{result_name}_v{v}"
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {old}")
        except Exception:
            pass
        self._version[result_name] = v - 1
        prev = f"ds_result_{result_name}_v{v-1}"
        logger.info("Geri alındı: %s → %s", old, prev)
        return prev

    # ──────────────────────────────────────────────────────────────────────
    # Sorgu & Yardımcılar
    # ──────────────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self.conn.execute(sql).df()
        except duckdb.Error as e:
            logger.error("SQL hata: %s | %s", e, sql[:120])
            raise

    def get_df(self, table_name: str) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {table_name}")

    def get_original(self, which: str = "p") -> pd.DataFrame | None:
        tbl = f"ds_original_{which}"
        return self.get_df(tbl) if self.table_exists(tbl) else None

    def update_working_copy(self, df: pd.DataFrame,
                            table_name: str = "ds_primary") -> None:
        """Çalışma kopyasını günceller. Orijinal backup'a dokunmaz."""
        self._register_table(df, table_name)

    def remove_file(self, table_name: str) -> bool:
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            backup = "ds_original_p" if table_name == "ds_primary" else "ds_original_s"
            self.conn.execute(f"DROP TABLE IF EXISTS {backup}")
            self._loaded.pop(table_name, None)
            self._profile_cache.pop(table_name, None)
            self._registry = [r for r in self._registry
                              if r["table"] != table_name]
            return True
        except Exception as e:
            logger.error("Tablo silinemedi %s: %s", table_name, e)
            return False

    def table_exists(self, table_name: str) -> bool:
        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 0")
            return True
        except duckdb.Error:
            return False

    def get_numeric_columns(self, table_name: str) -> list[str]:
        df_desc = self.query(f"DESCRIBE {table_name}")
        NUM = {"INTEGER","BIGINT","DOUBLE","FLOAT","DECIMAL",
               "HUGEINT","SMALLINT","TINYINT","UBIGINT","UINTEGER","REAL"}
        return [r["column_name"] for _, r in df_desc.iterrows()
                if any(t in str(r["column_type"]).upper() for t in NUM)]

    def get_string_columns(self, table_name: str) -> list[str]:
        df_desc = self.query(f"DESCRIBE {table_name}")
        return [r["column_name"] for _, r in df_desc.iterrows()
                if "VARCHAR" in str(r["column_type"]).upper()
                or "TEXT" in str(r["column_type"]).upper()]

    def get_all_columns(self, table_name: str) -> list[str]:
        if table_name in self._loaded:
            return self._loaded[table_name]["columns"]
        return self.query(f"DESCRIBE {table_name}")["column_name"].tolist()

    def get_loaded_tables(self) -> dict[str, dict]:
        return dict(self._loaded)

    def get_file_registry(self) -> list[dict]:
        return list(self._registry)

    def export_table(self, table_name: str, output_path: str | Path,
                     fmt: str = "csv") -> Path:
        df  = self.get_df(table_name)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out, index=False, encoding="utf-8-sig")
        elif fmt == "xlsx":
            df.to_excel(out, index=False, engine="openpyxl")
        elif fmt == "parquet":
            df.to_parquet(out, index=False)
        else:
            raise ValueError(f"Desteklenmeyen format: {fmt}")
        logger.info("Export: %s → %s", table_name, out)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Sütun Profil & Sağlık
    # ──────────────────────────────────────────────────────────────────────

    def _build_column_profiles(self, df: pd.DataFrame) -> dict:
        """
        Her sütun için hızlı profil:
          miss_pct, type_mismatch_pct, quality_score, top5, dtype_label
        """
        profiles = {}
        n = len(df)
        for col in df.columns:
            s = df[col]
            miss_pct = float(s.isna().mean() * 100)

            # Tip uyumsuzluğu: object sütunda karışık tipler
            type_mismatch_pct = 0.0
            if s.dtype == object:
                non_na = s.dropna().astype(str)
                num_like = non_na.apply(
                    lambda x: x.replace(".", "").replace(",", "").replace("-", "").lstrip("-").isdigit()
                ).mean()
                if 0 < num_like < 0.8:
                    type_mismatch_pct = float((1 - num_like) * 100)

            # Kalite skoru (0–100)
            qs = 100.0
            qs -= min(50, miss_pct * 2)          # eksiklik cezası
            qs -= min(30, type_mismatch_pct * 0.5)
            if pd.api.types.is_numeric_dtype(s):
                clean = pd.to_numeric(s, errors="coerce").dropna()
                if len(clean) > 5:
                    cv = float(clean.std() / clean.mean()) if clean.mean() != 0 else 0
                    if cv > 3:
                        qs -= min(20, cv * 2)
            qs = max(0.0, round(qs, 1))

            # Top 5 değer
            top5 = []
            try:
                vc = s.value_counts().head(5)
                top5 = [{"value": str(k), "count": int(v), "pct": round(float(v/n*100), 1)}
                        for k, v in vc.items()]
            except Exception:
                pass

            # İstatistik (sayısal sütunlar)
            stats = {}
            if pd.api.types.is_numeric_dtype(s):
                clean = pd.to_numeric(s, errors="coerce").dropna()
                if len(clean) > 0:
                    from scipy import stats as sp
                    stats = _sanitize({
                        "mean":    round(float(clean.mean()), 4),
                        "median":  round(float(clean.median()), 4),
                        "std":     round(float(clean.std()), 4),
                        "min":     round(float(clean.min()), 4),
                        "max":     round(float(clean.max()), 4),
                        "skew":    round(float(sp.skew(clean)), 4),
                        "kurt":    round(float(sp.kurtosis(clean)), 4),
                    })

            profiles[col] = _sanitize({
                "miss_pct":          miss_pct,
                "type_mismatch_pct": type_mismatch_pct,
                "quality_score":     qs,
                "top5":              top5,
                "dtype":             str(s.dtype),
                "stats":             stats,
                "n_unique":          int(s.nunique()),
            })
        return profiles

    def get_column_profile(self, table_name: str, col: str) -> dict:
        """Tek sütun profili — önbellekten veya hesaplar."""
        cache = self._profile_cache.get(table_name, {})
        if col in cache:
            return cache[col]
        df = self.get_df(table_name)
        profiles = self._build_column_profiles(df)
        self._profile_cache[table_name] = profiles
        return profiles.get(col, {})

    def _missing_summary(self, df: pd.DataFrame) -> dict:
        n = max(len(df), 1)
        return _sanitize({
            c: {
                "missing_count": int(df[c].isna().sum()),
                "missing_pct":   round(float(df[c].isna().mean() * 100), 2),
            }
            for c in df.columns
        })

    def _column_health(self, df: pd.DataFrame) -> dict:
        health = {}
        for col in df.columns:
            miss_pct = float(df[col].isna().mean() * 100)
            if miss_pct > 5:
                health[col] = "missing"
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                clean = df[col].dropna()
                if len(clean) > 10:
                    mean_ = float(clean.mean())
                    std_  = float(clean.std())
                    if mean_ != 0 and (std_ / abs(mean_)) > 2.0:
                        health[col] = "anomaly"
                        continue
            health[col] = "clean"
        return health

    def _compute_quality_score(self, df: pd.DataFrame,
                               missing_summary: dict) -> float:
        """
        Veri seti düzeyinde kalite skoru (0–100):
          - Eksik veri oranı
          - Tip uyumsuzluğu
          - Yüksek varyasyon katsayısı
        """
        n_cols = max(len(df.columns), 1)
        miss_avg = float(np.mean([v["missing_pct"]
                                  for v in missing_summary.values()]))
        score = 100.0
        score -= min(40, miss_avg * 2)

        # Anomali sütun oranı
        health = self._column_health(df)
        anom_pct = sum(1 for v in health.values() if v == "anomaly") / n_cols * 100
        score -= min(30, anom_pct)

        return max(0.0, round(score, 1))

    def compute_before_after_stats(
        self, before: pd.DataFrame, after: pd.DataFrame
    ) -> dict:
        """
        İki DataFrame arasındaki istatistiksel delta.
        Returns: {col: {mean_before, mean_after, std_before, std_after, …}}
        """
        from scipy import stats as sp
        result = {}
        num_cols = [c for c in before.columns
                    if c in after.columns
                    and pd.api.types.is_numeric_dtype(before[c])]
        for col in num_cols:
            b = pd.to_numeric(before[col], errors="coerce").dropna()
            a = pd.to_numeric(after[col],  errors="coerce").dropna()
            if len(b) < 2 or len(a) < 2:
                continue

            mb, ma = float(b.mean()), float(a.mean())
            vb, va = float(b.var()),  float(a.var())
            sb, sa = float(b.std()),  float(a.std())

            var_chg_pct = (
                round((va - vb) / abs(vb) * 100, 2)
                if vb != 0 else None
            )
            radical_change = (var_chg_pct is not None
                              and abs(var_chg_pct) > 20)

            result[col] = _sanitize({
                "mean_before":    round(mb, 4), "mean_after":    round(ma, 4),
                "std_before":     round(sb, 4), "std_after":     round(sa, 4),
                "var_before":     round(vb, 4), "var_after":     round(va, 4),
                "skew_before":    round(float(sp.skew(b.values)), 4),
                "skew_after":     round(float(sp.skew(a.values)), 4),
                "kurt_before":    round(float(sp.kurtosis(b.values)), 4),
                "kurt_after":     round(float(sp.kurtosis(a.values)), 4),
                "var_change_pct": var_chg_pct,
                "radical_change": radical_change,
            })
        return result

    def compute_correlation_matrix(
        self, table_name: str, max_cols: int = 8
    ) -> dict:
        """
        Pearson & Spearman korelasyon matrisini hesaplar.
        En önemli 5 çift korelasyonu döner.
        """
        from scipy import stats as sp
        df   = self.get_df(table_name)
        nums = self.get_numeric_columns(table_name)[:max_cols]
        if len(nums) < 2:
            return {}

        sub = df[nums].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 5:
            return {}

        pairs = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = sub[nums[i]].values, sub[nums[j]].values
                r_p, p_p = sp.pearsonr(a, b)
                r_s, p_s = sp.spearmanr(a, b)
                pairs.append(_sanitize({
                    "col_a":     nums[i],
                    "col_b":     nums[j],
                    "pearson_r": round(float(r_p), 4),
                    "pearson_p": round(float(p_p), 4),
                    "spearman_r": round(float(r_s), 4),
                    "spearman_p": round(float(p_s), 4),
                    "strength":  _corr_label(abs(float(r_p))),
                    "significant": bool(float(p_p) < 0.05),
                }))

        pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        return {"pairs": pairs[:10], "columns": nums}
    
    
    





    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _register_table(self, df: pd.DataFrame, name: str) -> None:
        """DataFrame'i DuckDB tablosuna yazar (DROP + CREATE)."""
        self.conn.execute(f"DROP TABLE IF EXISTS {name}")
        self.conn.register("__tmp__", df)
        self.conn.execute(f"CREATE TABLE {name} AS SELECT * FROM __tmp__")
        self.conn.unregister("__tmp__")

    def _add_lineage(self, op: str, table: str, **kwargs) -> None:
        self._lineage.append(_sanitize({
            "timestamp": time.strftime("%H:%M:%S"),
            "operation": op,
            "table":     table,
            **kwargs,
        }))

    @staticmethod
    def _sanitize_cols(cols: list[str]) -> list[str]:
        out = []
        for c in cols:
            s = re.sub(r"[^\w]", "_", str(c).strip())
            s = re.sub(r"_+", "_", s).strip("_")
            if s and s[0].isdigit():
                s = "col_" + s
            out.append(s or "column")
        seen: dict[str, int] = {}
        result = []
        for c in out:
            if c in seen:
                seen[c] += 1
                result.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                result.append(c)
        return result

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


def _corr_label(r: float) -> str:
    if r >= 0.8:  return "Çok Güçlü"
    if r >= 0.6:  return "Güçlü"
    if r >= 0.4:  return "Orta"
    if r >= 0.2:  return "Zayıf"
    return "Çok Zayıf"



# database.py içine eklenecek basit fonksiyon:
def create_restore_point(self, table_name="ds_primary"):
    df = self.get_df(table_name)
    self.write_result(df, "restore_point", source_table=table_name, operation="BACKUP")

def restore_from_point(self, target_table="ds_primary"):
    df = self.get_df("ds_restore_point") # Yedekten al
    self.update_working_copy(df) # Ana tabloya yaz