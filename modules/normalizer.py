"""
normalizer.py — Data-Autopsy v5  (Core Logic Restructure)

Düzeltmeler:
  1. Tip kontrolü: _is_text_column() ile yalnızca gerçek metin sütunlara işlem.
     Sayısal (int/float) sütunlara kesinlikle dokunulmaz.
  2. tr_lower/tr_upper: hatalı encoding (Ã¼ vb.) veya beklenmedik
     karakter gelse bile try/except ile çökmez, orijinali döner.
  3. normalize_dataframe: target_cols seçimi pd.api.types.is_string_dtype +
     içerik sayısallık oranı kontrolüyle yapılır.
  4. SeriesNormalizer: her adımda _guard() ile tip koruması.
"""
from __future__ import annotations

import re
import logging
import unicodedata
from typing import Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Türkçe Harf Tabloları ────────────────────────────────────────────────────
TR_UPPER_TO_LOWER = {
    "I": "ı", "İ": "i", "Ğ": "ğ", "Ü": "ü",
    "Ş": "ş", "Ö": "ö", "Ç": "ç",
}
TR_LOWER_TO_UPPER = {
    "ı": "I", "i": "İ", "ğ": "Ğ", "ü": "Ü",
    "ş": "Ş", "ö": "Ö", "ç": "Ç",
}
_TR_LOWER_TABLE = str.maketrans(TR_UPPER_TO_LOWER)
_TR_UPPER_TABLE = str.maketrans(TR_LOWER_TO_UPPER)

ENCODING_CORRUPTION_MAP = {
    "Ã¼": "ü", "Ã–": "Ö", "Ã§": "ç", "Ã¶": "ö", "Ã‡": "Ç",
    "Äž": "Ğ", "Ä±": "ı", "Ä°": "İ", "Å\x9e": "Ş", "Å\x9f": "ş",
}


# ── Tip Kontrolü ─────────────────────────────────────────────────────────────

def _is_text_column(series: pd.Series) -> bool:
    """
    Sütunun gerçekten metin içerip içermediğini doğrular.
    Sayısal dtype → False.
    String dtype ama içeriğin %80+ sayısal ise → False (sayısal veri string formatında).
    """
    if pd.api.types.is_numeric_dtype(series):
        return False
    if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
        return False
    # İçerik sayısallık oranı kontrolü
    non_na = series.dropna()
    if len(non_na) == 0:
        return True
    sample = non_na.head(200)
    numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()
    return float(numeric_ratio) < 0.80


# ── Güvenli Türkçe Harf Fonksiyonları ───────────────────────────────────────

def tr_lower(t: str) -> str:
    """Türkçe küçük harf dönüşümü. Bozuk encoding gelse çökmez."""
    if not isinstance(t, str):
        return t
    try:
        return t.translate(_TR_LOWER_TABLE).lower()
    except Exception:
        # Karakter bazlı fallback
        try:
            return "".join(
                TR_UPPER_TO_LOWER.get(ch, ch) for ch in t
            ).lower()
        except Exception:
            return t


def tr_upper(t: str) -> str:
    """Türkçe büyük harf dönüşümü. Çökme garantisi."""
    if not isinstance(t, str):
        return t
    try:
        return t.translate(_TR_UPPER_TABLE).upper()
    except Exception:
        try:
            return "".join(
                TR_LOWER_TO_UPPER.get(ch, ch) for ch in t
            ).upper()
        except Exception:
            return t


def tr_title(t: str) -> str:
    """Türkçe başlık harfi. Çökme garantisi."""
    if not isinstance(t, str):
        return t
    try:
        words = t.split()
        result = []
        for w in words:
            if not w:
                continue
            first = TR_LOWER_TO_UPPER.get(w[0], w[0].upper())
            rest  = tr_lower(w[1:]) if len(w) > 1 else ""
            result.append(first + rest)
        return " ".join(result)
    except Exception:
        return t


# ── Encoding Düzeltici ───────────────────────────────────────────────────────

def fix_encoding_corruption(text: str, aggressive: bool = False) -> str:
    """latin-1→UTF-8 çift encoding sorununu düzeltir. Asla çökmez."""
    if not isinstance(text, str):
        return text
    try:
        recovered = text.encode("latin-1").decode("utf-8")
        if _looks_more_turkish(recovered, text):
            text = recovered
    except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
        pass
    if aggressive:
        for broken, correct in ENCODING_CORRUPTION_MAP.items():
            try:
                text = text.replace(broken, correct)
            except Exception:
                pass
    return text


def _looks_more_turkish(cand: str, orig: str) -> bool:
    tr_chars = set("ğĞüÜşŞöÖçÇıİ")
    return (
        sum(1 for c in cand if c in tr_chars) >=
        sum(1 for c in orig if c in tr_chars)
    )


# ── Boşluk & Unicode ────────────────────────────────────────────────────────

def normalize_whitespace(t: str) -> str:
    if not isinstance(t, str):
        return t
    try:
        return re.sub(r"\s+", " ", t).strip()
    except Exception:
        return t


def normalize_unicode(t: str, form: str = "NFC") -> str:
    if not isinstance(t, str):
        return t
    try:
        return unicodedata.normalize(form, t)
    except Exception:
        return t


# ── Tarih Standartlaştırıcı ──────────────────────────────────────────────────

_DATE_FORMATS = [
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$"),  "{2}-{1:02d}-{0:02d}"),
    (re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$"),  "{0}-{1:02d}-{2:02d}"),
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2})$"),  "20{2}-{1:02d}-{0:02d}"),
]


def normalize_date(text: str) -> str:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    for pattern, fmt in _DATE_FORMATS:
        m = pattern.match(stripped)
        if m:
            try:
                parts = [int(x) for x in m.groups()]
                return fmt.format(*parts)
            except Exception:
                pass
    return text


# ── Türkiye Coğrafi Sözlüğü ─────────────────────────────────────────────────

GEO_DICT: dict[str, str] = {
    "istanbul": "İstanbul", "İstanbul": "İstanbul", "ıstanbul": "İstanbul",
    "istabul": "İstanbul",  "istanbull": "İstanbul", "34": "İstanbul",
    "ankara": "Ankara",     "ankra": "Ankara",   "06": "Ankara",
    "izmir": "İzmir",       "İzmir": "İzmir",    "35": "İzmir",
    "bursa": "Bursa",       "16": "Bursa",
    "adana": "Adana",       "01": "Adana",
    "antalya": "Antalya",   "07": "Antalya",
    "gaziantep": "Gaziantep", "antep": "Gaziantep", "27": "Gaziantep",
    "diyarbakır": "Diyarbakır", "dıyarbakır": "Diyarbakır",
    "diyarbakir": "Diyarbakır", "21": "Diyarbakır",
    "kocaeli": "Kocaeli",   "izmit": "Kocaeli",  "41": "Kocaeli",
    "konya": "Konya",       "42": "Konya",
    "mersin": "Mersin",     "33": "Mersin",
    "samsun": "Samsun",     "55": "Samsun",
    "erzurum": "Erzurum",   "25": "Erzurum",
    "eskişehir": "Eskişehir", "eskisehir": "Eskişehir", "26": "Eskişehir",
    "trabzon": "Trabzon",   "61": "Trabzon",
    "kayseri": "Kayseri",   "38": "Kayseri",
    "hatay": "Hatay",       "antakya": "Hatay",  "31": "Hatay",
    "manisa": "Manisa",     "45": "Manisa",
    "balıkesir": "Balıkesir", "balikesir": "Balıkesir", "10": "Balıkesir",
}


def normalize_city(text: str) -> str:
    if not isinstance(text, str):
        return text
    key = text.strip().lower()
    return GEO_DICT.get(key, GEO_DICT.get(text.strip(), text))


# ── SeriesNormalizer ─────────────────────────────────────────────────────────

def _safe_apply(fn: Callable[[str], str]) -> Callable[[pd.Series], pd.Series]:
    """fn'i her string elemana güvenle uygular. Hata olursa orijinali döner."""
    def _wrapper(series: pd.Series) -> pd.Series:
        def _per_cell(x):
            if not isinstance(x, str):
                return x
            try:
                return fn(x)
            except Exception:
                return x
        return series.apply(_per_cell)
    return _wrapper


class SeriesNormalizer:
    """
    Tip-güvenli Pandas Series normalizer.
    Sayısal sütun verilirse pipeline adımları eklenmez — hiçbir şeye dokunmaz.
    """

    def __init__(self, series: pd.Series) -> None:
        self._original = series.copy()
        self._current  = series.copy()
        self._pipeline: list[tuple[str, Callable]] = []
        self._is_text  = _is_text_column(series)

    def _guard(self, name: str, fn: Callable) -> "SeriesNormalizer":
        if self._is_text:
            self._pipeline.append((name, fn))
        else:
            logger.debug("Atlandı (sayısal): %s", name)
        return self

    def fix_encoding(self, aggressive: bool = False) -> "SeriesNormalizer":
        return self._guard(
            "fix_encoding",
            _safe_apply(lambda x: fix_encoding_corruption(x, aggressive)),
        )

    def turkish_lower(self)  -> "SeriesNormalizer":
        return self._guard("turkish_lower",  _safe_apply(tr_lower))

    def turkish_upper(self)  -> "SeriesNormalizer":
        return self._guard("turkish_upper",  _safe_apply(tr_upper))

    def turkish_title(self)  -> "SeriesNormalizer":
        return self._guard("turkish_title",  _safe_apply(tr_title))

    def strip_whitespace(self) -> "SeriesNormalizer":
        return self._guard("strip_whitespace", _safe_apply(normalize_whitespace))

    def unicode_normalize(self, form: str = "NFC") -> "SeriesNormalizer":
        return self._guard(
            "unicode_normalize",
            _safe_apply(lambda x: normalize_unicode(x, form)),
        )

    def date_normalize(self) -> "SeriesNormalizer":
        return self._guard("date_normalize", _safe_apply(normalize_date))

    def city_normalize(self) -> "SeriesNormalizer":
        return self._guard("city_normalize", _safe_apply(normalize_city))

    def apply(self) -> pd.Series:
        result = self._current.copy()
        for step_name, fn in self._pipeline:
            try:
                result = fn(result)
            except Exception as exc:
                logger.warning("Pipeline '%s' başarısız: %s", step_name, exc)
        self._current = result
        return result

    @property
    def change_count(self) -> int:
        return int((self._original.astype(str) != self._current.astype(str)).sum())

    @property
    def change_report(self) -> pd.DataFrame:
        mask = self._original.astype(str) != self._current.astype(str)
        return pd.DataFrame({
            "original":   self._original[mask],
            "normalized": self._current[mask],
        })

    def change_samples(self, n: int = 10) -> list[dict]:
        mask    = self._original.astype(str) != self._current.astype(str)
        idx     = list(self._original[mask].index)
        if not idx:
            return []
        chosen  = list(np.random.choice(idx, min(n, len(idx)), replace=False))
        return [
            {
                "row":    int(i),
                "before": str(self._original.iloc[i]) if i < len(self._original) else "—",
                "after":  str(self._current.iloc[i])  if i < len(self._current)  else "—",
            }
            for i in sorted(chosen)
        ]


# ── Toplu Normalizasyon ──────────────────────────────────────────────────────

def normalize_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    operations: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    DataFrame üzerinde tip-güvenli normalizasyon.

    Kural: Sayısal sütunlara kesinlikle dokunulmaz.
    Kural: Yalnızca _is_text_column()==True olan sütunlar işlenir.

    Returns (result_df, summary) — summary[col] içinde skipped_reason alanı var.
    """
    if operations is None:
        operations = ["fix_encoding", "strip_whitespace", "unicode_normalize"]

    result_df = df.copy()

    if columns is not None:
        text_cols    = [c for c in columns if c in df.columns and _is_text_column(df[c])]
        skipped_cols = [c for c in columns if c in df.columns and not _is_text_column(df[c])]
        for c in skipped_cols:
            logger.info("normalize_dataframe: '%s' sayısal sütun, atlandı.", c)
    else:
        text_cols    = [c for c in df.columns if _is_text_column(df[c])]
        skipped_cols = []

    summary: dict[str, dict] = {}

    for col in text_cols:
        sn = SeriesNormalizer(df[col])
        sn._original.name = col
        sn._current.name  = col

        for op in operations:
            if op == "fix_encoding":       sn.fix_encoding()
            elif op == "turkish_lower":    sn.turkish_lower()
            elif op == "turkish_upper":    sn.turkish_upper()
            elif op == "turkish_title":    sn.turkish_title()
            elif op == "strip_whitespace": sn.strip_whitespace()
            elif op == "unicode_normalize":sn.unicode_normalize()
            elif op == "date_normalize":   sn.date_normalize()
            elif op == "city_normalize":   sn.city_normalize()

        result_df[col] = sn.apply()

        def _qstats(s: pd.Series) -> dict:
            num = pd.to_numeric(s, errors="coerce").dropna()
            if len(num) < 2:
                return {}
            return {
                "mean":   round(float(num.mean()),   4),
                "std":    round(float(num.std()),    4),
                "median": round(float(num.median()), 4),
            }

        summary[col] = {
            "changes":        int(sn.change_count),
            "total":          int(len(df)),
            "operations":     list(operations),
            "change_samples": sn.change_samples(10),
            "before_stats":   _qstats(df[col]),
            "after_stats":    _qstats(result_df[col]),
            "skipped_reason": None,
        }

    # Sayısal olduğu için atlananlara bilgi kaydı
    for col in skipped_cols:
        summary[col] = {
            "changes": 0, "total": int(len(df)),
            "operations": [], "change_samples": [],
            "before_stats": {}, "after_stats": {},
            "skipped_reason": "sayısal_sütun",
        }

    return result_df, summary
