"""
smart_imputer.py — Data-Autopsy Akıllı Veri Doldurucu (Univariate)

Eksik veri desenlerini (MCAR / MAR / MNAR) eksiklik korelasyon matrisiyle analiz eder
ve veri tipine + dağılım şekline göre en uygun tek değişkenli doldurma yöntemini önerir.

Metodoloji:
  - MCAR / MAR Ayrımı: Bir sütundaki eksikliğin, diğer sütunlardaki eksiklikle 
    korelasyonuna bakılır. Korelasyon varsa MAR (Missing at Random), yoksa MCAR'dır.
  - MNAR Şüphesi: Eksiklik oranı çok yüksekse (>%40) MNAR şüphesi raporlanır.
  - Doldurma: Çarpık dağılımlarda Medyan, simetriklerde Ortalama. Sıralı veride İnterpolasyon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MissingPattern(str, Enum):
    MCAR = "MCAR"   # Eksiklik tamamen rastgele (diğer değişkenlerden bağımsız)
    MAR  = "MAR"    # Eksiklik rastgele (diğer değişkenlerdeki eksikliklerle ilişkili)
    MNAR = "MNAR"   # Eksiklik rastgele değil (sistematik hata veya yüksek kayıp)
    NONE = "NONE"   # Eksik değer yok


class ImputationMethod(str, Enum):
    MEAN          = "mean"
    MEDIAN        = "median"
    MODE          = "mode"
    CONSTANT      = "constant"
    FORWARD_FILL  = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE   = "interpolate" # Sahte KNN yerine bilimsel yaklaşım


@dataclass
class ColumnMissingInfo:
    """Tek sütunun eksiklik analizi."""
    column:              str
    dtype:               str
    total_count:         int
    missing_count:       int
    missing_pct:         float
    pattern:             MissingPattern
    recommended_method:  ImputationMethod
    method_reason:       str
    alternative_methods: list[ImputationMethod] = field(default_factory=list)


@dataclass
class ImputationResult:
    """Doldurma işlemi sonucu."""
    column:        str
    method:        ImputationMethod
    cells_filled:  int
    fill_value:    object  
    pattern:       MissingPattern


# ---------------------------------------------------------------------------
# Eksiklik Analizi
# ---------------------------------------------------------------------------

def analyze_missing_patterns(df: pd.DataFrame) -> dict[str, ColumnMissingInfo]:
    """
    DataFrame'deki tüm sütunlar için eksiklik analizi yapar.
    Gerçek istatistiksel yaklaşım: Missing Indicator Matrix Korelasyonu.
    """
    results: dict[str, ColumnMissingInfo] = {}
    total = len(df)
    if total == 0:
        return results

    # Eksiklik binary matrisi (1=Eksik, 0=Dolu)
    missing_matrix = df.isnull().astype(int)

    for col in df.columns:
        missing_count = int(missing_matrix[col].sum())
        missing_pct   = round(missing_count / total * 100, 2)

        if missing_count == 0:
            results[col] = ColumnMissingInfo(
                column=col, dtype=str(df[col].dtype), total_count=total,
                missing_count=0, missing_pct=0.0, pattern=MissingPattern.NONE,
                recommended_method=ImputationMethod.MEAN, method_reason="Eksik değer yok."
            )
            continue

        # MCAR / MAR Ayrımı için Korelasyon Analizi
        # Bu sütunun eksikliğinin, diğer sütunların eksikliğiyle ilişkisi var mı?
        other_cols = [c for c in df.columns if c != col and missing_matrix[c].sum() > 0]
        max_corr = 0.0

        if other_cols:
            corrs = missing_matrix[[col] + other_cols].corr()
            corr_series = corrs[col].drop(col).fillna(0)
            if not corr_series.empty:
                max_corr = float(corr_series.abs().max())

        pattern, reason = _classify_pattern(missing_pct, max_corr)
        method, m_reason = _recommend_method(df[col], pattern, missing_pct)

        full_reason = f"{reason} {m_reason}"

        results[col] = ColumnMissingInfo(
            column=col,
            dtype=str(df[col].dtype),
            total_count=total,
            missing_count=missing_count,
            missing_pct=missing_pct,
            pattern=pattern,
            recommended_method=method,
            method_reason=full_reason,
            alternative_methods=_alternative_methods(df[col], method),
        )

    return results


def _classify_pattern(missing_pct: float, max_corr: float) -> tuple[MissingPattern, str]:
    """Missing Indicator korelasyonuna ve kayıp oranına göre desen belirler."""
    
    if missing_pct > 40:
        return (
            MissingPattern.MNAR,
            f"Kritik eksiklik (%.1f%%) — Verinin yapısı bozulmuş olabilir (MNAR şüphesi)." % missing_pct
        )
    
    if max_corr >= 0.2:
        return (
            MissingPattern.MAR,
            f"Diğer eksikliklerle korele (r={max_corr:.2f}) — MAR."
        )
    else:
        return (
            MissingPattern.MCAR,
            f"Bağımsız rastgele eksiklik (r={max_corr:.2f}) — MCAR."
        )


def _recommend_method(series: pd.Series, pattern: MissingPattern, missing_pct: float) -> tuple[ImputationMethod, str]:
    """Dağılımın momentlerine ve veri tipine göre sağlam (robust) yöntem seçer."""
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if is_datetime:
        return ImputationMethod.FORWARD_FILL, "Zaman serisi için ileri doldurma önerilir."

    if not is_numeric:
        if missing_pct > 30 or pattern == MissingPattern.MNAR:
            return ImputationMethod.CONSTANT, "Yüksek oranda kayıp kategorik veri için sabit değer daha güvenlidir."
        return ImputationMethod.MODE, "Kategorik veri için mod (tepe değer) kullanımı uygundur."

    # Sayısal Veriler İçin İstatistiksel Kontrol
    if missing_pct > 40 or pattern == MissingPattern.MNAR:
        return ImputationMethod.MEDIAN, "Yüksek kayıp oranında medyan daha az önyargı yaratır."

    try:
        clean_series = series.dropna()
        skewness = float(clean_series.skew())
        if abs(skewness) < 0.5:
            return ImputationMethod.MEAN, f"Dağılım simetrik (skew={skewness:.2f}), ortalama uygundur."
        else:
            return ImputationMethod.MEDIAN, f"Dağılım çarpık (skew={skewness:.2f}), medyan daha dirençlidir."
    except Exception:
        return ImputationMethod.MEDIAN, "Dağılım hesaplanamadı, varsayılan olarak medyan önerilir."


def _alternative_methods(series: pd.Series, primary: ImputationMethod) -> list[ImputationMethod]:
    """Birincil yönteme göre mantıklı alternatifleri döndürür."""
    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric:
        cands = [ImputationMethod.MEAN, ImputationMethod.MEDIAN, ImputationMethod.INTERPOLATE]
    else:
        cands = [ImputationMethod.MODE, ImputationMethod.CONSTANT, ImputationMethod.FORWARD_FILL]
    
    return [m for m in cands if m != primary][:2]


# ---------------------------------------------------------------------------
# Doldurma İşlemi
# ---------------------------------------------------------------------------

def impute_column(
    series: pd.Series,
    method: ImputationMethod,
    constant_value=None,
) -> tuple[pd.Series, ImputationResult]:
    """Seçilen yönteme göre eksik değerleri güvenli bir şekilde doldurur."""
    original_missing = int(series.isnull().sum())
    filled = series.copy()
    fill_val = None

    if method == ImputationMethod.MEAN:
        fill_val = float(filled.mean())
        filled = filled.fillna(fill_val)

    elif method == ImputationMethod.MEDIAN:
        fill_val = float(filled.median())
        filled = filled.fillna(fill_val)

    elif method == ImputationMethod.MODE:
        mode_vals = filled.mode()
        fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else None
        if fill_val is not None:
            filled = filled.fillna(fill_val)

    elif method == ImputationMethod.CONSTANT:
        fill_val = constant_value if constant_value is not None else ("Bilinmiyor" if series.dtype == object else 0)
        filled = filled.fillna(fill_val)

    elif method == ImputationMethod.FORWARD_FILL:
        filled = filled.ffill()
        fill_val = "forward_fill"

    elif method == ImputationMethod.BACKWARD_FILL:
        filled = filled.bfill()
        fill_val = "backward_fill"

    elif method == ImputationMethod.INTERPOLATE:
        # KNN yerine lineer interpolasyon. Özellikle sıralı/zaman bazlı verilerde çok güçlüdür.
        if pd.api.types.is_numeric_dtype(series):
            filled = filled.interpolate(method='linear', limit_direction='both')
            fill_val = "linear_interpolation"
        else:
            filled = filled.ffill() # Sayısal değilse ffill fallback
            fill_val = "forward_fill_fallback"

    cells_filled = original_missing - int(filled.isnull().sum())

    result = ImputationResult(
        column=str(series.name),
        method=method,
        cells_filled=cells_filled,
        fill_value=fill_val,
        pattern=MissingPattern.NONE,
    )

    logger.info("Doldurma [%s, %s]: %d hücre dolduruldu", series.name, method.value, cells_filled)
    return filled, result