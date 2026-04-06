"""
statistical_auditor.py — Data-Autopsy v3

Yenilikler:
  - Shapiro-Wilk / D'Agostino normallik testi → otomatik Bilirkişi Notu
  - IQR + Robust Z-Score + Klasik Z-Score yan yana karşılaştırma
  - Benford: gözlenen vs. beklenen tablo + bar chart verisi
  - Effect size (mean_delta, var_delta, cohen_d) her analizde
  - Tüm çıktılar _sanitize() ile JSON-safe
  - Scipy'ye giden her array np.float64 formatında

Referanslar:
  Shapiro & Wilk (1965). Biometrika 52(3):591-611.
  D'Agostino & Pearson (1973). Biometrika 60(3):613-622.
  Iglewicz & Hoaglin (1993). How to Detect and Handle Outliers. ASQC.
  Nigrini (1996). Journal of the American Taxation Association 18:72-91.
  Cohen (1988). Statistical Power Analysis. 2nd ed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from core.audit_logger import _sanitize, compute_effect_size

logger = logging.getLogger(__name__)

BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
NIGRINI_MAD = {"Close": 0.006, "Acceptable": 0.012, "Marginal": 0.015}


# ---------------------------------------------------------------------------
# Yardımcı: güvenli numpy array üretici
# ---------------------------------------------------------------------------

def _to_arr(series: pd.Series) -> np.ndarray:
    """Pandas Series → temiz np.float64 array (NaN kaldırılmış, sıfırlanmış index)."""
    return np.array(
        pd.to_numeric(series, errors="coerce").dropna().values,
        dtype=np.float64
    )


def _desc(arr: np.ndarray) -> dict:
    return _sanitize({
        "n":      int(len(arr)),
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(arr.std()), 4),
        "var":    round(float(arr.var()), 4),
        "min":    round(float(arr.min()), 4),
        "max":    round(float(arr.max()), 4),
        "q1":     round(float(np.percentile(arr, 25)), 4),
        "q3":     round(float(np.percentile(arr, 75)), 4),
        "skew":   round(float(stats.skew(arr)), 4),
        "kurt":   round(float(stats.kurtosis(arr)), 4),
    })


# ---------------------------------------------------------------------------
# Bilirkişi Notu üretici
# ---------------------------------------------------------------------------

def _expert_note(is_normal: bool, skewness: float, kurtosis: float,
                 sample_size: int, lang: str = "tr") -> str:
    """
    Normallik testinin sonucuna göre hangi aykırı değer yönteminin
    kullanılması gerektiğini öneren danışman notu.

    Büyük örneklemlerde (N>5000) Shapiro-Wilk/D'Agostino p<0.05 çıkması
    istatistiksel güçten kaynaklanır; bu durum "hata" değil "bilgi" düzeyinde
    raporlanır.
    """
    abs_skew    = abs(skewness)
    heavy       = kurtosis > 3.0
    large_n     = sample_size > 5000   # büyük örneklem uyarısı

    if lang == "en":
        if large_n and not is_normal:
            prefix = (
                f"Note: With n={sample_size:,}, normality tests are highly "
                "sensitive to minor deviations. This is a methodological "
                "observation, not a critical error. "
            )
        else:
            prefix = ""

        if is_normal and abs_skew < 0.5:
            rec = prefix + "Classical Z-Score (±3 threshold) is appropriate."
        elif is_normal and abs_skew < 1.0:
            rec = prefix + "Mild skew detected; IQR (multiplier 1.5) is safer than Z-Score."
        elif heavy:
            rec = prefix + "Heavy-tailed distribution. Consider Robust Z-Score (MAD-based)."
        elif abs_skew >= 1.0:
            rec = prefix + "Right-skewed data. IQR or Robust Z-Score recommended over classical Z-Score."
        else:
            rec = prefix + "Non-normal distribution. IQR or Robust Z-Score preferred."

        if sample_size < 50:
            rec += f" Small sample (n={sample_size}) — interpret with caution."
    else:
        if large_n and not is_normal:
            prefix = (
                f"Bilgi: n={sample_size:,} büyük örneklemde normallik testleri "
                "küçük sapmaları da anlamlı bulur. Bu metodolojik bir gözlem; "
                "kritik hata değil. "
            )
        else:
            prefix = ""

        if is_normal and abs_skew < 0.5:
            rec = prefix + "Klasik Z-Score (±3 eşiği) uygundur."
        elif is_normal and abs_skew < 1.0:
            rec = prefix + "Hafif çarpıklık var; IQR (çarpan 1.5) önerilir."
        elif heavy:
            rec = prefix + "Kalın kuyruklu dağılım. Robust Z-Score (MAD tabanlı) değerlendirilebilir."
        elif abs_skew >= 1.0:
            rec = prefix + "Sağa çarpık dağılım. IQR veya Robust Z-Score daha güvenilir."
        else:
            rec = prefix + "Normal olmayan dağılım. IQR veya Robust Z-Score tercih edilebilir."

        if sample_size < 50:
            rec += f" Küçük örneklem (n={sample_size}) — sonuçlar ihtiyatla yorumlanmalı."

    return rec


# ---------------------------------------------------------------------------
# NormalityResult
# ---------------------------------------------------------------------------

@dataclass
class NormalityResult:
    test_name:    str
    statistic:    float
    p_value:      float
    is_normal:    bool
    alpha:        float
    sample_size:  int
    skewness:     float
    kurtosis:     float
    skew_label:   str
    expert_note:  str
    recommended_outlier_method: str  # "zscore" | "iqr" | "robust_zscore"

    def to_dict(self) -> dict:
        return _sanitize(self.__dict__)


def run_normality_test(
    series: pd.Series,
    alpha: float = 0.05,
    lang:  str   = "tr",
) -> NormalityResult:
    arr = _to_arr(series)
    n   = len(arr)
    if n < 3:
        raise ValueError(f"Normallik testi için en az 3 gözlem gerekli (mevcut: {n}).")

    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis

    if n < 5000:
        test_name = "Shapiro-Wilk"
        stat, p   = stats.shapiro(arr[:5000])
        # Normal alpha eşiği
        effective_alpha = alpha
    else:
        test_name = "D'Agostino-Pearson"
        stat, p   = stats.normaltest(arr)
        # Büyük örneklemde istatistiksel güç çok yüksek olduğundan
        # p<0.05 neredeyse garantili. Daha kısıtlayıcı eşik kullan:
        # N>5000 → sadece çok ciddi sapmalar (p<0.001) normallik ihlali say.
        effective_alpha = 0.001

    is_normal  = bool(float(p) > effective_alpha)
    abs_skew   = abs(skew)

    if abs_skew < 0.5:   skew_label = ("Simetrik"    if lang == "tr" else "Symmetric")
    elif abs_skew < 1.0: skew_label = ("Hafif Çarpık" if lang == "tr" else "Mildly Skewed")
    else:                skew_label = ("Çarpık"        if lang == "tr" else "Skewed")

    note = _expert_note(is_normal, skew, kurt, n, lang)

    if is_normal and abs_skew < 0.5:       rec_method = "zscore"
    elif abs(kurt) > 3 or abs_skew >= 1.0: rec_method = "robust_zscore"
    else:                                   rec_method = "iqr"

    return NormalityResult(
        test_name=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        is_normal=is_normal,
        alpha=float(effective_alpha),
        sample_size=int(n),
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        skew_label=skew_label,
        expert_note=note,
        recommended_outlier_method=rec_method,
    )


# ---------------------------------------------------------------------------
# Distribution Summary (histogram + box verisi)
# ---------------------------------------------------------------------------

@dataclass
class DistributionSummary:
    column:      str
    descriptive: dict
    hist_bins:   list[float]
    hist_counts: list[int]
    normality:   NormalityResult

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["normality"] = self.normality.to_dict()
        return _sanitize(d)


def compute_distribution(
    series: pd.Series,
    column_name: str = "",
    bins: int = 24,
    lang: str = "tr",
) -> DistributionSummary:
    arr = _to_arr(series)
    counts, edges = np.histogram(arr, bins=bins)
    norm = run_normality_test(series, lang=lang)
    return DistributionSummary(
        column=column_name,
        descriptive=_desc(arr),
        hist_bins=[round(float(e), 4) for e in edges],
        hist_counts=[int(c) for c in counts],
        normality=norm,
    )


# ---------------------------------------------------------------------------
# Benford's Law
# ---------------------------------------------------------------------------

@dataclass
class BenfordResult:
    column:        str
    observed_dist: dict   # digit → fraction
    expected_dist: dict
    observed_counts: dict # digit → count
    bar_data:      list   # [{digit, obs_pct, exp_pct, diff_pct, suspicious}]
    chi_square:    float
    p_value:       float
    dof:           int
    mad:           float
    conformity:    str
    is_suspicious: bool
    alpha:         float
    sample_size:   int
    descriptive:   dict
    expert_note:   str
    normality:     NormalityResult | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.normality:
            d["normality"] = self.normality.to_dict()
        return _sanitize(d)


def run_benford_test(
    series: pd.Series,
    column_name: str = "",
    alpha: float = 0.05,
    min_n: int = 100,
    lang: str = "tr",
) -> BenfordResult:
    arr_raw = pd.to_numeric(series, errors="coerce").dropna()
    arr_pos = arr_raw[arr_raw > 0]
    n       = int(len(arr_pos))
    if n < min_n:
        raise ValueError(f"Benford için en az {min_n} pozitif sayı gerekli (mevcut: {n}).")

    def first_digit(x: float) -> int:
        s = f"{abs(x):.10e}"
        for c in s:
            if c.isdigit() and c != "0":
                return int(c)
        return 0

    digits       = arr_pos.apply(first_digit)
    digit_counts = {d: int(digits.value_counts().get(d, 0)) for d in range(1, 10)}
    obs_dist     = {d: digit_counts[d] / n for d in range(1, 10)}

    obs_freq = np.array([digit_counts[d] for d in range(1, 10)], dtype=np.float64)
    exp_freq = np.array([BENFORD_EXPECTED[d] * n for d in range(1, 10)], dtype=np.float64)
    chi_sq, p_val = stats.chisquare(f_obs=obs_freq, f_exp=exp_freq)

    mad = float(np.mean(np.abs(
        np.array(list(obs_dist.values())) -
        np.array(list(BENFORD_EXPECTED.values()))
    )))

    if mad <= NIGRINI_MAD["Close"]:       conformity = "Close conformity"
    elif mad <= NIGRINI_MAD["Acceptable"]: conformity = "Acceptable conformity"
    elif mad <= NIGRINI_MAD["Marginal"]:   conformity = "Marginal conformity"
    else:                                   conformity = "Nonconformity"

    # Bar chart verisi — her basamak için gözlenen/beklenen/fark
    bar_data = []
    for d in range(1, 10):
        obs_pct = round(obs_dist[d] * 100, 2)
        exp_pct = round(BENFORD_EXPECTED[d] * 100, 2)
        diff    = round(obs_pct - exp_pct, 2)
        bar_data.append({
            "digit":      d,
            "obs_pct":    obs_pct,
            "exp_pct":    exp_pct,
            "diff_pct":   diff,
            "suspicious": abs(diff) > 3.0,
        })

    arr_clean = _to_arr(series)
    desc = _desc(arr_clean) if len(arr_clean) > 0 else {}

    # BÜYÜK VERİDE CHI-SQUARE YANILTICIDIR. Kararı Nigrini'nin MAD (Mean Absolute Deviation) eşiğine bağlıyoruz.
    # Eğer uyum "Marginal" (0.015) sınırından daha kötüyse şüpheli (is_suspicious = True) say.
    is_suspicious_flag = mad > NIGRINI_MAD["Marginal"]

    note = (
        f"{'İncelenmeli: Benford uyumsuzluğu (MAD eşiği aşıldı)' if is_suspicious_flag else 'Benford dağılımıyla uyumlu'} "
        f"(χ²={chi_sq:.2f}, p={float(p_val):.4f}, MAD={mad:.4f}). "
        f"Uyum düzeyi: {conformity}."
    )

    try:
        norm = run_normality_test(series, lang=lang)
    except Exception:
        norm = None

    return BenfordResult(
        column=column_name,
        observed_dist={int(k): round(float(v), 6) for k, v in obs_dist.items()},
        expected_dist={int(k): round(float(v), 6) for k, v in BENFORD_EXPECTED.items()},
        observed_counts={int(k): int(v) for k, v in digit_counts.items()},
        bar_data=[_sanitize(b) for b in bar_data],
        chi_square=round(float(chi_sq), 4),
        p_value=round(float(p_val), 6),
        dof=8,
        mad=round(mad, 6),
        conformity=conformity,
        is_suspicious=is_suspicious_flag, # KESKİN DÜZELTME
        alpha=float(alpha),
        sample_size=int(n),
        descriptive=desc,
        expert_note=note,
        normality=norm,
    )


# ---------------------------------------------------------------------------
# IQR Outlier Detection
# ---------------------------------------------------------------------------

@dataclass
class OutlierResult:
    column:          str
    method:          str
    threshold:       float
    outlier_indices: list
    outlier_values:  list
    normal_sample:   list
    outlier_count:   int
    total_count:     int
    outlier_rate:    float
    lower_bound:     float
    upper_bound:     float
    descriptive:     dict
    normality:       NormalityResult | None
    effect_size:     dict
    expert_note:     str
    distribution:    DistributionSummary | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.normality:    d["normality"]    = self.normality.to_dict()
        if self.distribution: d["distribution"] = self.distribution.to_dict()
        return _sanitize(d)


def run_iqr_outlier_detection(
    series: pd.Series,
    column_name: str = "",
    multiplier: float = 1.5,
    lang: str = "tr",
) -> OutlierResult:
    arr = _to_arr(series)
    n   = len(arr)
    s   = pd.Series(arr)

    q1  = float(np.percentile(arr, 25))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lo  = q1 - multiplier * iqr
    hi  = q3 + multiplier * iqr

    mask      = (s < lo) | (s > hi)
    norm_mask = ~mask

    try:    norm_res = run_normality_test(series, lang=lang)
    except: norm_res = None

    try:    dist = compute_distribution(series, column_name, lang=lang)
    except: dist = None

    before = s
    after  = s[norm_mask]
    eff    = compute_effect_size(before, after)

    note   = norm_res.expert_note if norm_res else ""

    return OutlierResult(
        column=column_name, method="IQR",
        threshold=float(multiplier),
        outlier_indices=[int(i) for i in s[mask].index.tolist()],
        outlier_values=[round(float(v), 4) for v in s[mask].tolist()],
        normal_sample=[round(float(v), 4) for v in s[norm_mask].head(50).tolist()],
        outlier_count=int(mask.sum()),
        total_count=int(n),
        outlier_rate=round(float(mask.mean() * 100), 2),
        lower_bound=round(lo, 4),
        upper_bound=round(hi, 4),
        descriptive=_desc(arr),
        normality=norm_res,
        effect_size=eff,
        expert_note=note,
        distribution=dist,
    )


def run_zscore_outlier_detection(
    series: pd.Series,
    column_name: str = "",
    threshold: float = 3.0,
    robust: bool = True,
    lang: str = "tr",
) -> OutlierResult:
    arr = _to_arr(series)
    n   = len(arr)
    s   = pd.Series(arr)

    median = float(np.median(arr))

    if robust:
        mad_val = float(np.median(np.abs(arr - median)))
        if mad_val == 0:
            std_ = float(arr.std())
            z    = np.abs((arr - median) / std_) if std_ > 0 else np.zeros(n)
        else:
            z    = np.abs(0.6745 * (arr - median) / mad_val)
        center, scale = median, mad_val if mad_val != 0 else float(arr.std())
        method = "Robust Z-Score (MAD)"
    else:
        mean_, std_ = float(arr.mean()), float(arr.std())
        z    = np.abs((arr - mean_) / std_) if std_ > 0 else np.zeros(n)
        center, scale = mean_, std_
        method = "Z-Score (Klasik)"

    z_series  = pd.Series(z)
    mask      = z_series > threshold
    norm_mask = ~mask

    try:    norm_res = run_normality_test(series, lang=lang)
    except: norm_res = None

    try:    dist = compute_distribution(series, column_name, lang=lang)
    except: dist = None

    eff  = compute_effect_size(s, s[norm_mask])
    note = norm_res.expert_note if norm_res else ""

    return OutlierResult(
        column=column_name, method=method,
        threshold=float(threshold),
        outlier_indices=[int(i) for i in s[mask].index.tolist()],
        outlier_values=[round(float(v), 4) for v in s[mask].tolist()],
        normal_sample=[round(float(v), 4) for v in s[norm_mask].head(50).tolist()],
        outlier_count=int(mask.sum()),
        total_count=int(n),
        outlier_rate=round(float(mask.mean() * 100), 2),
        lower_bound=round(float(center - threshold * scale), 4),
        upper_bound=round(float(center + threshold * scale), 4),
        descriptive=_desc(arr),
        normality=norm_res,
        effect_size=eff,
        expert_note=note,
        distribution=dist,
    )


# ---------------------------------------------------------------------------
# Variance Impact Analysis
# ---------------------------------------------------------------------------

@dataclass
class VarianceResult:
    target_col:    str
    group_col:     str
    eta_squared:   float
    omega_squared: float
    f_statistic:   float
    p_value:       float
    df_between:    int
    df_within:     int
    group_means:   dict
    group_sizes:   dict
    effect:        str
    is_significant: bool
    normality_by_group: dict
    expert_note:   str

    def to_dict(self) -> dict:
        return _sanitize(self.__dict__)


def run_variance_impact_analysis(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    alpha: float = 0.05,
    lang: str = "tr",
) -> VarianceResult:
    sub = df[[target_col, group_col]].copy()
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna()

    groups    = sub.groupby(group_col)[target_col]
    arrays    = [np.array(g.values, dtype=np.float64) for _, g in groups]
    group_keys = [str(k) for k, _ in groups]

    if len(arrays) < 2:
        raise ValueError("Varyans analizi için en az 2 grup gerekli.")

    f, p         = stats.f_oneway(*arrays)
    grand_mean   = float(sub[target_col].mean())
    ss_total     = float(np.sum((sub[target_col] - grand_mean) ** 2))
    ss_between   = float(np.sum([len(g) * (g.mean() - grand_mean) ** 2 for g in arrays]))
    ss_within    = ss_total - ss_between
    k, n_total   = len(arrays), int(len(sub))
    df_b, df_w   = k - 1, n_total - k
    ms_w         = ss_within / df_w if df_w > 0 else 0.0
    eta_sq       = float(ss_between / ss_total) if ss_total > 0 else 0.0
    omega_sq     = max(0.0, float((ss_between - df_b * ms_w) / (ss_total + ms_w))) if ss_total > 0 else 0.0

    if eta_sq < 0.01:   effect = "Ihmal Edilebilir"
    elif eta_sq < 0.06: effect = "Küçük"
    elif eta_sq < 0.14: effect = "Orta"
    else:               effect = "Büyük"

    norm_by_group = {}
    for gkey, garr in zip(group_keys, arrays):
        try:
            nr = run_normality_test(pd.Series(garr), lang=lang)
            norm_by_group[gkey] = nr.to_dict()
        except Exception:
            norm_by_group[gkey] = {"error": "yetersiz veri"}

    note = (
        f"η²={eta_sq:.4f} ({effect} etki). "
        f"{'İstatistiksel olarak anlamlı' if float(p) < alpha else 'Anlamlı değil'} "
        f"(F={float(f):.2f}, p={float(p):.4f}). "
        f"ω²={omega_sq:.4f}."
    )

    return VarianceResult(
        target_col=target_col, group_col=group_col,
        eta_squared=round(eta_sq, 4), omega_squared=round(omega_sq, 4),
        f_statistic=round(float(f), 4), p_value=round(float(p), 6),
        df_between=int(df_b), df_within=int(df_w),
        group_means={str(k): round(float(v.mean()), 4) for k, v in zip(group_keys, arrays)},
        group_sizes={str(k): int(len(v)) for k, v in zip(group_keys, arrays)},
        effect=effect, is_significant=bool(float(p) < alpha),
        normality_by_group=norm_by_group,
        expert_note=note,
    )
