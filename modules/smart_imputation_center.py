"""
smart_imputation_center.py — Data-Autopsy  Modül 3: Akıllı Veri Onarım İstasyonu

İki katmanlı süreç:
  1. Doldurma: KNN veya MICE (IterativeImputer)
  2. Doğrulama: Wasserstein + KS + Varyans Oranı → Dağılım Bozulma Skoru

Ceza skoru (P) yorumu:
  P < 0.05  → Dağılım mükemmel korundu (Grade A)
  P < 0.15  → Kabul edilebilir bozulma    (Grade B)
  P < 0.30  → Dikkat — alternatif dene   (Grade C)
  P < 0.50  → Yüksek bozulma             (Grade D)
  P ≥ 0.50  → Doldurma tavsiye edilmez   (Grade F)

%10 varyans sapma eşiği: |VKO − 1| > 0.10 → Warning otomatik eklenir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

VARIANCE_WARNING_THRESHOLD = 0.10   # %10 sapma → uyarı


class ImputationMethod(str, Enum):
    KNN   = "knn"
    MICE  = "mice"
    MEAN  = "mean"
    MEDIAN= "median"


@dataclass
class DistributionPenalty:
    """Tek sütun için dağılım bozulma metriği."""
    column:          str
    method:          str
    n_filled:        int

    # Metrikler
    ks_stat:         float
    ks_pvalue:       float
    wasserstein_raw: float
    wasserstein_norm: float    # std ile normalize edilmiş
    var_ratio:       float     # var_after / var_before
    mean_shift_pct:  float     # (mean_after - mean_before) / mean_before × 100

    # Bileşik skor
    penalty_score:   float     # 0=iyi, 1=kötü
    grade:           str
    verdict:         str
    warning:         str       # boş ise sorun yok


@dataclass
class ImputationResult:
    """SmartImputationCenter.impute() dönüş değeri."""
    df_original:   pd.DataFrame
    df_imputed:    pd.DataFrame
    method:        ImputationMethod
    columns_filled: list[str]
    n_total_filled: int
    penalties:     dict[str, DistributionPenalty]
    overall_grade: str
    summary:       dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "method":          self.method.value,
            "columns_filled":  self.columns_filled,
            "n_total_filled":  self.n_total_filled,
            "overall_grade":   self.overall_grade,
            "penalties": {
                col: {
                    "grade":         p.grade,
                    "penalty_score": p.penalty_score,
                    "verdict":       p.verdict,
                    "warning":       p.warning,
                    "ks_stat":       p.ks_stat,
                    "wasserstein_norm": p.wasserstein_norm,
                    "var_ratio":     p.var_ratio,
                    "mean_shift_pct":p.mean_shift_pct,
                    "n_filled":      p.n_filled,
                }
                for col, p in self.penalties.items()
            },
            "summary": self.summary,
        }


class SmartImputationCenter:
    """
    Tek API — akıllı eksik veri doldurma.

    Kullanım:
        center = SmartImputationCenter(method=ImputationMethod.MICE)
        result = center.impute(df, target_cols=["maas", "yas"])
        print(result.to_dict())

    Otomatik mod:
        center = SmartImputationCenter()
        result = center.impute(df)   # tüm eksik sayısal sütunlar
    """

    def __init__(
        self,
        method: ImputationMethod = ImputationMethod.MICE,
        n_neighbors: int = 5,
        max_iter: int = 10,
        random_state: int = 42,
    ):
        self.method       = method
        self.n_neighbors  = n_neighbors
        self.max_iter     = max_iter
        self.random_state = random_state

    # ── Ana API ──────────────────────────────────────────────────────────

    def impute(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
        auto_select_method: bool = False,
    ) -> ImputationResult:
        """
        Eksik veriyi doldurur ve dağılım bozulma skoru hesaplar.

        auto_select_method=True → KNN vs MICE karşılaştırır, daha iyi olanı seçer.
        """
        if auto_select_method:
            return self._auto_best(df, target_cols)

        num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()
        ]
        if target_cols is not None:
            num_cols = [c for c in target_cols if c in num_cols]

        if not num_cols:
            raise ValueError("Doldurulacak eksik değer bulunamadı.")

        logger.info("SmartImputer: %s, %d sütun", self.method.value, len(num_cols))

        # Büyük veri uyarısı
        n = len(df)
        if n > 50_000 and self.method == ImputationMethod.KNN:
            logger.warning("Büyük veri (n=%d) + KNN → yavaş olabilir. "
                           "MICE veya MEDIAN önerilir.", n)

        # Doldurma öncesi snapshot
        before_stats = {
            col: _series_stats(df[col]) for col in num_cols
        }
        n_missing_before = {col: int(df[col].isna().sum()) for col in num_cols}

        # Doldur
        df_filled = self._apply_method(df, num_cols)

        # Penalty hesapla
        penalties: dict[str, DistributionPenalty] = {}
        for col in num_cols:
            before_vals = df[col].dropna().values
            after_vals  = df_filled[col].dropna().values
            if len(before_vals) < 5 or len(after_vals) < 5:
                continue
            penalties[col] = _compute_penalty(
                col=col,
                method=self.method.value,
                n_filled=n_missing_before[col],
                before=before_vals,
                after=after_vals,
            )

        overall = _aggregate_grade(penalties)
        total_filled = sum(p.n_filled for p in penalties.values())

        summary = {
            "method":       self.method.value,
            "n_rows":       n,
            "columns":      num_cols,
            "total_filled": total_filled,
            "overall_grade":overall,
            "per_column": {
                col: {
                    "before_mean": round(before_stats[col]["mean"], 4),
                    "before_std":  round(before_stats[col]["std"],  4),
                    "n_missing":   n_missing_before[col],
                    "penalty":     penalties[col].penalty_score
                    if col in penalties else None,
                    "warning":     penalties[col].warning
                    if col in penalties else "",
                }
                for col in num_cols
            },
        }

        return ImputationResult(
            df_original=df,
            df_imputed=df_filled,
            method=self.method,
            columns_filled=num_cols,
            n_total_filled=total_filled,
            penalties=penalties,
            overall_grade=overall,
            summary=summary,
        )

    # ── Metod uygulama ───────────────────────────────────────────────────

    def _apply_method(
        self, df: pd.DataFrame, target_cols: list[str]
    ) -> pd.DataFrame:
        df_out  = df.copy()
        
        # Zehirli sütun filtreleme: Tamamen benzersiz olan (ID) veya hiç değişmeyen (sabit) sayısal sütunları dışla
        all_num = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                n_unique = df[c].nunique()
                if n_unique == 1: continue  # Sıfır varyans
                if n_unique == len(df) and pd.api.types.is_integer_dtype(df[c]): continue # ID / Index şüphesi
                all_num.append(c)

        # Hedef sütunlar ID gibi görünse bile doldurulması istenmişse geri ekle
        for c in target_cols:
            if c not in all_num and pd.api.types.is_numeric_dtype(df[c]):
                all_num.append(c)

        if self.method == ImputationMethod.MEAN:
            for col in target_cols:
                df_out[col] = df_out[col].fillna(df_out[col].mean())
            return df_out

        if self.method == ImputationMethod.MEDIAN:
            for col in target_cols:
                df_out[col] = df_out[col].fillna(df_out[col].median())
            return df_out

        sub      = df[all_num].apply(pd.to_numeric, errors="coerce")
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(sub.values)
        X_df     = pd.DataFrame(X_scaled, columns=all_num, index=df.index)

        if self.method == ImputationMethod.KNN:
            imputer = KNNImputer(
                n_neighbors=min(self.n_neighbors, max(3, len(df) // 10)),
                weights="distance",
            )
        else:
            imputer = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state,
                tol=1e-3,
                verbose=0,
            )

        X_filled  = imputer.fit_transform(X_df.values)
        X_inv     = scaler.inverse_transform(X_filled)
        df_filled = pd.DataFrame(X_inv, columns=all_num, index=df.index)

        # Integer Tipi Koruması: Orijinal veri tamsayıysa, float küsuratları tam sayıya yuvarla
        for col in target_cols:
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                df_out[col] = np.round(df_filled[col])
            else:
                df_out[col] = df_filled[col]

        return df_out

    # ── Otomatik metod seçimi ────────────────────────────────────────────

    def _auto_best(
        self, df: pd.DataFrame, target_cols: list[str] | None
    ) -> ImputationResult:
        """KNN ve MICE'ı karşılaştır, daha düşük toplam penalty'yi seç."""
        results = []
        for m in (ImputationMethod.KNN, ImputationMethod.MICE):
            center = SmartImputationCenter(
                method=m, n_neighbors=self.n_neighbors,
                max_iter=self.max_iter, random_state=self.random_state,
            )
            try:
                r = center.impute(df, target_cols)
                avg_p = np.mean([p.penalty_score for p in r.penalties.values()]) \
                        if r.penalties else 1.0
                results.append((avg_p, r))
                logger.info("Auto: %s avg_penalty=%.4f", m.value, avg_p)
            except Exception as exc:
                logger.warning("Auto: %s başarısız: %s", m.value, exc)

        if not results:
            raise RuntimeError("Hiçbir metod çalışmadı.")

        best = min(results, key=lambda x: x[0])[1]
        logger.info("Auto seçim: %s", best.method.value)
        return best


# ── Dağılım Bozulma Skoru ────────────────────────────────────────────────────

def _compute_penalty(
    col: str, method: str, n_filled: int,
    before: np.ndarray, after: np.ndarray,
) -> DistributionPenalty:
    sb, sa = float(before.std()), float(after.std())
    mb, ma = float(before.mean()), float(after.mean())

    # Kolmogorov-Smirnov
    ks_stat, ks_p = ks_2samp(before, after)

    # Wasserstein
    raw_w  = float(wasserstein_distance(before, after))
    w_norm = min(1.0, raw_w / (sb + 1e-8))

    # Varyans oranı
    var_b  = sb ** 2
    var_a  = sa ** 2
    vr     = float(var_a / (var_b + 1e-8))

    # Ortalama kayması
    ms_pct = float((ma - mb) / (abs(mb) + 1e-8) * 100)

    # Bileşik ceza: KS × 0.40 + W_norm × 0.40 + |VR−1| × 0.20
    vr_pen = min(1.0, abs(np.log(vr + 1e-8) / np.log(2)))
    P = float(np.clip(0.40 * ks_stat + 0.40 * w_norm + 0.20 * vr_pen, 0, 1))

    grade, verdict = _penalty_grade(P)

    # Varyans uyarısı
    warning = ""
    if abs(1.0 - vr) > VARIANCE_WARNING_THRESHOLD:
        direction = "azaldı" if vr < 1 else "arttı"
        warning = (
            f"Warning: Veri kalitesi düştü, orijinal dağılım bozuldu. "
            f"'{col}' sütununda varyans %{abs(1-vr)*100:.1f} oranında {direction}. "
            f"Orijinal dağılım korunmadı."
        )

    return DistributionPenalty(
        column=col, method=method, n_filled=n_filled,
        ks_stat=round(float(ks_stat), 4), ks_pvalue=round(float(ks_p), 4),
        wasserstein_raw=round(raw_w, 4), wasserstein_norm=round(w_norm, 4),
        var_ratio=round(vr, 4), mean_shift_pct=round(ms_pct, 2),
        penalty_score=round(P, 4), grade=grade, verdict=verdict, warning=warning,
    )


def _penalty_grade(P: float) -> tuple[str, str]:
    if P < 0.05:
        return "A", "Dağılım mükemmel korundu."
    if P < 0.15:
        return "B", "Dağılım büyük ölçüde korundu."
    if P < 0.30:
        return "C", "Orta düzey bozulma — kabul edilebilir."
    if P < 0.50:
        return "D", "Yüksek bozulma — farklı yöntem deneyin."
    return "F", "Dağılım ciddi şekilde bozuldu — bu doldurmayı kullanmayın."


def _aggregate_grade(penalties: dict[str, DistributionPenalty]) -> str:
    if not penalties:
        return "N/A"
    grades = [p.grade for p in penalties.values()]
    order  = {"A":0,"B":1,"C":2,"D":3,"F":4,"N/A":5}
    return max(grades, key=lambda g: order.get(g, 5))


def _series_stats(s: pd.Series) -> dict:
    num = pd.to_numeric(s, errors="coerce").dropna()
    if len(num) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "mean":   float(num.mean()),
        "std":    float(num.std()),
        "median": float(num.median()),
    }
