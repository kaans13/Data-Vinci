"""
hypothesis_tester.py — Data-Autopsy  Modül 1: Otonom Hipotez Fabrikası

Karar ağacı:
    ┌── 2 Grup
    │     ├── Sürekli × Sürekli → Pearson / Spearman
    │     ├── Sürekli × Kategorik(2) → T-Test / Mann-Whitney U
    │     └── Kategorik × Kategorik → Chi-Kare / Fisher
    └── 3+ Grup
          ├── Sürekli → ANOVA / Kruskal-Wallis
          └── Kategorik → Chi-Kare

Normallik eşiği:   p > 0.05 → normal
Homojenlik eşiği:  p > 0.05 → varyans homojen
n < 5000           → Shapiro-Wilk
n ≥ 5000           → D'Agostino-Pearson
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA = 0.05   # anlamlılık düzeyi


# ── Tip sistemi ──────────────────────────────────────────────────────────────

class VarType(str, Enum):
    CONTINUOUS   = "continuous"
    CATEGORICAL  = "categorical"
    BINARY       = "binary"       # 2 kategorili özel durum


class TestName(str, Enum):
    TTEST_IND       = "Bağımsız Örneklem T-Testi"
    TTEST_WELCH     = "Welch T-Testi (eşitsiz varyans)"
    ANOVA           = "Tek Yönlü ANOVA"
    MANN_WHITNEY    = "Mann-Whitney U Testi"
    KRUSKAL_WALLIS  = "Kruskal-Wallis H Testi"
    CHI_SQUARE      = "Ki-Kare Bağımsızlık Testi"
    FISHER_EXACT    = "Fisher Kesin Testi"
    PEARSON         = "Pearson Korelasyon"
    SPEARMAN        = "Spearman Rank Korelasyon"
    POINT_BISERIAL  = "Point-Biserial Korelasyon"


@dataclass
class NormalityResult:
    test_name:  str
    statistic:  float
    p_value:    float
    is_normal:  bool
    n:          int


@dataclass
class HypothesisResult:
    """Tek bir testin çıktısı."""
    test_name:        str
    statistic:        float
    p_value:          float
    degrees_of_freedom: float | None
    effect_size:      float | None
    effect_label:     str           # "Küçük / Orta / Büyük"
    is_significant:   bool
    alpha:            float

    # Karar yolu bilgisi
    normality:        dict[str, NormalityResult]   # grup → normallik
    variance_homogeneous: bool | None              # Levene sonucu
    selected_reason:  str                          # neden bu test seçildi

    # Doğal dil çıktısı
    verdict:          str   # "Anlamlı fark VAR / YOK"
    explanation:      str   # tam Türkçe açıklama
    recommendation:   str   # sonraki adım önerisi


@dataclass
class FullTestReport:
    """AutoHypothesisTester.test() dönüş değeri."""
    var_x_name:   str
    var_y_name:   str
    var_x_type:   VarType
    var_y_type:   VarType
    n_total:      int
    result:       HypothesisResult
    raw_data_summary: dict[str, Any]   # grup istatistikleri

    def to_dict(self) -> dict:
        r = self.result
        return {
            "variables":   {"x": self.var_x_name, "y": self.var_y_name},
            "types":       {"x": self.var_x_type.value, "y": self.var_y_type.value},
            "n":           self.n_total,
            "test":        r.test_name,
            "statistic":   r.statistic,
            "p_value":     r.p_value,
            "significant": r.is_significant,
            "effect_size": r.effect_size,
            "effect_label":r.effect_label,
            "verdict":     r.verdict,
            "explanation": r.explanation,
            "recommendation": r.recommendation,
            "decision_path": {
                "normality":   {k: v.__dict__ for k, v in r.normality.items()},
                "homogeneous_variance": r.variance_homogeneous,
                "reason":      r.selected_reason,
            },
            "group_summary": self.raw_data_summary,
        }


# ── Ana sınıf ────────────────────────────────────────────────────────────────

class AutoHypothesisTester:
    """
    Kullanıcı istatistik bilmeden değişken çifti seçer,
    uygun test otomatik seçilir ve sonuç Türkçe açıklanır.

    Kullanım:
        tester = AutoHypothesisTester()
        report = tester.test(df, x_col="maas", y_col="departman")
        print(report.result.explanation)
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha

    # ── Genel API ────────────────────────────────────────────────────────

    def test(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
    ) -> FullTestReport:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Sütun bulunamadı: {x_col!r} veya {y_col!r}")

        x = df[x_col].dropna()
        y = df[y_col].dropna()
        n = min(len(x), len(y))

        vx = _infer_type(x)
        vy = _infer_type(y)

        logger.info("Hipotez testi: %s(%s) ~ %s(%s) | n=%d",
                    x_col, vx.value, y_col, vy.value, n)

        # Hangi yol?
        result = self._dispatch(x, y, vx, vy, x_col, y_col)
        summary = self._group_summary(x, y, vx, vy)

        return FullTestReport(
            var_x_name=x_col, var_y_name=y_col,
            var_x_type=vx,    var_y_type=vy,
            n_total=n, result=result,
            raw_data_summary=summary,
        )

    def test_multiple(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
    ) -> list[FullTestReport]:
        """Hedef değişken ile birden fazla feature arasında toplu test."""
        reports = []
        for col in feature_cols:
            try:
                reports.append(self.test(df, target_col, col))
            except Exception as exc:
                logger.warning("Test başarısız %s ~ %s: %s", target_col, col, exc)
        # p-değerine göre sırala
        reports.sort(key=lambda r: r.result.p_value)
        return reports

    # ── Dispatcher ───────────────────────────────────────────────────────

    def _dispatch(
        self,
        x: pd.Series, y: pd.Series,
        vx: VarType, vy: VarType,
        xn: str, yn: str,
    ) -> HypothesisResult:
        # Her iki değişken de sürekli → korelasyon
        if vx == VarType.CONTINUOUS and vy == VarType.CONTINUOUS:
            return self._correlation(x, y, xn, yn)

        # Sürekli × Kategorik (veya ikili)
        if vx == VarType.CONTINUOUS and vy in (VarType.CATEGORICAL, VarType.BINARY):
            return self._continuous_vs_categorical(x, y, xn, yn)
        if vy == VarType.CONTINUOUS and vx in (VarType.CATEGORICAL, VarType.BINARY):
            return self._continuous_vs_categorical(y, x, yn, xn)

        # İkisi de kategorik → Chi-Kare
        if vx in (VarType.CATEGORICAL, VarType.BINARY) and \
           vy in (VarType.CATEGORICAL, VarType.BINARY):
            return self._categorical_vs_categorical(x, y, xn, yn)

        raise ValueError(f"Desteklenmeyen tip kombinasyonu: {vx} × {vy}")

    # ── Sürekli × Sürekli (Korelasyon) ───────────────────────────────────

    def _correlation(
        self, x: pd.Series, y: pd.Series,
        xn: str, yn: str,
    ) -> HypothesisResult:
        n   = len(x)
        nr  = _normality(x, "x")
        normal = nr.is_normal and _normality(y, "y").is_normal

        if normal and n >= 30:
            r, p   = stats.pearsonr(x, y)
            tname  = TestName.PEARSON.value
            reason = "Her iki dağılım normal → Pearson korelasyon."
            eff    = abs(float(r))
        else:
            r, p   = stats.spearmanr(x, y)
            tname  = TestName.SPEARMAN.value
            reason = "Normal dağılım varsayımı sağlanamadı → Spearman sıra korelasyonu."
            eff    = abs(float(r))

        eff_lbl = _corr_effect_label(eff)
        sig     = bool(p < self.alpha)

        direction = "pozitif" if float(r) > 0 else "negatif"
        verdict   = "Anlamlı ilişki VAR" if sig else "Anlamlı ilişki YOK"
        expl = (
            f"{tname} uygulandı (n={n}). "
            f"Korelasyon katsayısı r={r:.4f} ({direction}), p={p:.4f}. "
            f"{'İstatistiksel olarak anlamlı bir' if sig else 'Anlamlı bir'} "
            f"{eff_lbl.lower()} ilişki "
            f"{'tespit edildi' if sig else 'tespit edilemedi'} "
            f"(α={self.alpha})."
        )
        rec = _correlation_recommendation(eff, sig, direction)

        return HypothesisResult(
            test_name=tname, statistic=round(float(r), 4), p_value=round(float(p), 6),
            degrees_of_freedom=n - 2, effect_size=round(eff, 4),
            effect_label=eff_lbl, is_significant=sig, alpha=self.alpha,
            normality={xn: nr}, variance_homogeneous=None,
            selected_reason=reason, verdict=verdict,
            explanation=expl, recommendation=rec,
        )

    # ── Sürekli × Kategorik ───────────────────────────────────────────────

    def _continuous_vs_categorical(
        self, cont: pd.Series, cat: pd.Series,
        cont_name: str, cat_name: str,
    ) -> HypothesisResult:
        groups    = {str(k): cont[cat == k].dropna().values for k in cat.unique()}
        groups    = {k: v for k, v in groups.items() if len(v) >= 3}
        n_groups  = len(groups)

        if n_groups < 2:
            raise ValueError("Test için en az 2 grup gerekli.")

        group_arrays = list(groups.values())
        group_keys   = list(groups.keys())

        # Normallik ve CLT kontrolü
        norm_results = {}
        all_normal   = True
        min_n        = min(len(arr) for arr in group_arrays)
        
        for k, arr in groups.items():
            nr = _normality(pd.Series(arr), k)
            norm_results[k] = nr
            if not nr.is_normal:
                all_normal = False

        # Merkezi Limit Teoremi: N>=30 ise normallik varsayımı esnetilebilir.
        clt_applies = min_n >= 30
        use_parametric = all_normal or clt_applies

        # Varyans homojenliği (Levene)
        lev_stat, lev_p = stats.levene(*group_arrays)
        homogeneous     = bool(lev_p > self.alpha)

        # Test seçimi
        if n_groups == 2:
            arr1, arr2 = group_arrays[0], group_arrays[1]
            if use_parametric:
                if homogeneous:
                    stat, p = stats.ttest_ind(arr1, arr2, equal_var=True)
                    tname   = TestName.TTEST_IND.value
                    reason  = ("Normallik ✓ (veya N>=30 CLT) + Varyans Homojen ✓ → Bağımsız T-Testi.")
                else:
                    stat, p = stats.ttest_ind(arr1, arr2, equal_var=False)
                    tname   = TestName.TTEST_WELCH.value
                    reason  = ("Varyanslar heterojen (Levene p<0.05) → Welch T-Testi (Daha güvenilir).")
                eff = _cohens_d(arr1, arr2)
                eff_lbl = _d_effect_label(abs(eff))
            else:
                stat, p = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
                tname   = TestName.MANN_WHITNEY.value
                reason  = ("Küçük örneklem (N<30) ve normal değil → Mann-Whitney U.")
                eff = _rank_biserial(arr1, arr2, float(stat))
                eff_lbl = _r_effect_label(abs(eff))
            dof = _ttest_dof(arr1, arr2, homogeneous)
        else:
            # 3+ grup
            if use_parametric and homogeneous:
                stat, p = stats.f_oneway(*group_arrays)
                tname   = TestName.ANOVA.value
                reason  = ("Normallik ✓ (veya CLT) + Varyans Homojen ✓ → Tek Yönlü ANOVA.")
                eff = _eta_squared(group_arrays, float(stat))
                eff_lbl = _eta_effect_label(eff)
            else:
                # Scipy'de Welch ANOVA olmadığı için Kruskal'a düşüyoruz ama uyarı ekleyerek.
                stat, p = stats.kruskal(*group_arrays)
                tname   = TestName.KRUSKAL_WALLIS.value
                reason  = ("Parametrik varsayımlar tam sağlanamadı → Kruskal-Wallis H Testi.")
                eff = _epsilon_squared(group_arrays, float(stat))
                eff_lbl = _eta_effect_label(eff)
            dof = None

        sig     = bool(p < self.alpha)
        verdict = f"Gruplar arasında {cont_name} açısından anlamlı fark " + ("VAR" if sig else "YOK")
        expl    = _build_explanation_groups(
            tname, groups, cont_name, cat_name,
            float(stat), float(p), eff, eff_lbl, sig, self.alpha,
        )
        rec     = _group_recommendation(sig, n_groups, eff_lbl)

        return HypothesisResult(
            test_name=tname, statistic=round(float(stat), 4), p_value=round(float(p), 6),
            degrees_of_freedom=dof, effect_size=round(float(eff), 4), effect_label=eff_lbl,
            is_significant=sig, alpha=self.alpha, normality=norm_results, 
            variance_homogeneous=homogeneous, selected_reason=reason, 
            verdict=verdict, explanation=expl, recommendation=rec,
        )

    # ── Kategorik × Kategorik ─────────────────────────────────────────────

    def _categorical_vs_categorical(
        self, x: pd.Series, y: pd.Series,
        xn: str, yn: str,
    ) -> HypothesisResult:
        ct = pd.crosstab(x, y)
        n  = int(ct.values.sum())

        # Önce Ki-Kare'yi çalıştırıp Beklenen (Expected) frekansları alıyoruz
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        
        # Fisher Kesin Testi için gerçek kural: Tablo 2x2 olmalı VE
        # Beklenen frekansların (gözlenen değil) en az biri 5'ten küçük olmalı.
        if ct.shape == (2, 2) and (expected < 5).any():
            res       = stats.fisher_exact(ct.values)
            stat, p   = float(res.statistic), float(res.pvalue)
            tname     = TestName.FISHER_EXACT.value
            reason    = "2×2 tablo + Beklenen frekans < 5 → Fisher Kesin Testi."
            eff       = _odds_ratio_effect(stat)
            eff_lbl   = f"OR={stat:.3f}"
        else:
            stat     = float(chi2)
            p        = float(p)
            tname    = TestName.CHI_SQUARE.value
            reason   = f"Kategorik × Kategorik → Ki-Kare (dof={dof})."
            eff      = _cramers_v(stat, n, ct.shape)
            eff_lbl  = _cramers_effect_label(eff)

        sig     = bool(p < self.alpha)
        verdict = (f"'{xn}' ile '{yn}' arasında anlamlı bir ilişki " + ("VAR" if sig else "YOK"))
        expl    = (
            f"{tname} uygulandı (n={n}, tablo {ct.shape[0]}×{ct.shape[1]}). "
            f"İstatistik={stat:.4f}, p={p:.4f}. "
            f"{'Anlamlı bir bağımlılık' if sig else 'Anlamlı bir bağımlılık'} "
            f"{'tespit edildi' if sig else 'tespit edilemedi'} (α={self.alpha}). "
            f"Etki büyüklüğü: {eff_lbl}."
        )
        rec = ("Post-hoc analiz veya ayrıntılı çapraz tablo incelemesi önerilir." if sig else
               "Değişkenler arasında bağımsızlık varsayımı korunabilir.")

        return HypothesisResult(
            test_name=tname, statistic=round(stat, 4), p_value=round(float(p), 6),
            degrees_of_freedom=None, effect_size=round(float(eff), 4),
            effect_label=eff_lbl, is_significant=sig, alpha=self.alpha,
            normality={}, variance_homogeneous=None, selected_reason=reason, 
            verdict=verdict, explanation=expl, recommendation=rec,
        )

    # ── Özet istatistikler ────────────────────────────────────────────────

    def _group_summary(
        self, x: pd.Series, y: pd.Series,
        vx: VarType, vy: VarType,
    ) -> dict:
        summary: dict = {}
        if vy in (VarType.CATEGORICAL, VarType.BINARY):
            for grp in y.unique():
                sub = x[y == grp].dropna()
                if len(sub) == 0:
                    continue
                if pd.api.types.is_numeric_dtype(sub):
                    summary[str(grp)] = {
                        "n":      int(len(sub)),
                        "mean":   round(float(sub.mean()), 4),
                        "median": round(float(sub.median()), 4),
                        "std":    round(float(sub.std()), 4),
                        "min":    round(float(sub.min()), 4),
                        "max":    round(float(sub.max()), 4),
                    }
                else:
                    vc = sub.value_counts()
                    summary[str(grp)] = {
                        "n":     int(len(sub)),
                        "top":   str(vc.index[0]) if len(vc) else "",
                        "unique":int(sub.nunique()),
                    }
        else:
            def _safe_stats(s: pd.Series, label: str) -> dict:
                if pd.api.types.is_numeric_dtype(s):
                    return {"n": int(len(s)), "mean": round(float(s.mean()), 4),
                            "std": round(float(s.std()), 4)}
                return {"n": int(len(s)), "unique": int(s.nunique())}
            summary["x"] = _safe_stats(x, "x")
            summary["y"] = _safe_stats(y, "y")
        return summary


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _infer_type(s: pd.Series, max_unique_ratio: float = 0.05) -> VarType:
    if pd.api.types.is_numeric_dtype(s):
        u = s.nunique()
        if u <= 2:
            return VarType.BINARY
        if u / len(s) < max_unique_ratio and u <= 20:
            return VarType.CATEGORICAL
        return VarType.CONTINUOUS
    return VarType.BINARY if s.nunique() == 2 else VarType.CATEGORICAL


def _normality(s: pd.Series, label: str) -> NormalityResult:
    arr  = np.array(pd.to_numeric(s, errors="coerce").dropna(), dtype=np.float64)
    n    = len(arr)
    if n < 3:
        return NormalityResult("Yetersiz veri", 0.0, 1.0, True, n)
    if n < 5000:
        stat, p = stats.shapiro(arr[:5000])
        tname   = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(arr)
        tname   = "D'Agostino-Pearson"
    return NormalityResult(
        test_name=tname, statistic=round(float(stat), 4),
        p_value=round(float(p), 6), is_normal=bool(float(p) > ALPHA), n=n,
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def _rank_biserial(a: np.ndarray, b: np.ndarray, u: float) -> float:
    return float(1 - (2 * u) / (len(a) * len(b)))


def _eta_squared(arrays: list[np.ndarray], f_stat: float) -> float:
    total     = np.concatenate(arrays)
    grand_m   = float(total.mean())
    ss_total  = float(np.sum((total - grand_m) ** 2))
    ss_between= float(sum(len(g) * (np.mean(g) - grand_m) ** 2 for g in arrays))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _epsilon_squared(arrays: list[np.ndarray], h: float) -> float:
    n = sum(len(g) for g in arrays)
    return float((h - len(arrays) + 1) / (n - len(arrays))) if n > len(arrays) else 0.0


def _cramers_v(chi2: float, n: int, shape: tuple) -> float:
    k = min(shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0


def _odds_ratio_effect(or_: float) -> float:
    return float(abs(np.log(or_))) if or_ > 0 else 0.0


def _ttest_dof(a: np.ndarray, b: np.ndarray, homogeneous: bool) -> float:
    if homogeneous:
        return float(len(a) + len(b) - 2)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    num    = (va/na + vb/nb) ** 2
    den    = (va/na)**2/(na-1) + (vb/nb)**2/(nb-1)
    return float(num/den) if den > 0 else float(na + nb - 2)


# Etki büyüklüğü etiketleri
def _d_effect_label(d: float) -> str:
    if d < 0.2:  return "İhmal Edilebilir"
    if d < 0.5:  return "Küçük"
    if d < 0.8:  return "Orta"
    return "Büyük"

def _r_effect_label(r: float) -> str:
    if r < 0.1:  return "İhmal Edilebilir"
    if r < 0.3:  return "Küçük"
    if r < 0.5:  return "Orta"
    return "Büyük"

def _eta_effect_label(eta: float) -> str:
    if eta < 0.01: return "İhmal Edilebilir"
    if eta < 0.06: return "Küçük"
    if eta < 0.14: return "Orta"
    return "Büyük"

def _corr_effect_label(r: float) -> str:
    if r < 0.10:  return "İhmal Edilebilir İlişki"
    if r < 0.30:  return "Zayıf İlişki"
    if r < 0.50:  return "Orta Güçte İlişki"
    if r < 0.70:  return "Güçlü İlişki"
    return "Çok Güçlü İlişki"

def _cramers_effect_label(v: float) -> str:
    if v < 0.1:  return "Zayıf İlişki"
    if v < 0.3:  return "Orta İlişki"
    return "Güçlü İlişki"


def _build_explanation_groups(
    tname: str, groups: dict, cont: str, cat: str,
    stat: float, p: float, eff: float, eff_lbl: str,
    sig: bool, alpha: float,
) -> str:
    g_means = ", ".join(
        f"{k}(ort={np.mean(v):.2f}, n={len(v)})"
        for k, v in groups.items()
    )
    return (
        f"{tname} uygulandı. Gruplar: {g_means}. "
        f"Test istatistiği={stat:.4f}, p={p:.6f}. "
        f"{'Gruplar arasında' if sig else 'Gruplar arasında istatistiksel olarak'} "
        f"'{cont}' değişkeni açısından "
        f"{'anlamlı bir fark tespit edildi' if sig else 'anlamlı bir fark tespit edilemedi'} "
        f"(α={alpha}). Etki büyüklüğü {eff_lbl.lower()} (={eff:.4f})."
    )


def _correlation_recommendation(eff: float, sig: bool, direction: str) -> str:
    if not sig:
        return "Değişkenler arasında istatistiksel ilişki yok; başka faktörler araştırılabilir."
    if eff > 0.5:
        return (f"Güçlü {direction} ilişki: Bir değişkeni diğerinin tahmincisi "
                "olarak kullanmayı değerlendirin (regresyon analizi önerilir).")
    if eff > 0.3:
        return (f"Orta {direction} ilişki: Kontrol değişkeni olarak "
                "çok değişkenli analizde kullanılabilir.")
    return "Zayıf ilişki: Pratik önemi sınırlı olabilir, örneklem boyutunu artırmayı düşünün."


def _group_recommendation(sig: bool, n_groups: int, eff_lbl: str) -> str:
    if not sig:
        return "Gruplar arasında anlamlı fark yok; gruplandırma değişkenini gözden geçirin."
    if n_groups > 2:
        return ("Post-hoc analiz (Tukey HSD veya Dunn's testi) ile hangi "
                "grup çiftlerinin farklı olduğu belirlenmelidir.")
    return (f"İki grup anlamlı biçimde farklı. Etki büyüklüğü '{eff_lbl}' — "
            "pratik önem için güç analizi önerilir.")
