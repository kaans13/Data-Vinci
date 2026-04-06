"""
anomaly_radar.py — Data-Autopsy  Modül 2: Anomali Radarı

Algoritma seçim kuralı:
  n_samples < 1000  veya yüksek boyutluluk (>10 sütun)  → IsolationForest
  n_samples ≥ 1000  ve  ≤ 10 sütun                       → LOF + IF karşılaştırma

SHAP bağımlılığı yok — SHAP'a özgü mantığı TreeExplainer yerine
"conditional mean ablation" yöntemiyle kopyalıyoruz:
  contribution(feature_j, row_i) = score(row_i with feature_j set to median)
                                  − score(row_i with original feature_j)
  Yüksek pozitif → o feature anomali skoruna katkıda bulunuyor.

Referans: Ribeiro et al. (2016) "Why Should I Trust You?" — LIME yaklaşımı
          Scott et al. (2017) Shapley Value decomposition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ── Veri sınıfları ───────────────────────────────────────────────────────────

class AnomalyAlgorithm(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOF              = "lof"
    ENSEMBLE         = "ensemble"   # her ikisi de, consensus gerektirir


@dataclass
class AnomalyExplanation:
    """Tek bir anomalinin açıklaması."""
    row_index:         int
    anomaly_score:     float     # Isolation Forest: daha negatif → daha anormal
    lof_score:         float | None
    feature_contributions: dict[str, float]   # feature → katkı skoru
    top_features:      list[str]              # en fazla katkı yapan sütunlar
    z_scores:          dict[str, float]       # ham z-skor (hızlı bakış)
    verdict:           str                    # "Anomali"
    explanation:       str                    # Türkçe doğal dil
    row_values:        dict[str, Any]         # o satırın değerleri


@dataclass
class AnomalyReport:
    """AnomalyRadar.detect() dönüş değeri."""
    algorithm:       str
    n_total:         int
    n_anomalies:     int
    contamination:   float
    feature_cols:    list[str]
    anomalies:       list[AnomalyExplanation]
    global_importance: dict[str, float]   # hangi feature en çok anomaliye katkıda bulunuyor

    def to_dict(self) -> dict:
        return {
            "algorithm":      self.algorithm,
            "n_total":        self.n_total,
            "n_anomalies":    self.n_anomalies,
            "contamination":  self.contamination,
            "features":       self.feature_cols,
            "global_importance": self.global_importance,
            "anomalies": [
                {
                    "row_index":   a.row_index,
                    "score":       a.anomaly_score,
                    "top_features": a.top_features,
                    "contributions": a.feature_contributions,
                    "z_scores":    a.z_scores,
                    "verdict":     a.verdict,
                    "explanation": a.explanation,
                    "values":      a.row_values,
                }
                for a in self.anomalies
            ],
        }


# ── Ana sınıf ────────────────────────────────────────────────────────────────

class AnomalyRadar:
    """
    Çok boyutlu anomali tespiti.

    Kullanım:
        radar  = AnomalyRadar(contamination=0.05)
        report = radar.detect(df, feature_cols=["maas","yas","skor"])
        for a in report.anomalies:
            print(a.explanation)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        algorithm: AnomalyAlgorithm = AnomalyAlgorithm.ISOLATION_FOREST,
        random_state: int = 42,
        n_estimators: int = 100,
        n_neighbors_lof: int = 20,
        top_k_features: int = 3,
        max_explain: int = 100,   # En fazla bu kadar anomali için feature contribution hesapla
    ):
        self.contamination    = contamination
        self.algorithm        = algorithm
        self.random_state     = random_state
        self.n_estimators     = n_estimators
        self.n_neighbors_lof  = n_neighbors_lof
        self.top_k_features   = top_k_features
        self.max_explain      = max_explain

    # ── Genel API ────────────────────────────────────────────────────────

    def detect(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> AnomalyReport:
        """
        DataFrame üzerinde anomali tespiti ve açıklama üretir.

        feature_cols: None → tüm sayısal sütunlar otomatik seçilir.
        """
        # Sütun seçimi
        if feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]
        if len(feature_cols) < 2:
            raise ValueError(
                "Anomali tespiti için en az 2 sayısal sütun gerekli."
            )

        # Eksik değer sütununu kapat, temiz alt set
        sub = df[feature_cols].copy()
        sub = sub.apply(pd.to_numeric, errors="coerce")

        # Eksik değerleri medyan ile doldur (tespit öncesi minimal imputation)
        for col in sub.columns:
            median_val = sub[col].median()
            sub[col] = sub[col].fillna(median_val)

        n, d  = sub.shape
        algo  = self._select_algorithm(n, d)
        logger.info("AnomalyRadar: n=%d, d=%d, algo=%s", n, d, algo.value)

        # Normalizasyon (RobustScaler — aykırı değerlerden etkilenmez)
        scaler   = RobustScaler()
        X_scaled = scaler.fit_transform(sub.values)

        # Model eğit
        if_model   = None
        lof_model  = None
        if_scores  = None
        lof_scores = None

        if algo in (AnomalyAlgorithm.ISOLATION_FOREST, AnomalyAlgorithm.ENSEMBLE):
            if_model  = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if_preds  = if_model.fit_predict(X_scaled)
            if_scores = if_model.score_samples(X_scaled)

        if algo in (AnomalyAlgorithm.LOF, AnomalyAlgorithm.ENSEMBLE):
            lof_model  = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors_lof, n // 5 or 5),
                contamination=self.contamination,
                novelty=False,
                n_jobs=-1,
            )
            lof_preds  = lof_model.fit_predict(X_scaled)
            lof_scores = -lof_model.negative_outlier_factor_

        # Anomali indekslerini belirle
        if algo == AnomalyAlgorithm.ISOLATION_FOREST:
            anomaly_mask = if_preds == -1
        elif algo == AnomalyAlgorithm.LOF:
            anomaly_mask = lof_preds == -1
        else:
            # Ensemble: her iki modelin de anomali dediği satırlar
            anomaly_mask = (if_preds == -1) & (lof_preds == -1)

        anomaly_idx = np.where(anomaly_mask)[0]
        logger.info("Tespit edilen anomali: %d/%d", len(anomaly_idx), n)

        # Z-skorlar (ham — hızlı referans)
        z_matrix = _compute_zscores(sub)

        # Feature contribution: max_explain ile sınırla, tümü için batch hesapla
        explain_idx = anomaly_idx[:self.max_explain]
        d = len(feature_cols)

        if if_model is not None and len(explain_idx) > 0:
            # Tüm anomaliler için pertürbasyon matrisini tek seferde oluştur
            # Shape: (n_explain × (d+1), d)
            n_exp = len(explain_idx)
            all_perturb = np.zeros((n_exp * (d + 1), d))
            for k, idx in enumerate(explain_idx):
                base_row = X_scaled[idx]
                offset   = k * (d + 1)
                all_perturb[offset] = base_row              # orijinal
                for j in range(d):
                    all_perturb[offset + j + 1] = base_row.copy()
                    all_perturb[offset + j + 1, j] = 0.0   # feature_j → medyan

            all_scores = if_model.score_samples(all_perturb)  # tek çağrı
        else:
            all_scores = None

        explanations = []
        for k, idx in enumerate(explain_idx):
            if all_scores is not None:
                offset   = k * (d + 1)
                base_s   = float(all_scores[offset])
                perturbed = all_scores[offset + 1: offset + d + 1]
                
                # DOĞRU MANTIK: Sütun çıkarıldığında skor ne kadar ARTTI (normale döndü)?
                # Artış pozitifse, o sütun anomaliye sebep oluyordur.
                raw       = perturbed - base_s 
                
                max_abs   = max(abs(raw).max(), 1e-8)
                norm      = (raw / max_abs).tolist()
                contribs  = {col: round(float(v), 4) for col, v in zip(feature_cols, norm)}
            else:
                contribs  = {col: 0.0 for col in feature_cols}

            z_row = {col: round(float(z_matrix[col].iloc[idx]), 3) for col in feature_cols}
            lof_s = float(lof_scores[idx]) if lof_scores is not None else None
            if_s  = float(if_scores[idx])  if if_scores  is not None else 0.0

            top_feats = sorted(contribs, key=lambda k: contribs[k], reverse=True)[:self.top_k_features]
            row_vals  = {col: _safe_val(df[col].iloc[idx]) for col in feature_cols}
            expl = _build_explanation(idx, if_s, lof_s, contribs, top_feats, z_row, row_vals, algo)
            explanations.append(expl)

        # max_explain dışı anomaliler için basit explanation (contribution hesabı yok)
        for idx in anomaly_idx[self.max_explain:]:
            z_row = {col: round(float(z_matrix[col].iloc[idx]), 3) for col in feature_cols}
            lof_s = float(lof_scores[idx]) if lof_scores is not None else None
            if_s  = float(if_scores[idx])  if if_scores  is not None else 0.0
            empty_c = {col: 0.0 for col in feature_cols}
            expl = _build_explanation(idx, if_s, lof_s, empty_c, [], z_row,
                                       {col: _safe_val(df[col].iloc[idx]) for col in feature_cols},
                                       algo)
            explanations.append(expl)

        # Küresel feature importance: ortalamaları
        global_imp: dict[str, float] = {}
        if explanations:
            for col in feature_cols:
                global_imp[col] = round(
                    float(np.mean([
                        e.feature_contributions.get(col, 0.0)
                        for e in explanations
                    ])), 4,
                )
        global_imp = dict(
            sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
        )

        return AnomalyReport(
            algorithm=algo.value,
            n_total=n,
            n_anomalies=len(anomaly_idx),
            contamination=self.contamination,
            feature_cols=feature_cols,
            anomalies=explanations,
            global_importance=global_imp,
        )

    # ── Algoritma seçimi ─────────────────────────────────────────────────

    def _select_algorithm(self, n: int, d: int) -> AnomalyAlgorithm:
        if self.algorithm != AnomalyAlgorithm.ENSEMBLE:
            return self.algorithm
            
        # Boyut (d) çok yüksekse mesafe metrikleri (LOF) çöker (Curse of Dimensionality)
        if d > 15:
            return AnomalyAlgorithm.ISOLATION_FOREST
            
        # Örneklem küçükse IF ağaçları ayrım yapamaz, Lokal Yoğunluk (LOF) en iyi seçimdir.
        if n < 500:
            return AnomalyAlgorithm.LOF
            
        return AnomalyAlgorithm.ENSEMBLE


# ── Feature contribution (SHAP-free) ────────────────────────────────────────

def _feature_contributions(
    model: IsolationForest | None,
    X_scaled: np.ndarray,
    idx: int,
    feature_cols: list[str],
) -> dict[str, float]:
    """
    Vektörize feature contribution (SHAP-free).

    Tüm pertürbasyonları tek np.ndarray'e yığıp
    tek score_samples() çağrısıyla hesaplar.

    Önceki: d ayrı score_samples çağrısı
    Şimdi:  1 score_samples çağrısı (d+1 satırlık batch)
    """
    if model is None:
        return {col: 0.0 for col in feature_cols}

    d   = len(feature_cols)
    row = X_scaled[idx:idx + 1]          # (1, d)
    base = float(model.score_samples(row)[0])

    # (d+1, d) matris: ilk satır orijinal, sonrakiler her feature sıfırlanmış
    batch        = np.tile(row, (d + 1, 1))   # (d+1, d)
    for j in range(d):
        batch[j + 1, j] = 0.0               # feature_j → medyan(=0 scaled)

    scores = model.score_samples(batch)     # tek çağrı
    base_check = float(scores[0])           # tutarlılık kontrolü

    contribs = {}
    for j, col in enumerate(feature_cols):
        delta          = float(scores[j + 1]) - base
        contribs[col]  = round(float(delta) * -1, 4)   # pozitif → anomaliye katkı

    # Normalize: max abs = 1
    max_abs = max(abs(v) for v in contribs.values()) or 1.0
    return {k: round(v / max_abs, 4) for k, v in contribs.items()}


def _compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std() + 1e-8)


def _safe_val(v: Any) -> Any:
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    if v != v: return None   # NaN
    return v


# ── Doğal dil açıklama üretici ───────────────────────────────────────────────

def _build_explanation(
    idx: int,
    if_score: float,
    lof_score: float | None,
    contribs: dict[str, float],
    top_feats: list[str],
    z_scores: dict[str, float],
    row_vals: dict[str, Any],
    algo: AnomalyAlgorithm,
) -> AnomalyExplanation:
    # Anomali şiddeti
    if if_score < -0.15:
        severity = "yüksek şiddetli"
    elif if_score < -0.05:
        severity = "orta şiddetli"
    else:
        severity = "düşük şiddetli"

    # Temel açıklama
    parts = [f"Satır {idx}: {severity} anomali tespit edildi."]

    # Feature katkısına dayalı açıklama
    if top_feats:
        feat_parts = []
        for feat in top_feats:
            val  = row_vals.get(feat, "?")
            z    = z_scores.get(feat, 0.0)
            c    = contribs.get(feat, 0.0)
            if abs(z) > 3:
                direction = "aşırı yüksek" if z > 0 else "aşırı düşük"
                feat_parts.append(
                    f"'{feat}' değeri {direction} ({val}, z={z:.2f})"
                )
            elif abs(c) > 0.3:
                feat_parts.append(
                    f"'{feat}' normal dağılım dışında ({val}, katkı={c:.3f})"
                )
        if feat_parts:
            parts.append("Sebepler: " + "; ".join(feat_parts) + ".")

    # Sütunlar arası uyumsuzluk
    if len(top_feats) >= 2:
        f1, f2 = top_feats[0], top_feats[1]
        z1, z2 = z_scores.get(f1, 0), z_scores.get(f2, 0)
        if z1 * z2 < 0:   # zıt yönde sapmalar → uyumsuzluk
            parts.append(
                f"'{f1}' ve '{f2}' arasında korelasyon uyumsuzluğu var "
                f"(biri yüksek, diğeri düşük)."
            )

    if lof_score is not None and lof_score > 2.0:
        parts.append(
            f"LOF skoru {lof_score:.2f} — lokal yoğunluk açısından da aykırı."
        )

    algo_name = {
        AnomalyAlgorithm.ISOLATION_FOREST: "Isolation Forest",
        AnomalyAlgorithm.LOF:              "LOF",
        AnomalyAlgorithm.ENSEMBLE:         "IF+LOF Ensemble",
    }.get(algo, algo.value)
    parts.append(f"[{algo_name}, IF skor={if_score:.4f}]")

    return AnomalyExplanation(
        row_index=int(idx),
        anomaly_score=round(if_score, 4),
        lof_score=round(lof_score, 4) if lof_score is not None else None,
        feature_contributions={k: round(v, 4) for k, v in contribs.items()},
        top_features=top_feats,
        z_scores=z_scores,
        verdict="Anomali",
        explanation=" ".join(parts),
        row_values=row_vals,
    )
