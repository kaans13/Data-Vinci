"""
fuzzy_matcher.py — Data-Autopsy v4.1

Değişiklikler:
  - rapidfuzz/thefuzz yoksa saf Python fallback (difflib + pure Jaro-Winkler)
  - import hatası yerine graceful degradation
  - Büyük veri için chunk-based işleme (>5000 kayıt)
  - merge_on_match_result suffix çakışma düzeltmesi
"""
from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Kütüphane önceliği: rapidfuzz > thefuzz > saf Python ───────────────────
_BACKEND = "pure"

try:
    from rapidfuzz import fuzz as _fuzz, process as _process
    _BACKEND = "rapidfuzz"
    logger.info("fuzzy_matcher: rapidfuzz kullanılıyor")
except ImportError:
    try:
        from thefuzz import fuzz as _fuzz, process as _process
        _BACKEND = "thefuzz"
        logger.info("fuzzy_matcher: thefuzz kullanılıyor")
    except ImportError:
        _fuzz = None
        _process = None
        logger.info("fuzzy_matcher: saf Python fallback (difflib + Jaro-Winkler)")


class MatchAlgorithm(str, Enum):
    LEVENSHTEIN  = "levenshtein"
    JARO_WINKLER = "jaro_winkler"
    TOKEN_SORT   = "token_sort"


@dataclass
class MatchPair:
    primary_idx:     int
    primary_value:   str
    secondary_idx:   int
    secondary_value: str
    score:           float   # 0–100
    status:          str     # "matched" | "review" | "unmatched"


@dataclass
class MatchResult:
    matched:             list[MatchPair]
    unmatched_primary:   list[int]
    unmatched_secondary: list[int]
    algorithm:           str
    threshold:           float
    total_primary:       int
    total_secondary:     int
    duration_ms:         float
    match_rate_pct:      float
    backend:             str = _BACKEND

    @property
    def matched_count(self) -> int:
        return len(self.matched)


# ── Saf Python Algoritmaları ────────────────────────────────────────────────

def _levenshtein_ratio(s1: str, s2: str) -> float:
    """difflib SequenceMatcher — Levenshtein'a yakın, stdlib."""
    return difflib.SequenceMatcher(None, s1, s2).ratio() * 100


def _jaro(s1: str, s2: str) -> float:
    """Jaro benzerliği — saf Python."""
    if s1 == s2: return 100.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0: return 0.0
    match_dist = max(l1, l2) // 2 - 1
    s1m = [False] * l1; s2m = [False] * l2
    matches = 0
    for i in range(l1):
        lo = max(0, i - match_dist); hi = min(i + match_dist + 1, l2)
        for j in range(lo, hi):
            if s2m[j] or s1[i] != s2[j]: continue
            s1m[i] = s2m[j] = True; matches += 1; break
    if matches == 0: return 0.0
    trans = 0; k = 0
    for i in range(l1):
        if not s1m[i]: continue
        while not s2m[k]: k += 1
        if s1[i] != s2[k]: trans += 1
        k += 1
    return (matches/l1 + matches/l2 + (matches - trans/2)/matches) / 3 * 100


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler — önek bonusu ekli."""
    j = _jaro(s1, s2) / 100
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2: prefix += 1
        else: break
    return (j + prefix * p * (1 - j)) * 100


def _token_sort(s1: str, s2: str) -> float:
    """Token sıralı benzerlik — kelime sırasından bağımsız."""
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio() * 100


# ── Algoritma Seçici ────────────────────────────────────────────────────────

def _score(s1: str, s2: str, algo: MatchAlgorithm) -> float:
    """İki string arasında seçilen algoritmayla skor hesaplar (0-100)."""
    if _BACKEND in ("rapidfuzz", "thefuzz"):
        if algo == MatchAlgorithm.LEVENSHTEIN:
            return float(_fuzz.ratio(s1, s2))
        elif algo == MatchAlgorithm.JARO_WINKLER:
            fn = getattr(_fuzz, "jaro_winkler_similarity", None) or _fuzz.WRatio
            v = fn(s1, s2)
            return float(v * 100 if v <= 1.0 else v)
        elif algo == MatchAlgorithm.TOKEN_SORT:
            return float(_fuzz.token_sort_ratio(s1, s2))
        return float(_fuzz.WRatio(s1, s2))
    else:
        if algo == MatchAlgorithm.LEVENSHTEIN:  return _levenshtein_ratio(s1, s2)
        elif algo == MatchAlgorithm.JARO_WINKLER: return _jaro_winkler(s1, s2)
        elif algo == MatchAlgorithm.TOKEN_SORT:   return _token_sort(s1, s2)
        return _jaro_winkler(s1, s2)


def _best_match_pure(
    query: str, candidates: list[str], algo: MatchAlgorithm,
    cutoff: float,
) -> tuple[int, float] | None:
    """
    Saf Python iki aşamalı eşleştirme:
      1. Prefix (3 karakter) hızlı ön eleme
      2. Kalan adaylar üzerinde tam skor hesapla

    Prefix eşleşmesi yoksa quick_ratio ile en iyi 15 aday seçilir.
    """
    if not candidates:
        return None

    q_prefix = query[:3].lower() if len(query) >= 3 else query.lower()

    # Adayları prefix ile filtrele
    prefix_pool = [
        i for i, c in enumerate(candidates)
        if c[:3].lower() == q_prefix
    ]

    if not prefix_pool:
        # Prefix eşleşmesi yok → quick_ratio ile en yakın 15'i bul
        scored = sorted(
            ((i, difflib.SequenceMatcher(None, query, c, autojunk=False).quick_ratio())
             for i, c in enumerate(candidates)),
            key=lambda x: x[1], reverse=True,
        )[:15]
        prefix_pool = [i for i, _ in scored]

    # Pool üzerinde tam skor hesapla
    best_pos, best_score = -1, -1.0
    for i in prefix_pool:
        s = _score(query, candidates[i], algo)
        if s > best_score:
            best_score = s; best_pos = i

    # Prefix dışında da daha iyi eşleşme olabilir — top genel aday kontrol
    if best_score < cutoff:
        for i, c in enumerate(candidates):
            if i in prefix_pool:
                continue
            # Hızlı ön kontrol: length farkı fazlaysa atla
            if abs(len(c) - len(query)) > max(len(query), len(c)) * 0.5:
                continue
            s = _score(query, c, algo)
            if s > best_score:
                best_score = s; best_pos = i

    if best_pos >= 0 and best_score >= cutoff:
        return best_pos, best_score
    return None


# ── Ana Fonksiyon ────────────────────────────────────────────────────────────

def run_fuzzy_match(
    primary:      pd.Series,
    secondary:    pd.Series,
    algorithm:    MatchAlgorithm | str = MatchAlgorithm.JARO_WINKLER,
    threshold:    float = 80.0,
    review_zone:  float = 10.0,
    preprocessor=None,
    chunk_size:   int = 2000,
) -> MatchResult:
    """
    İki string serisini bulanık eşleştirme ile karşılaştırır.

    chunk_size: büyük veri setlerinde primary'yi parçalara böler;
                bellek verimliliği için kullanılır.
    """
    t0 = time.perf_counter()
    alg = MatchAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm
    prep = preprocessor or _default_preprocessor
    cutoff = max(0.0, threshold - review_zone)

    # Ön işleme
    p_clean = primary.fillna("").apply(prep)
    s_clean = secondary.fillna("").apply(prep)
    s_list  = s_clean.tolist()
    s_orig  = secondary.tolist()
    p_index = list(primary.index)
    s_index = list(secondary.index)

    matched: list[MatchPair] = []
    matched_s_pos: set[int] = set()

    # Chunk işleme — büyük primary setleri için
    for chunk_start in range(0, len(p_clean), chunk_size):
        p_chunk     = p_clean.iloc[chunk_start:chunk_start + chunk_size]
        p_chunk_idx = p_index[chunk_start:chunk_start + chunk_size]

        for local_i, (p_idx, p_val) in enumerate(zip(p_chunk_idx, p_chunk)):
            if not p_val:
                continue

            # Backend'e göre en iyi eşleşmeyi bul
            if _BACKEND in ("rapidfuzz", "thefuzz"):
                try:
                    scorer = _get_lib_scorer(alg)
                    res = _process.extractOne(
                        p_val, s_list, scorer=scorer,
                        score_cutoff=cutoff,
                    )
                    if res is None:
                        continue
                    # rapidfuzz: (match, score, index) | thefuzz: (match, score)
                    if len(res) == 3:
                        best_val, best_score, best_pos = res
                    else:
                        best_val, best_score = res
                        best_pos = s_list.index(best_val)
                    best_score = float(best_score)
                    # Jaro-Winkler rapidfuzz'da 0-1 döner
                    if best_score <= 1.0:
                        best_score *= 100
                except Exception:
                    res2 = _best_match_pure(p_val, s_list, alg, cutoff)
                    if res2 is None: continue
                    best_pos, best_score = res2
            else:
                res2 = _best_match_pure(p_val, s_list, alg, cutoff)
                if res2 is None:
                    continue
                best_pos, best_score = res2

            best_score = float(best_score)
            if best_score < cutoff:
                continue

            status = "matched" if best_score >= threshold else "review"
            matched.append(MatchPair(
                primary_idx     = p_idx,
                primary_value   = str(primary.iloc[p_index.index(p_idx)]),
                secondary_idx   = s_index[best_pos],
                secondary_value = str(s_orig[best_pos]),
                score           = round(best_score, 2),
                status          = status,
            ))
            matched_s_pos.add(best_pos)

    matched_p_set = {m.primary_idx for m in matched}
    unmatched_p   = [i for i in p_index if i not in matched_p_set]
    unmatched_s   = [s_index[i] for i in range(len(s_list))
                     if i not in matched_s_pos]

    dur  = (time.perf_counter() - t0) * 1000
    rate = len(matched) / max(len(primary), 1) * 100

    logger.info("Eşleştirme (%s, backend=%s, eşik=%.0f): %d/%d, %.1fms",
                alg.value, _BACKEND, threshold, len(matched), len(primary), dur)

    return MatchResult(
        matched=matched, unmatched_primary=unmatched_p,
        unmatched_secondary=unmatched_s, algorithm=alg.value,
        threshold=threshold, total_primary=len(primary),
        total_secondary=len(secondary), duration_ms=round(dur, 2),
        match_rate_pct=round(rate, 2), backend=_BACKEND,
    )


def merge_on_match_result(
    primary_df:   pd.DataFrame,
    secondary_df: pd.DataFrame,
    match_result: MatchResult,
    suffix_primary:   str = "_p",
    suffix_secondary: str = "_s",
) -> pd.DataFrame:
    """Eşleştirme sonucunu kullanarak iki DataFrame'i birleştirir."""
    records = []
    p_cols = set(primary_df.columns)
    s_cols = set(secondary_df.columns)
    conflict = p_cols & s_cols

    for pair in match_result.matched:
        try:
            p_row = primary_df.loc[pair.primary_idx].to_dict()
            s_row = secondary_df.loc[pair.secondary_idx].to_dict()
        except KeyError:
            continue
        merged: dict = {}
        for col, val in p_row.items():
            merged[col + suffix_primary if col in conflict else col] = val
        for col, val in s_row.items():
            merged[col + suffix_secondary if col in conflict else col] = val
        merged["_match_score"]  = pair.score
        merged["_match_status"] = pair.status
        records.append(merged)

    return pd.DataFrame(records)


# ── Yardımcılar ─────────────────────────────────────────────────────────────

def _get_lib_scorer(algo: MatchAlgorithm):
    if algo == MatchAlgorithm.LEVENSHTEIN:
        return _fuzz.ratio
    elif algo == MatchAlgorithm.JARO_WINKLER:
        return getattr(_fuzz, "jaro_winkler_similarity",
                       getattr(_fuzz, "WRatio", _fuzz.ratio))
    elif algo == MatchAlgorithm.TOKEN_SORT:
        return _fuzz.token_sort_ratio
    return _fuzz.WRatio


def _default_preprocessor(text: str) -> str:
    if not text:
        return ""
    text = (text
            .replace("İ", "i").replace("I", "ı")
            .replace("Ğ", "ğ").replace("Ü", "ü")
            .replace("Ş", "ş").replace("Ö", "ö").replace("Ç", "ç")
            .lower())
    return re.sub(r"\s+", " ", text).strip()
