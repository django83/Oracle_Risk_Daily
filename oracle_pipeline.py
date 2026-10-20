# =============================================================================================
# Oracle Macro-Micro v9.4 — version GitHub Actions (headless) + notification email
# v9.4 : EW entraîné sur les vraies baisses du S&P (et non sur le label BGMM), stress sans courbe/taux,
#         repli du seuil WFO à 0.50 quand aucun seuil n'est exploitable
# Exécution : python oracle_macro.py
# Variables d'environnement : voir README.md
# =============================================================================================

import os
import sys
import time
import html
import json
import ssl
import smtplib
import logging
import warnings
import traceback
from datetime import datetime
from email.message import EmailMessage

import yaml
import requests
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from scipy.optimize import linear_sum_assignment

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
from tqdm import tqdm

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MacroPipeline")

VERSION = "9.4"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  0. CONFIGURATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FMP_API_KEY = (os.getenv("FMP_API_KEY") or "").strip() or None

os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.yaml")

DEFAULT_CONFIG = {
    "pipeline": {
        "start_date": "2005-01-01",
        "n_regimes_max": 4,
        "min_train_days": 1260,
        "window_size": 1260,
        "refit_freq": 21,
        "full_refit_every_n_days": 63,
        "coverage_threshold": 0.60,
        "pca_variance_target": 0.95,
        "max_staleness_days": 45,
        "rolling_imputation_window": 252,
        "recalc_tail_days": 5,
        "crisis_stress_threshold": 0.70,
        "min_cluster_weight": 0.02,
        "late_start_limit": "2008-06-01",
        # NOUVEAU v9.4 : l'Early Warning prédit une baisse >= X% du S&P dans les 21 prochains jours
        "ew_drawdown_threshold": 0.05
    },
    "fred_series": {
        "Credit_Spread": "BAA10Y",
        "Yield_Curve": "T10Y2Y",
        "Long_Rate": "DGS10",
        "NFCI": "NFCI",
        "Fed_BS": "WALCL"
    }
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f) or {}

config_updated = False
for section, values in DEFAULT_CONFIG.items():
    if section not in CFG:
        CFG[section] = values
        config_updated = True
    elif isinstance(values, dict):
        for key, val in values.items():
            if key not in CFG[section]:
                CFG[section][key] = val
                config_updated = True

if config_updated:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(CFG, f, default_flow_style=False)

START_DATE              = CFG["pipeline"].get("start_date", "2005-01-01")
N_REGIMES_MAX           = CFG["pipeline"].get("n_regimes_max", 4)
MIN_TRAIN_DAYS          = CFG["pipeline"].get("min_train_days", 1260)
WINDOW_SIZE             = CFG["pipeline"].get("window_size", 1260)
REFIT_FREQ              = CFG["pipeline"].get("refit_freq", 21)
FULL_REFIT_EVERY_N_DAYS = CFG["pipeline"].get("full_refit_every_n_days", 63)
COVERAGE_THRESHOLD      = CFG["pipeline"].get("coverage_threshold", 0.60)
PCA_VARIANCE_TARGET     = CFG["pipeline"].get("pca_variance_target", 0.95)
MAX_STALENESS_DAYS      = CFG["pipeline"].get("max_staleness_days", 45)
ROLLING_IMP_WINDOW      = CFG["pipeline"].get("rolling_imputation_window", 252)
RECALC_TAIL_DAYS        = CFG["pipeline"].get("recalc_tail_days", 5)
CRISIS_STRESS_THRESHOLD = CFG["pipeline"].get("crisis_stress_threshold", 0.70)
MIN_CLUSTER_WEIGHT      = CFG["pipeline"].get("min_cluster_weight", 0.02)
LATE_START_LIMIT        = pd.Timestamp(CFG["pipeline"].get("late_start_limit", "2008-06-01"))
EW_DD_THRESHOLD         = float(CFG["pipeline"].get("ew_drawdown_threshold", 0.05))
FRED_SERIES_MAP         = CFG.get("fred_series", DEFAULT_CONFIG["fred_series"])

SPARSE_BY_DESIGN = {"Yield_Curve_Inversion_Depth"}

DATE_STAMP   = datetime.now().strftime("%Y-%m-%d")
MODELS_PATH  = os.path.join(OUTPUT_DIR, "models_v94.joblib")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history_v94.parquet")
META_PATH    = os.path.join(OUTPUT_DIR, "meta_v94.json")

REGIME_LABELS = {
    0: "RISK_ON_EXPANSION",
    1: "RISK_ON_NORMAL",
    2: "TRANSITION",
    3: "RISK_OFF_CRISE",
    4: "INDETERMINE"
}
N_LABELS = len(REGIME_LABELS)
VALID_REGIMES = ["RISK_ON_EXPANSION", "RISK_ON_NORMAL", "TRANSITION", "RISK_OFF_CRISE"]

log.info(f"🚀 PIPELINE MACRO Quant v{VERSION} | {DATE_STAMP}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. DONNÉES & FEATURE ENGINEERING (FMP + FRED + YFINANCE)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fetch_fmp_stable_eod(symbol: str, start: str = START_DATE) -> pd.Series:
    if not FMP_API_KEY:
        return pd.Series(dtype=float, name=symbol)

    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    chunks = [
        (start, "2014-12-31"),
        ("2015-01-01", datetime.now().strftime("%Y-%m-%d"))
    ]
    all_rows = []
    for dt_from, dt_to in chunks:
        if dt_from > dt_to:
            continue
        try:
            r = requests.get(
                url,
                params={"symbol": symbol, "from": dt_from, "to": dt_to, "apikey": FMP_API_KEY},
                timeout=15
            )
            if r.status_code == 200 and isinstance(r.json(), list):
                all_rows.extend(r.json())
        except Exception as e:
            log.warning(f"   ⚠️ Erreur FMP ({symbol}, {dt_from}->{dt_to}) : {type(e).__name__}")

    if not all_rows:
        return pd.Series(dtype=float, name=symbol)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    col = "adjClose" if "adjClose" in df.columns else "close"
    return pd.to_numeric(df[col], errors="coerce").dropna().rename(symbol)

def fetch_fred_series_api(series_id: str, start: str, retries: int = 3) -> pd.Series:
    if not FRED_API_KEY:
        raise ValueError("❌ Clé API FRED manquante (variable d'environnement FRED_API_KEY).")
    for attempt in range(retries):
        try:
            series = Fred(api_key=FRED_API_KEY).get_series(series_id, observation_start=start)
            series.name = series_id
            return pd.to_numeric(series, errors="coerce")
        except Exception as e:
            log.warning(f"   ⚠️ Échec FRED '{series_id}' (tentative {attempt + 1}/{retries}) : {type(e).__name__}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.error(f"   ❌ Abandon de la série FRED '{series_id}' après {retries} tentatives.")
                return pd.Series(dtype=float, name=series_id)

def download_yf_with_retry(tickers: list, start: str, retries: int = 4) -> pd.DataFrame:
    """Yahoo limite parfois les IP partagées des runners GitHub : on réessaie avec backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, progress=False, auto_adjust=True, threads=False)
            close = raw["Close"]
            if len(close) > 1000 and close["^GSPC"].notna().sum() > 1000:
                return close
            last_err = RuntimeError("Téléchargement Yahoo incomplet")
        except Exception as e:
            last_err = e
        wait = 5 * (2 ** attempt)
        log.warning(f"   ⚠️ yfinance tentative {attempt + 1}/{retries} échouée ({type(last_err).__name__}) — pause {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"❌ yfinance indisponible après {retries} tentatives : {last_err}")

def collect_data(start_date: str = START_DATE) -> pd.DataFrame:
    log.info("📊 Collecte hybride avancée (FMP /stable + yfinance + FRED)...")

    tickers_yf = ["^VIX", "^VIX9D", "^VIX3M", "^SKEW", "^MOVE", "^VVIX", "DX-Y.NYB", "^DJT", "^GSPC", "XLY", "XLP"]
    raw_yf = download_yf_with_retry(tickers_yf, start_date)
    df_yf = raw_yf.ffill().rename(columns={
        "^VIX": "VIX", "^VIX9D": "VIX9D", "^VIX3M": "VIX3M",
        "^SKEW": "SKEW", "^MOVE": "MOVE", "^VVIX": "VVIX",
        "DX-Y.NYB": "DXY", "^DJT": "Transports", "^GSPC": "SP500",
        "XLY": "Cyclicals", "XLP": "Defensives"
    })

    if FMP_API_KEY:
        fmp_map = {"^GSPC": "SP500", "^VIX": "VIX", "HYG": "HYG", "IEF": "IEF", "XLY": "Cyclicals", "XLP": "Defensives"}
        loaded_fmp = []
        for fmp_sym, col_name in fmp_map.items():
            s_fmp = fetch_fmp_stable_eod(fmp_sym, start=start_date)
            if len(s_fmp) > 1000:
                loaded_fmp.append(f"{col_name} ({len(s_fmp)}j)")
                if col_name in df_yf.columns:
                    df_yf[col_name] = s_fmp.reindex(df_yf.index).combine_first(df_yf[col_name])
                else:
                    df_yf[col_name] = s_fmp.reindex(df_yf.index)
        if loaded_fmp:
            log.info(f"   ✅ Séries FMP (/stable) intégrées : {', '.join(loaded_fmp)}")

    fred_series_list = []
    for name, sid in FRED_SERIES_MAP.items():
        s = fetch_fred_series_api(sid, start_date)
        fred_series_list.append(s.rename(name))

    df_fred = pd.concat(fred_series_list, axis=1)
    df_macro = df_yf.join(df_fred, how="outer")

    for col in list(FRED_SERIES_MAP.keys()) + ["VIX9D", "VIX3M", "HYG", "IEF", "Cyclicals", "Defensives"]:
        if col in df_macro.columns:
            df_macro[col] = df_macro[col].ffill(limit=10)

    df_macro = df_macro.loc[df_yf.index].dropna(subset=["VIX", "DXY", "SP500"])
    log.info(f"   ✅ {len(df_macro)} jours collectés | du {df_macro.index[0].date()} au {df_macro.index[-1].date()}")
    return df_macro

def _roc_in_sigma(s: pd.Series, lag: int = 5, win: int = 63) -> pd.Series:
    """Variation sur `lag` jours exprimée en écarts-types rolling du niveau."""
    return s.diff(lag) / s.rolling(win).std()

def engineer_features(df_macro: pd.DataFrame) -> tuple:
    df = df_macro.copy()

    if "NFCI" in df.columns:
        df["NFCI"] = df["NFCI"].shift(5)
        df["NFCI_ROC21"] = df["NFCI"].diff(21)
    if "Fed_BS" in df.columns:
        df["Fed_BS"] = df["Fed_BS"].shift(5)
        df["Fed_BS_ROC21"] = df["Fed_BS"].pct_change(21)

    df["VIX_SMA21"] = df["VIX"].rolling(21).mean()
    df["VIX_STD21"] = df["VIX"].rolling(21).std()
    df["VIX_ZScore"] = (df["VIX"] - df["VIX_SMA21"]) / df["VIX_STD21"]
    df["VIX_PctRank252"] = df["VIX"].rolling(252, min_periods=63).rank(pct=True)
    df["VIX_ROC5"] = df["VIX"].pct_change(5)

    if "VIX9D" in df.columns:
        df["VIX9D_VIX_Ratio"] = df["VIX9D"] / df["VIX"]
    if "VIX3M" in df.columns:
        df["VIX_Term_Spread"] = df["VIX"] - df["VIX3M"]

    df["DXY_ROC5"] = df["DXY"].pct_change(5)
    if "VVIX" in df.columns:
        df["VVIX_ROC5"] = df["VVIX"].pct_change(5)

    df["SP500_SMA200"] = df["SP500"].rolling(200).mean()
    df["SP500_Trend"] = (df["SP500"] / df["SP500_SMA200"]) - 1

    if "Cyclicals" in df.columns and "Defensives" in df.columns:
        df["Cyclical_Defensive_Ratio"] = df["Cyclicals"] / df["Defensives"]
        df["Cyclical_Defensive_ROC21"] = df["Cyclical_Defensive_Ratio"].pct_change(21)

    if "Long_Rate" in df.columns:
        df["Long_Rate_Chg63"] = df["Long_Rate"].diff(63)

    credit_col = "Credit_Spread" if "Credit_Spread" in df.columns else ("HY_Spread" if "HY_Spread" in df.columns else None)
    z_fred, roc_fred = None, None
    if credit_col is not None:
        c_sma = df[credit_col].rolling(63).mean()
        c_std = df[credit_col].rolling(63).std()
        z_fred = (df[credit_col] - c_sma) / c_std
        roc_fred = _roc_in_sigma(df[credit_col])

    if "HYG" in df.columns and "IEF" in df.columns and df["HYG"].notna().sum() > 1000:
        hyg_ief_stress = -np.log(df["HYG"] / df["IEF"])
        z_fmp = (hyg_ief_stress - hyg_ief_stress.rolling(63).mean()) / hyg_ief_stress.rolling(63).std()
        roc_fmp = _roc_in_sigma(hyg_ief_stress)
        df["Credit_ZScore"] = z_fmp.combine_first(z_fred) if z_fred is not None else z_fmp
        df["Credit_ROC5"] = roc_fmp.combine_first(roc_fred) if roc_fred is not None else roc_fmp
    elif z_fred is not None:
        df["Credit_ZScore"] = z_fred
        df["Credit_ROC5"] = roc_fred

    if "Yield_Curve" in df.columns:
        yc_mean = df["Yield_Curve"].rolling(252, min_periods=63).mean()
        yc_std = df["Yield_Curve"].rolling(252, min_periods=63).std()
        df["Yield_Curve_ZScore252"] = (df["Yield_Curve"] - yc_mean) / yc_std
        df["Yield_Curve_Momentum"] = df["Yield_Curve"] - df["Yield_Curve"].shift(63)
        df["Yield_Curve_Inversion_Depth"] = np.minimum(df["Yield_Curve"], 0.0)

    if "Transports" in df.columns:
        df["Transports_ROC21"] = (df["Transports"] / df["SP500"]).pct_change(21)

    features_ideales = [
        "VIX_ZScore", "VIX_PctRank252", "VIX_ROC5", "VIX_Term_Spread", "VIX9D_VIX_Ratio",
        "SKEW", "MOVE", "VVIX", "VVIX_ROC5",
        "Credit_ZScore", "Credit_ROC5",
        "SP500_Trend", "DXY_ROC5", "Transports_ROC21",
        "Cyclical_Defensive_ROC21", "Long_Rate_Chg63",
        "Yield_Curve_ZScore252", "Yield_Curve_Momentum", "Yield_Curve_Inversion_Depth",
        "NFCI", "NFCI_ROC21", "Fed_BS_ROC21"
    ]

    features_cols = [c for c in features_ideales if c in df.columns and df[c].notna().mean() >= COVERAGE_THRESHOLD]

    first_valid = {c: df[c].first_valid_index() for c in features_cols}
    late = {c: d for c, d in first_valid.items() if d is not None and d > LATE_START_LIMIT}
    if late:
        late_str = {c: str(d.date()) for c, d in late.items()}
        log.warning(f"   ⚠️ Features écartées (début après {LATE_START_LIMIT.date()}) : {late_str}")
        features_cols = [c for c in features_cols if c not in late]

    critical_features = ["VIX_ZScore", "VIX_PctRank252", "SP500_Trend", "Credit_ZScore"]
    missing_critical = [f for f in critical_features if f not in features_cols]
    if missing_critical:
        raise RuntimeError(f"❌ ERREUR PIPELINE : Features clés manquantes : {missing_critical}")

    features_core = ["VIX_ZScore", "SP500_Trend", "Credit_ZScore"]
    features_secondary = [c for c in features_cols if c not in features_core]

    df[features_secondary] = df[features_secondary].ffill(limit=30)
    df = df.dropna(subset=features_core).copy()
    df[features_secondary] = df[features_secondary].ffill()

    first_valid_indices = [df[col].first_valid_index() for col in features_cols]
    if None not in first_valid_indices:
        latest_start = max(first_valid_indices)
        if latest_start > df.index[0]:
            culprit = features_cols[first_valid_indices.index(latest_start)]
            log.info(f"   ✂️ Historique tronqué au {latest_start.date()} à cause de '{culprit}'")
            df = df.loc[latest_start:].copy()

    for col in features_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].rolling(window=ROLLING_IMP_WINDOW, min_periods=1).median())
        if df[col].isna().sum() > 0:
            df[col] = df[col].ffill()

    log.info(f"   ✅ Matrice prête : {len(df)} lignes × {len(features_cols)} features | début : {df.index[0].date()}")
    return df, features_cols

def validate_features_for_pca(df: pd.DataFrame, features_cols: list) -> list:
    cleaned_features = []
    for col in features_cols:
        if df[col].var() < 1e-8:
            continue
        if col not in SPARSE_BY_DESIGN:
            real_changes = df[col].diff().abs() > 1e-10
            if not real_changes.any():
                continue
            last_real_change = df.index[real_changes][-1]
            if (df.index[-1] - last_real_change).days > MAX_STALENESS_DAYS:
                continue
        cleaned_features.append(col)
    dropped = sorted(set(features_cols) - set(cleaned_features))
    if dropped:
        log.warning(f"   ⚠️ Features retirées par validate_features_for_pca : {dropped}")
    if len(cleaned_features) < 5:
        raise ValueError(f"❌ Trop de features invalides ou figées ({len(cleaned_features)} restantes).")
    return cleaned_features

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. MODÈLES & SCORE DE STRESS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fit_bgmm(X: np.ndarray) -> BayesianGaussianMixture:
    return BayesianGaussianMixture(
        n_components=N_REGIMES_MAX,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.1,
        n_init=5,
        max_iter=250,
        random_state=42
    ).fit(X)

def compute_cluster_stress_scores(centroids_scaled: np.ndarray, features: list) -> np.ndarray:
    scores = np.zeros(centroids_scaled.shape[0])
    # v9.4 : uniquement des capteurs de stress "coïncidents". La courbe des taux (inversion) et le
    # mouvement du taux 10 ans restent dans les features du modèle mais ne définissent plus la crise :
    # l'inversion est un signal avancé de récession (12-18 mois), pas un stress de marché, et elle
    # étiquetait "CRISE" des périodes haussières (2023-2024).
    weights = {
        "VIX_ZScore": 0.25,
        "VIX_PctRank252": 0.20,
        "Credit_ZScore": 0.25,
        "Cyclical_Defensive_ROC21": -0.10,
        "SP500_Trend": -0.20
    }
    total_w = 0.0
    for feat, w in weights.items():
        if feat in features:
            idx = features.index(feat)
            scores += w * centroids_scaled[:, idx]
            total_w += abs(w)
    if total_w > 0:
        scores /= total_w

    if "SP500_Trend" in features:
        trend_idx = features.index("SP500_Trend")
        trend_values = centroids_scaled[:, trend_idx]
        scores = np.where(trend_values < 0, scores + 0.35 * abs(trend_values), scores)

    return scores

def get_ordered_reference_centroids(bgmm, pca, scaler, features):
    centroids_scaled = pca.inverse_transform(bgmm.means_)
    stress_scores = compute_cluster_stress_scores(centroids_scaled, features)
    order = np.argsort(stress_scores)
    return centroids_scaled[order], scaler

def build_stable_mapping(new_bgmm, new_pca, new_scaler, ref_centroids_scaled, ref_scaler, features):
    new_centroids_raw = new_scaler.inverse_transform(new_pca.inverse_transform(new_bgmm.means_))
    new_centroids_in_ref_scale = ref_scaler.transform(new_centroids_raw)

    active_mask = new_bgmm.weights_ >= MIN_CLUSTER_WEIGHT
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        active_indices = np.arange(N_REGIMES_MAX)

    cost_matrix = np.linalg.norm(
        new_centroids_in_ref_scale[active_indices, np.newaxis, :] - ref_centroids_scaled[np.newaxis, :, :],
        axis=2
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = np.full(N_REGIMES_MAX, N_LABELS - 1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        mapping[active_indices[r]] = c

    for idx in np.where(~active_mask)[0]:
        dists = np.linalg.norm(new_centroids_in_ref_scale[idx] - ref_centroids_scaled, axis=1)
        mapping[idx] = int(np.argmin(dists))

    stress_scores = compute_cluster_stress_scores(new_centroids_in_ref_scale, features)
    for k in range(N_REGIMES_MAX):
        if mapping[k] == 3 and stress_scores[k] < CRISIS_STRESS_THRESHOLD:
            mapping[k] = 2

    return mapping

def fit_temporally_calibrated_classifier(base_estimator, X: pd.DataFrame, y: pd.Series, calib_frac: float = 0.20):
    n = len(X)
    split_idx = int(n * (1.0 - calib_frac))
    train_end = max(100, split_idx - 21)

    if train_end >= split_idx or len(np.unique(y.iloc[:train_end])) < 2 or len(np.unique(y.iloc[split_idx:])) < 2:
        model = CalibratedClassifierCV(clone(base_estimator), method="sigmoid", cv=TimeSeriesSplit(n_splits=3, gap=21))
        model.fit(X, y)
        return model

    est = clone(base_estimator)
    est.fit(X.iloc[:train_end], y.iloc[:train_end])
    try:
        from sklearn.frozen import FrozenEstimator
        cal_model = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid")
    except ImportError:
        cal_model = CalibratedClassifierCV(est, method="sigmoid", cv="prefit")

    cal_model.fit(X.iloc[split_idx:], y.iloc[split_idx:])
    return cal_model

def compute_forward_drawdown(sp: pd.Series, horizon: int = 21) -> pd.Series:
    """Pire repli du S&P sur les `horizon` prochains jours, relatif au cours du jour (NaN sur les derniers jours)."""
    fwd_min = sp.iloc[::-1].rolling(horizon + 1).min().iloc[::-1]
    dd = fwd_min / sp - 1.0
    dd.iloc[-horizon:] = np.nan
    return dd

def generate_oof_ew_predictions(df_results: pd.DataFrame, features_cols: list, horizon: int = 21) -> tuple:
    log.info(f"🧠 Validation Out-of-Fold temporelle (gap={horizon}j) Early Warning...")
    df_ml = df_results.copy()
    # v9.4 : cible = vraie baisse du S&P (>= EW_DD_THRESHOLD dans les `horizon` jours), pas le label BGMM.
    # L'ancienne cible (Is_RiskOff) n'avait aucun positif avant mars 2020 : l'EW n'apprenait rien sur 2018.
    fwd_dd = compute_forward_drawdown(df_ml["SP500"], horizon)
    df_ml["Target_Drawdown"] = np.where(fwd_dd.notna(), (fwd_dd <= -EW_DD_THRESHOLD).astype(float), np.nan)
    valid_mask = df_ml[features_cols + ["Target_Drawdown"]].notna().all(axis=1)
    df_ml = df_ml.loc[valid_mask].copy()
    df_ml["Target_Drawdown"] = df_ml["Target_Drawdown"].astype(int)

    X_ml = df_ml[features_cols]
    y_ml = df_ml["Target_Drawdown"]
    log.info(f"   🎯 Cible EW : baisse >= {EW_DD_THRESHOLD:.0%} à {horizon}j | positifs : {y_ml.mean():.1%} ({int(y_ml.sum())}/{len(y_ml)})")

    # v9.4 : modèle peu profond et régularisé (≈170 fenêtres de 21j réellement indépendantes sur 14 ans),
    # SANS calibration : le wrapper sigmoïde sur les 20 % de fin de pli dégradait l'AUC OOF (0.43 vs ~0.55
    # en test) et produisait des sorties quasi constantes. Le seuil WFO ne dépend que du classement ;
    # avec class_weight="balanced", le score est centré autour de 0.3-0.5 : la plage de seuils 0.35-0.75 reste adaptée.
    # Ce score est un indice de risque, pas une probabilité calibrée.
    base_ew = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, max_iter=80, min_samples_leaf=60,
        random_state=42, class_weight="balanced"
    )

    tscv = TimeSeriesSplit(n_splits=5, gap=horizon)
    oof_preds = pd.Series(np.nan, index=df_results.index, dtype=float)

    for train_idx, test_idx in tscv.split(X_ml):
        X_tr, y_tr = X_ml.iloc[train_idx], y_ml.iloc[train_idx]
        X_te = X_ml.iloc[test_idx]
        test_dates = X_ml.index[test_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        fold_model = clone(base_ew).fit(X_tr, y_tr)
        oof_preds.loc[test_dates] = fold_model.predict_proba(X_te)[:, 1]

    final_ew_model = clone(base_ew).fit(X_ml, y_ml)
    tail_dates = df_results.index[df_results.index > X_ml.index[-1]]
    if len(tail_dates) > 0:
        oof_preds.loc[tail_dates] = final_ew_model.predict_proba(df_results.loc[tail_dates, features_cols])[:, 1]

    return final_ew_model, oof_preds

def evaluate_strategy_performance(df_slice: pd.DataFrame, threshold: float) -> float:
    # Signal décidé à la clôture de t-1, appliqué au rendement de t (pas de look-ahead)
    df_sim = df_slice.dropna(subset=["EW_Proba_RiskOff"]).copy()
    if len(df_sim) < 100:
        return -999.0

    signal_cash = (df_sim["EW_Proba_RiskOff"] > threshold).shift(1, fill_value=False)
    cash_share = signal_cash.mean()
    if cash_share > 0.40 or cash_share < 0.01:
        return -999.0

    sp_ret = df_sim["SP500"].pct_change().fillna(0.0)
    strat_ret = sp_ret.where(~signal_cash, 0.0)

    cum = (1 + strat_ret).cumprod()
    max_dd = abs((cum / cum.cummax() - 1).min())
    ann_return = strat_ret.mean() * 252
    return float(ann_return / (max_dd + 1e-4))

def walk_forward_optimize_ew(df_results: pd.DataFrame, train_window: int = 756, test_window: int = 126) -> pd.DataFrame:
    log.info("🔄 Optimisation Walk-Forward (WFO) du seuil Early Warning...")
    n_days = len(df_results)
    if n_days < (train_window + test_window):
        df_results["WFO_Threshold"] = 0.50
        df_results["WFO_Signal_RiskOff"] = df_results["EW_Proba_RiskOff"] > 0.50
        return df_results

    wfo_signals = pd.Series(False, index=df_results.index)
    threshold_history = pd.Series(np.nan, index=df_results.index)

    start_idx = train_window
    while start_idx < n_days:
        end_idx = min(start_idx + test_window, n_days)
        df_train = df_results.iloc[start_idx - train_window : start_idx]

        def objective(trial):
            th = trial.suggest_float("threshold", 0.35, 0.75)
            return evaluate_strategy_performance(df_train, th)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        if study.best_value <= -900:      # aucun seuil exploitable sur cette fenêtre (EW NaN, trop/peu de cash)
            best_threshold = 0.50
        else:
            best_threshold = study.best_params["threshold"]

        df_test = df_results.iloc[start_idx:end_idx]
        test_dates = df_test.index

        wfo_signals.loc[test_dates] = df_test["EW_Proba_RiskOff"] > best_threshold
        threshold_history.loc[test_dates] = best_threshold
        start_idx = end_idx

    df_results["WFO_Threshold"] = threshold_history.ffill().fillna(0.50)
    df_results["WFO_Signal_RiskOff"] = wfo_signals
    log.info("   ✅ WFO EW finalisé avec succès.")
    return df_results

def train_regime_forecaster(df_results: pd.DataFrame, features_cols: list, horizon: int = 21):
    log.info(f"🔮 Entraînement du Regime Forecaster multiclasse (Horizon {horizon}j)...")
    df_ml = df_results.copy()
    df_ml["Target_Regime"] = df_ml["Regime_IA"].shift(-horizon)
    df_ml = df_ml.dropna(subset=features_cols + ["Target_Regime"])
    df_ml = df_ml[df_ml["Target_Regime"].isin(VALID_REGIMES)]

    X_ml, y_ml = df_ml[features_cols], df_ml["Target_Regime"]
    base_forecaster = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42)
    forecaster = CalibratedClassifierCV(base_forecaster, method="sigmoid", cv=TimeSeriesSplit(n_splits=3, gap=horizon))
    forecaster.fit(X_ml, y_ml)
    return forecaster

def evaluate_market_reality(df_results: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    df_eval = df_results.copy()
    sp_ret = df_eval["SP500"].pct_change()
    df_eval["Fwd_Realized_Vol_21d"] = sp_ret.rolling(horizon).std().shift(-horizon) * np.sqrt(252) * 100
    df_eval["Fwd_Return_21d"] = (df_eval["SP500"].shift(-horizon) / df_eval["SP500"] - 1.0) * 100

    fwd_max_dd = np.full(len(df_eval), np.nan)
    sp_vals = df_eval["SP500"].values
    for idx in range(len(sp_vals) - horizon):
        window = sp_vals[idx: idx + horizon + 1]
        fwd_max_dd[idx] = ((np.min(window) / sp_vals[idx]) - 1.0) * 100
    df_eval["Fwd_MaxDD_21d"] = fwd_max_dd

    summary = (
        df_eval[df_eval["Regime_IA"].isin(VALID_REGIMES)]
        .groupby("Regime_IA")[["Fwd_Return_21d", "Fwd_Realized_Vol_21d", "Fwd_MaxDD_21d"]]
        .agg(["count", "mean", "median"])
    )
    return summary

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. PIPELINE COMPLÈTE & SAUVEGARDE/CHARGEMENT                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_full_pipeline(df_model: pd.DataFrame = None, features_cols: list = None):
    log.info(f"🚀 EXÉCUTION PIPELINE COMPLÈTE v{VERSION}")
    if df_model is None or features_cols is None:
        df_macro = collect_data()
        df_model, features_cols = engineer_features(df_macro)
        features_cols = validate_features_for_pca(df_model, features_cols)

    X_all = df_model[features_cols].values

    scaler_ref = StandardScaler()
    pca_ref = PCA(n_components=PCA_VARIANCE_TARGET, random_state=42)
    X_init_scaled = scaler_ref.fit_transform(X_all[:MIN_TRAIN_DAYS])
    X_init_pca = pca_ref.fit_transform(X_init_scaled)

    bgmm_ref = fit_bgmm(X_init_pca)
    ref_centroids_scaled, ref_scaler = get_ordered_reference_centroids(bgmm_ref, pca_ref, scaler_ref, features_cols)
    current_mapping = build_stable_mapping(bgmm_ref, pca_ref, scaler_ref, ref_centroids_scaled, ref_scaler, features_cols)
    current_bgmm, current_scaler, current_pca, last_refit_idx = bgmm_ref, scaler_ref, pca_ref, MIN_TRAIN_DAYS

    log.info("🔄 Inférence causale rolling BGMM...")
    total_days = len(df_model)
    regimes_predits, riskoff_predits, posterior_max_list = [], [], []
    proba_expansion_list, proba_normal_list, proba_transition_list, proba_crise_list = [], [], [], []

    pbar = tqdm(total=total_days - MIN_TRAIN_DAYS, desc="Rolling BGMM", unit="j", ncols=100,
                disable=not sys.stderr.isatty())
    i = MIN_TRAIN_DAYS
    while i < total_days:
        if (i - last_refit_idx) >= REFIT_FREQ:
            window_start = max(0, i - WINDOW_SIZE)
            new_scaler = StandardScaler()
            new_pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=42)
            X_pca = new_pca.fit_transform(new_scaler.fit_transform(X_all[window_start:i]))
            new_bgmm = fit_bgmm(X_pca)
            current_mapping = build_stable_mapping(
                new_bgmm, new_pca, new_scaler, ref_centroids_scaled, ref_scaler, features_cols
            )
            current_bgmm, current_scaler, current_pca, last_refit_idx = new_bgmm, new_scaler, new_pca, i

        end_batch = min(i + REFIT_FREQ - (i - last_refit_idx), total_days)
        batch_size = end_batch - i
        X_batch_pca = current_pca.transform(current_scaler.transform(X_all[i:end_batch]))
        proba_vecs = current_bgmm.predict_proba(X_batch_pca)

        for j in range(batch_size):
            mapped_probas = np.zeros(N_REGIMES_MAX)
            for raw_clust_idx in range(N_REGIMES_MAX):
                mapped_rank = current_mapping[raw_clust_idx]
                if mapped_rank < N_REGIMES_MAX:
                    mapped_probas[mapped_rank] += proba_vecs[j, raw_clust_idx]

            stable_rank = int(np.argmax(mapped_probas))
            regime_name = REGIME_LABELS.get(min(stable_rank, N_LABELS - 1), "INDETERMINE")

            regimes_predits.append(regime_name)
            riskoff_predits.append(regime_name == "RISK_OFF_CRISE")
            posterior_max_list.append(float(mapped_probas.max()))
            proba_expansion_list.append(float(mapped_probas[0]))
            proba_normal_list.append(float(mapped_probas[1]))
            proba_transition_list.append(float(mapped_probas[2]))
            proba_crise_list.append(float(mapped_probas[3]))

        pbar.update(batch_size)
        i = end_batch
    pbar.close()

    df_results = df_model.iloc[MIN_TRAIN_DAYS:].copy()
    df_results["Regime_IA"] = regimes_predits
    df_results["Is_RiskOff"] = riskoff_predits
    df_results["Posterior_BGMM"] = posterior_max_list
    df_results["Proba_Expansion"] = proba_expansion_list
    df_results["Proba_Normal"] = proba_normal_list
    df_results["Proba_Transition"] = proba_transition_list
    df_results["Proba_Crise"] = proba_crise_list

    ew_model, oof_ew_series = generate_oof_ew_predictions(df_results, features_cols, horizon=21)
    df_results["EW_Proba_RiskOff"] = oof_ew_series

    df_results = walk_forward_optimize_ew(df_results)
    forecaster_21d = train_regime_forecaster(df_results, features_cols, horizon=21)

    last_refit_date = str(df_model.index[last_refit_idx - 1].date())
    last_full_train_date = str(df_results.index[-1].date())

    save_all(
        df_results, current_bgmm, current_scaler, current_pca, current_mapping,
        ref_centroids_scaled, ref_scaler, ew_model, forecaster_21d,
        last_refit_date, last_full_train_date, features_cols
    )
    return {
        "df_results": df_results,
        "ew_model": ew_model,
        "forecaster": forecaster_21d,
        "features_cols": features_cols
    }

def save_all(df_results, bgmm, scaler, pca, mapping, ref_centroids_scaled, ref_scaler,
             ew_model, forecaster, last_refit_date, last_full_train_date, features_cols):
    joblib.dump({
        "bgmm": bgmm,
        "scaler": scaler,
        "pca": pca,
        "mapping": mapping,
        "ref_centroids_scaled": ref_centroids_scaled,
        "ref_scaler": ref_scaler,
        "ew_model": ew_model,
        "forecaster_21d": forecaster,
        "last_refit_date": last_refit_date,
        "last_full_train_date": last_full_train_date
    }, MODELS_PATH, compress=3)
    df_results.to_parquet(HISTORY_PATH)
    with open(META_PATH, "w") as f:
        json.dump({
            "last_date": str(df_results.index[-1].date()),
            "last_full_train_date": last_full_train_date,
            "last_refit_date": last_refit_date,
            "features_cols": features_cols,
            "version": VERSION,
            "saved_at": datetime.now().isoformat()
        }, f, indent=2)
    log.info(f"   💾 Artefacts sauvegardés (v{VERSION})")

def load_all():
    if not all(os.path.exists(p) for p in [MODELS_PATH, HISTORY_PATH, META_PATH]):
        return None
    try:
        models = joblib.load(MODELS_PATH)
        df_results = pd.read_parquet(HISTORY_PATH)
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        if meta.get("version") != VERSION:
            log.info("   ℹ️ Version antérieure détectée -> Recalcul complet requis.")
            return None
        log.info(f"   ✅ Sauvegarde v{VERSION} chargée (dernier point : {meta['last_date']})")
        return {
            "df_results": df_results,
            "bgmm": models["bgmm"],
            "scaler": models["scaler"],
            "pca": models["pca"],
            "mapping": models["mapping"],
            "ref_centroids_scaled": models["ref_centroids_scaled"],
            "ref_scaler": models["ref_scaler"],
            "ew_model": models["ew_model"],
            "forecaster": models["forecaster_21d"],
            "last_refit_date": models["last_refit_date"],
            "last_full_train_date": models["last_full_train_date"],
            "features_cols": meta["features_cols"],
            "last_date": meta["last_date"]
        }
    except Exception as e:
        # Typiquement : cache produit avec une autre version de scikit-learn -> recalcul complet
        log.error(f"   ❌ Erreur chargement ({type(e).__name__}) -> recalcul complet")
        return None

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. PIPELINE INCRÉMENTALE & ORCHESTRATEUR                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_incremental_logic(saved, df_model: pd.DataFrame, features_cols: list):
    df_prev = saved["df_results"]
    current_bgmm = saved["bgmm"]
    current_scaler = saved["scaler"]
    current_pca = saved["pca"]
    current_mapping = saved["mapping"]
    ref_centroids_scaled = saved["ref_centroids_scaled"]
    ref_scaler = saved["ref_scaler"]
    ew_model = saved["ew_model"]
    forecaster = saved["forecaster"]
    last_refit_date = pd.Timestamp(saved["last_refit_date"])
    last_full_train_date = saved["last_full_train_date"]

    cutoff_idx = max(0, len(df_prev) - RECALC_TAIL_DAYS)
    recalc_start_date = df_prev.index[cutoff_idx]

    df_slice = df_model.loc[df_model.index >= recalc_start_date].copy()
    new_dates = df_slice.index

    days_since_bgmm_refit = len(df_model.loc[df_model.index > last_refit_date])
    if days_since_bgmm_refit >= REFIT_FREQ:
        hist_before_slice = df_model.loc[df_model.index < recalc_start_date, features_cols].values
        X_window = hist_before_slice[-WINDOW_SIZE:]
        current_scaler = StandardScaler()
        current_pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=42)
        X_pca = current_pca.fit_transform(current_scaler.fit_transform(X_window))
        current_bgmm = fit_bgmm(X_pca)
        current_mapping = build_stable_mapping(
            current_bgmm, current_pca, current_scaler, ref_centroids_scaled, ref_scaler, features_cols
        )
        last_refit_date = df_model.index[df_model.index < recalc_start_date][-1]

    X_new = df_slice[features_cols].values
    X_new_pca = current_pca.transform(current_scaler.transform(X_new))
    proba_vecs = current_bgmm.predict_proba(X_new_pca)

    new_regimes, new_riskoff, new_posterior = [], [], []
    proba_exp, proba_norm, proba_trans, proba_cris = [], [], [], []

    for j in range(len(X_new)):
        mapped_probas = np.zeros(N_REGIMES_MAX)
        for raw_clust_idx in range(N_REGIMES_MAX):
            mapped_rank = current_mapping[raw_clust_idx]
            if mapped_rank < N_REGIMES_MAX:
                mapped_probas[mapped_rank] += proba_vecs[j, raw_clust_idx]

        stable_rank = int(np.argmax(mapped_probas))
        regime_name = REGIME_LABELS.get(min(stable_rank, N_LABELS - 1), "INDETERMINE")

        new_regimes.append(regime_name)
        new_riskoff.append(regime_name == "RISK_OFF_CRISE")
        new_posterior.append(float(mapped_probas.max()))
        proba_exp.append(float(mapped_probas[0]))
        proba_norm.append(float(mapped_probas[1]))
        proba_trans.append(float(mapped_probas[2]))
        proba_cris.append(float(mapped_probas[3]))

    new_rows = df_model.loc[new_dates].copy()
    new_rows["Regime_IA"] = new_regimes
    new_rows["Is_RiskOff"] = new_riskoff
    new_rows["Posterior_BGMM"] = new_posterior
    new_rows["Proba_Expansion"] = proba_exp
    new_rows["Proba_Normal"] = proba_norm
    new_rows["Proba_Transition"] = proba_trans
    new_rows["Proba_Crise"] = proba_cris
    new_rows["EW_Proba_RiskOff"] = ew_model.predict_proba(new_rows[features_cols])[:, 1]

    last_known_threshold = df_prev["WFO_Threshold"].iloc[-1] if "WFO_Threshold" in df_prev.columns else 0.50
    new_rows["WFO_Threshold"] = last_known_threshold
    new_rows["WFO_Signal_RiskOff"] = new_rows["EW_Proba_RiskOff"] > last_known_threshold

    df_results = pd.concat([df_prev.loc[df_prev.index < recalc_start_date], new_rows], axis=0).sort_index()

    save_all(
        df_results, current_bgmm, current_scaler, current_pca, current_mapping,
        ref_centroids_scaled, ref_scaler, ew_model, forecaster,
        str(last_refit_date.date()), last_full_train_date, features_cols
    )
    return {
        "df_results": df_results,
        "ew_model": ew_model,
        "forecaster": forecaster,
        "features_cols": features_cols
    }

def automated_run() -> dict:
    log.info(f"{'═'*65}\n  🤖 Oracle Macro-Micro v{VERSION} — ORCHESTRATEUR\n{'═'*65}")
    df_macro = collect_data()
    df_model, features_cols_new = engineer_features(df_macro)
    features_cols_new = validate_features_for_pca(df_model, features_cols_new)

    saved = load_all()
    if saved is None:
        log.info(f"🚀 Décision : Aucune sauvegarde v{VERSION} exploitable -> PIPELINE COMPLÈTE")
        return execute_and_finalize("FULL", df_model=df_model, features_cols=features_cols_new)

    if features_cols_new != saved["features_cols"]:
        log.info("🛠️ Décision : Set de features modifié -> PIPELINE COMPLÈTE")
        return execute_and_finalize("FULL", df_model=df_model, features_cols=features_cols_new)

    last_train_dt = pd.Timestamp(saved["last_full_train_date"])
    days_since_full_train = len(df_model.loc[df_model.index > last_train_dt])

    if days_since_full_train >= FULL_REFIT_EVERY_N_DAYS:
        log.info(f"⏳ Décision : {days_since_full_train}j depuis le dernier full fit -> PIPELINE COMPLÈTE")
        return execute_and_finalize("FULL", df_model=df_model, features_cols=features_cols_new)

    log.info(f"⚡ Décision : Mise à jour incrémentale (recalcul des {RECALC_TAIL_DAYS} derniers jours)")
    return execute_and_finalize("INCREMENTAL", saved=saved, df_model=df_model, features_cols=features_cols_new)

def execute_and_finalize(mode: str, saved=None, df_model=None, features_cols=None) -> dict:
    if mode == "FULL":
        result = run_full_pipeline(df_model=df_model, features_cols=features_cols)
    elif mode == "INCREMENTAL":
        result = run_incremental_logic(saved, df_model, features_cols)
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    df_results = result["df_results"]
    ew_model = result["ew_model"]
    forecaster = result["forecaster"]
    features_cols = result["features_cols"]

    current_regime = df_results["Regime_IA"].iloc[-1]
    previous_regime = df_results["Regime_IA"].iloc[-2] if len(df_results) > 1 else current_regime
    previous_signal = bool(df_results["WFO_Signal_RiskOff"].iloc[-2]) if len(df_results) > 1 else False

    latest_features = df_results[features_cols].iloc[-1:]
    if latest_features.isna().any().any():
        nan_cols = latest_features.columns[latest_features.isna().iloc[0]].tolist()
        raise RuntimeError(f"❌ ERREUR CRITIQUE : NaN détectés dans latest_features : {nan_cols}")

    proba_riskoff_future = float(ew_model.predict_proba(latest_features)[0, 1])
    current_wfo_threshold = float(df_results["WFO_Threshold"].iloc[-1])
    is_wfo_risk_off = bool(df_results["WFO_Signal_RiskOff"].iloc[-1])

    probas_futur = forecaster.predict_proba(latest_features)[0]
    forecast_dict = {c: float(p) for c, p in zip(forecaster.classes_, probas_futur)}

    market_audit = evaluate_market_reality(df_results, horizon=21)

    crise_by_year = (df_results["Regime_IA"] == "RISK_OFF_CRISE").groupby(df_results.index.year).sum()
    log.info("   🔎 Jours classés CRISE par année : " + ", ".join(f"{y}:{int(n)}" for y, n in crise_by_year.items() if n > 0))

    # ── DASHBOARD PLOTLY ──────────────────────────────────────────────────────
    df_plot = df_results.copy()
    df_plot["Group"] = (df_plot["Regime_IA"] != df_plot["Regime_IA"].shift(1)).cumsum()
    idx_name = df_plot.index.name or "index"

    regime_blocks = (
        df_plot.reset_index()
        .groupby(["Group", "Regime_IA"])[idx_name]
        .agg(start="first", end="last")
        .reset_index()
    )
    last_group_id = regime_blocks["Group"].max()
    regime_blocks = regime_blocks[
        ((regime_blocks["end"] - regime_blocks["start"]) >= pd.Timedelta(days=3)) |
        (regime_blocks["Group"] == last_group_id)
    ].copy()

    REGIME_COLORS = {
        "RISK_ON_EXPANSION": "rgba(0,230,118,0.25)",
        "RISK_ON_NORMAL": "rgba(0,150,255,0.22)",
        "TRANSITION": "rgba(255,214,0,0.20)",
        "RISK_OFF_CRISE": "rgba(220,30,30,0.38)"
    }

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
        subplot_titles=(
            f"S&P 500 & Régimes Causaux (v{VERSION})",
            "Posterior Max BGMM",
            "Early Warning OOF & Seuil WFO Dynamique",
            "Ratio Cycliques / Défensifs (XLY/XLP)",
            "Credit Stress (Z-Score FMP + FRED)"
        )
    )

    custom_probs = np.stack([
        df_plot["Proba_Expansion"] * 100,
        df_plot["Proba_Normal"] * 100,
        df_plot["Proba_Transition"] * 100,
        df_plot["Proba_Crise"] * 100
    ], axis=-1)

    fig.add_trace(
        go.Scatter(
            x=df_plot.index, y=df_plot["SP500"], name="SP500", customdata=custom_probs,
            line=dict(color="white", width=1.5),
            hovertemplate=(
                "<b>S&P 500</b>: %{y:.1f}<br>"
                "P(Expansion): %{customdata[0]:.1f}% | P(Normal): %{customdata[1]:.1f}%<br>"
                "P(Transition): %{customdata[2]:.1f}% | P(Crise): %{customdata[3]:.1f}%<extra></extra>"
            )
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot.index, y=df_plot["Posterior_BGMM"] * 100, name="Posterior Max",
            fill="tozeroy", fillcolor="rgba(100,180,255,0.12)", line=dict(color="#64b4ff", width=1)
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["EW_Proba_RiskOff"] * 100, name="EW Risk-Off", line=dict(color="#ff4444", width=1.5)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["WFO_Threshold"] * 100, name="WFO Threshold", line=dict(color="#00e676", width=1.5, dash="dash")),
        row=3, col=1
    )
    if "Cyclical_Defensive_Ratio" in df_plot.columns:
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["Cyclical_Defensive_Ratio"], name="XLY/XLP Ratio", line=dict(color="#29b6f6", width=1.5)),
            row=4, col=1
        )
    if "Credit_ZScore" in df_plot.columns:
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot["Credit_ZScore"], name="Credit Z-Score", line=dict(color="#ffa726", width=1.5)),
            row=5, col=1
        )
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.6)", row=5, col=1)

    for _, row_data in regime_blocks.iterrows():
        if row_data["Regime_IA"] in REGIME_COLORS:
            fig.add_vrect(
                x0=row_data["start"], x1=row_data["end"],
                fillcolor=REGIME_COLORS[row_data["Regime_IA"]],
                opacity=1, layer="below", line_width=0, row=1, col=1
            )

    forecast_subtitle = " | ".join([f"<b>{c.split('_')[-1]}</b>: {p:.1%}" for c, p in forecast_dict.items()])
    title_text = (
        f"<b>Oracle Macro-Micro v{VERSION}</b> | {DATE_STAMP}<br>"
        f"<span style='font-size:14px; color:#ffffff'>"
        f"📍 Régime actuel : <b>{current_regime}</b> | "
        f"🚨 Early Warning (score) : <b>{proba_riskoff_future:.1%}</b> (Seuil WFO : <b>{current_wfo_threshold:.2f}</b> → Signal : <b>{'RISK-OFF' if is_wfo_risk_off else 'NORMAL'}</b>)"
        f"</span><br>"
        f"<span style='font-size:12px; color:#a0a0a0'>🔮 Prévisions Régimes à 21j : {forecast_subtitle}</span>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        margin=dict(t=140),
        template="plotly_dark", paper_bgcolor="#080c10", plot_bgcolor="#0d1117",
        font=dict(color="white"), height=1150, hovermode="x unified"
    )
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")

    html_path = os.path.join(OUTPUT_DIR, f"Dashboard_Macro_v{VERSION.replace('.', '')}_{DATE_STAMP}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(to_html(fig, full_html=True, include_plotlyjs="cdn"))

    log.info(f"✅ PIPELINE v{VERSION} TERMINÉE (Mode : {mode}) | Régime : {current_regime} | "
             f"EW : {proba_riskoff_future:.1%} (seuil {current_wfo_threshold:.2f}) | Dashboard : {html_path}")

    return {
        "mode": mode,
        "last_date": str(df_results.index[-1].date()),
        "history_start": str(df_results.index[0].date()),
        "n_features": len(features_cols),
        "current_regime": current_regime,
        "previous_regime": previous_regime,
        "regime_changed": current_regime != previous_regime,
        "ew_proba": proba_riskoff_future,
        "wfo_threshold": current_wfo_threshold,
        "is_risk_off": is_wfo_risk_off,
        "signal_changed": is_wfo_risk_off != previous_signal,
        "forecast": forecast_dict,
        "market_audit": market_audit,
        "html_path": html_path,
    }

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. NOTIFICATION EMAIL                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def send_email(subject: str, text_body: str, html_body: str, attachments: list = None) -> bool:
    """Envoie un email via SMTP. Retourne False si l'email n'est pas configuré ; lève une exception si l'envoi échoue."""
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    mail_to = os.getenv("MAIL_TO")
    if not all([host, user, password, mail_to]):
        log.warning("📧 Email non configuré (SMTP_HOST / SMTP_USER / SMTP_PASSWORD / MAIL_TO) — envoi ignoré.")
        return False

    port = int(os.getenv("SMTP_PORT") or "465")
    mail_from = os.getenv("MAIL_FROM") or user
    recipients = [a.strip() for a in mail_to.replace(";", ",").split(",") if a.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    for path in attachments or []:
        if os.path.exists(path):
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype="text", subtype="html", filename=os.path.basename(path))

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as server:
            server.login(user, password)
            server.send_message(msg, to_addrs=recipients)
    else:
        with smtplib.SMTP(host, port, timeout=60) as server:
            server.starttls(context=context)
            server.login(user, password)
            server.send_message(msg, to_addrs=recipients)

    log.info(f"📧 Email envoyé à {', '.join(recipients)}")
    return True

def _audit_table_html(audit: pd.DataFrame) -> str:
    th = "style='padding:4px 8px;border-bottom:1px solid #999;text-align:right'"
    td = "style='padding:4px 8px;border-bottom:1px solid #ddd;text-align:right'"
    rows = ""
    for regime in audit.index:
        r = audit.loc[regime]
        rows += (
            f"<tr><td style='padding:4px 8px;border-bottom:1px solid #ddd'>{html.escape(str(regime))}</td>"
            f"<td {td}>{int(r[('Fwd_Return_21d', 'count')])}</td>"
            f"<td {td}>{r[('Fwd_Return_21d', 'mean')]:.2f}%</td>"
            f"<td {td}>{r[('Fwd_Return_21d', 'median')]:.2f}%</td>"
            f"<td {td}>{r[('Fwd_Realized_Vol_21d', 'mean')]:.1f}</td>"
            f"<td {td}>{r[('Fwd_MaxDD_21d', 'mean')]:.2f}%</td></tr>"
        )
    return (
        "<table style='border-collapse:collapse;font-size:13px'>"
        f"<tr><th style='padding:4px 8px;border-bottom:1px solid #999;text-align:left'>Régime</th>"
        f"<th {th}>Jours</th><th {th}>Rdt 21j moy.</th><th {th}>Rdt 21j méd.</th>"
        f"<th {th}>Vol réal. moy.</th><th {th}>MaxDD 21j moy.</th></tr>{rows}</table>"
    )

def build_email(summary: dict) -> tuple:
    s = summary
    signal_txt = "RISK-OFF" if s["is_risk_off"] else "NORMAL"
    icon = "🔴" if s["is_risk_off"] else "🟢"
    alert_flags = []
    if s["regime_changed"]:
        alert_flags.append(f"changement de régime ({s['previous_regime']} → {s['current_regime']})")
    if s["signal_changed"]:
        alert_flags.append(f"signal WFO passé à {signal_txt}")

    subject = (
        f"{icon}{'⚠️' if alert_flags else ''} [Oracle Macro] {s['last_date']} — "
        f"{s['current_regime']} | EW {s['ew_proba']:.0%} → {signal_txt}"
    )

    forecast_sorted = sorted(s["forecast"].items(), key=lambda kv: kv[1], reverse=True)
    forecast_html = "".join(
        f"<tr><td style='padding:3px 8px'>{html.escape(c)}</td>"
        f"<td style='padding:3px 8px;text-align:right'><b>{p:.1%}</b></td></tr>"
        for c, p in forecast_sorted
    )

    stale_days = (pd.Timestamp(DATE_STAMP) - pd.Timestamp(s["last_date"])).days
    stale_html = ""
    stale_txt = ""
    if stale_days > 4:
        stale_html = (f"<p style='color:#b00'><b>⚠️ Dernière donnée vieille de {stale_days} jours : "
                      f"vérifier les sources.</b></p>")
        stale_txt = f"\n!! Dernière donnée vieille de {stale_days} jours : vérifier les sources.\n"

    alerts_html = ""
    if alert_flags:
        alerts_html = "<p style='background:#fff3cd;padding:8px;border-radius:4px'>⚠️ " + " ; ".join(alert_flags) + "</p>"

    html_body = f"""
<div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#222;max-width:720px">
  <h2 style="margin-bottom:4px">{icon} Oracle Macro-Micro v{VERSION}</h2>
  <div style="color:#666;margin-bottom:12px">Dernière donnée : {s['last_date']} · Mode : {s['mode']} · {s['n_features']} features · historique depuis {s['history_start']}</div>
  {alerts_html}{stale_html}
  <table style="font-size:15px;border-collapse:collapse">
    <tr><td style="padding:3px 8px">📍 Régime actuel</td><td style="padding:3px 8px"><b>{html.escape(s['current_regime'])}</b></td></tr>
    <tr><td style="padding:3px 8px">🚨 Early Warning (risk-off à 21j)</td><td style="padding:3px 8px"><b>{s['ew_proba']:.1%}</b></td></tr>
    <tr><td style="padding:3px 8px">🎯 Seuil WFO</td><td style="padding:3px 8px">{s['wfo_threshold']:.2f}</td></tr>
    <tr><td style="padding:3px 8px">Signal</td><td style="padding:3px 8px"><b>{signal_txt}</b></td></tr>
  </table>
  <h3 style="margin-bottom:4px">🔮 Prévision des régimes à 21 jours</h3>
  <table style="font-size:14px;border-collapse:collapse">{forecast_html}</table>
  <h3 style="margin-bottom:4px">📉 Audit : comportement réel du S&amp;P 500 à t+21j</h3>
  {_audit_table_html(s['market_audit'])}
  <p style="color:#888;font-size:12px;margin-top:16px">Dashboard interactif en pièce jointe (HTML). Information à titre indicatif, ne constitue pas un conseil en investissement.</p>
</div>"""

    text_body = (
        f"Oracle Macro-Micro v{VERSION} — données au {s['last_date']} (mode {s['mode']})\n"
        + (("ALERTE : " + " ; ".join(alert_flags) + "\n") if alert_flags else "")
        + stale_txt
        + f"\nRégime actuel : {s['current_regime']}\n"
        f"Early Warning : {s['ew_proba']:.1%} (seuil WFO {s['wfo_threshold']:.2f}) -> {signal_txt}\n\n"
        "Prévision des régimes à 21 jours :\n"
        + "\n".join(f"  {c}: {p:.1%}" for c, p in forecast_sorted)
        + "\n\nAudit S&P 500 à t+21j :\n" + s["market_audit"].round(2).to_string()
        + "\n\nDashboard HTML en pièce jointe.\n"
    )
    return subject, text_body, html_body

def should_send(summary: dict) -> bool:
    mode = (os.getenv("MAIL_MODE") or "always").lower()
    if mode == "alerts":
        return summary["is_risk_off"] or summary["regime_changed"] or summary["signal_changed"]
    return True

def write_github_summary(summary: dict) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"### Oracle Macro v{VERSION} — {summary['last_date']}\n\n"
            f"- Régime : **{summary['current_regime']}**\n"
            f"- Early Warning : **{summary['ew_proba']:.1%}** (seuil {summary['wfo_threshold']:.2f}) → "
            f"**{'RISK-OFF' if summary['is_risk_off'] else 'NORMAL'}**\n"
            f"- Mode : {summary['mode']}\n"
        )

def main() -> None:
    try:
        summary = automated_run()
        write_github_summary(summary)
        if should_send(summary):
            subject, text_body, html_body = build_email(summary)
            send_email(subject, text_body, html_body, attachments=[summary["html_path"]])
        else:
            log.info("📧 MAIL_MODE=alerts : aucune alerte aujourd'hui, pas d'email.")
    except Exception:
        tb = traceback.format_exc()
        log.error(tb)
        try:
            send_email(
                f"❌ [Oracle Macro] Échec du pipeline — {DATE_STAMP}",
                tb,
                f"<h3>❌ Échec du pipeline Oracle Macro</h3><pre>{html.escape(tb)}</pre>"
            )
        except Exception as mail_err:
            log.error(f"Envoi de l'email d'erreur impossible : {type(mail_err).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    main()
