# =============================================================================================
# Oracle Macro-Micro v9.2 — Version GitHub Headless & Notification Email
# =============================================================================================

import os
import time
import warnings
import logging
import json
import yaml
import requests
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.base import clone
from scipy.optimize import linear_sum_assignment

import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
from tqdm import tqdm
import joblib

warnings.filterwarnings("default")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", handlers=[logging.StreamHandler()])
log = logging.getLogger("MacroPipeline")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  0. CONFIGURATION & SECRETS ENVIRONNEMENT                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
FRED_API_KEY = os.getenv("FRED_API_KEY")
FMP_API_KEY  = os.getenv("FMP_API_KEY")
EMAIL_USER   = os.getenv("EMAIL_USER")
EMAIL_PASS   = os.getenv("EMAIL_PASSWORD")

if not FRED_API_KEY:
    raise ValueError("❌ Clé FRED_API_KEY manquante dans les variables d'environnement.")

DRIVE_PATH = "./Risk_ON_OFF_Output"
os.makedirs(DRIVE_PATH, exist_ok=True)
CONFIG_PATH = os.path.join(DRIVE_PATH, "config.yaml")

DEFAULT_CONFIG = {
    "pipeline": {
        "start_date": "2005-01-01", "n_regimes_max": 4, "min_train_days": 1260,
        "window_size": 1260, "refit_freq": 21, "full_refit_every_n_days": 63,
        "coverage_threshold": 0.60, "pca_variance_target": 0.95,
        "max_staleness_days": 45, "rolling_imputation_window": 252,
        "recalc_tail_days": 5, "crisis_stress_threshold": 0.70, "min_cluster_weight": 0.02
    },
    "fred_series": {
        "Credit_Spread": "BAA10Y", "Yield_Curve": "T10Y2Y",
        "Long_Rate": "DGS10", "NFCI": "NFCI", "Fed_BS": "WALCL"
    }
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f: yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)

with open(CONFIG_PATH, "r") as f: CFG = yaml.safe_load(f)

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
FRED_SERIES_MAP         = CFG.get("fred_series", DEFAULT_CONFIG["fred_series"])

DATE_STAMP   = datetime.now().strftime("%Y-%m-%d")
MODELS_PATH  = os.path.join(DRIVE_PATH, "models_v92.joblib")
HISTORY_PATH = os.path.join(DRIVE_PATH, "history_v92.parquet")
META_PATH    = os.path.join(DRIVE_PATH, "meta_v92.json")

REGIME_LABELS = {0: "RISK_ON_EXPANSION", 1: "RISK_ON_NORMAL", 2: "TRANSITION", 3: "RISK_OFF_CRISE", 4: "INDETERMINE"}
N_LABELS = len(REGIME_LABELS)
VALID_REGIMES = ["RISK_ON_EXPANSION", "RISK_ON_NORMAL", "TRANSITION", "RISK_OFF_CRISE"]

log.info(f"🚀 PIPELINE MACRO Quant v9.2 (GitHub Actions) | {DATE_STAMP}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. COLLECTE ET FEATURES                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fetch_fmp_stable_eod(symbol: str, start: str = START_DATE) -> pd.Series:
    if not FMP_API_KEY: return pd.Series(dtype=float, name=symbol)
    url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
    chunks = [(start, "2014-12-31"), ("2015-01-01", datetime.now().strftime("%Y-%m-%d"))]
    all_rows = []
    for dt_from, dt_to in chunks:
        if dt_from > dt_to: continue
        try:
            r = requests.get(url, params={"symbol": symbol, "from": dt_from, "to": dt_to, "apikey": FMP_API_KEY}, timeout=15)
            if r.status_code == 200 and isinstance(r.json(), list): all_rows.extend(r.json())
        except Exception: pass
    if not all_rows: return pd.Series(dtype=float, name=symbol)
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    col = "adjClose" if "adjClose" in df.columns else "close"
    return pd.to_numeric(df[col], errors="coerce").dropna().rename(symbol)

def fetch_fred_series_api(series_id: str, start: str, retries: int = 3) -> pd.Series:
    for attempt in range(retries):
        try:
            series = Fred(api_key=FRED_API_KEY).get_series(series_id, observation_start=start)
            series.name = series_id
            return pd.to_numeric(series, errors="coerce")
        except Exception:
            if attempt < retries - 1: time.sleep(2 ** attempt)
            else: return pd.Series(dtype=float, name=series_id)

def collect_data(start_date: str = START_DATE) -> pd.DataFrame:
    log.info("📊 Collecte des données...")
    tickers_yf = ["^VIX", "^VIX9D", "^VIX3M", "^SKEW", "^MOVE", "^VVIX", "DX-Y.NYB", "^DJT", "^GSPC", "XLY", "XLP"]
    raw_yf = yf.download(tickers_yf, start=start_date, progress=False)["Close"]
    df_yf = raw_yf.ffill().rename(columns={
        "^VIX": "VIX", "^VIX9D": "VIX9D", "^VIX3M": "VIX3M", "^SKEW": "SKEW", "^MOVE": "MOVE", "^VVIX": "VVIX",
        "DX-Y.NYB": "DXY", "^DJT": "Transports", "^GSPC": "SP500", "XLY": "Cyclicals", "XLP": "Defensives"
    })

    if FMP_API_KEY:
        fmp_map = {"^GSPC": "SP500", "^VIX": "VIX", "HYG": "HYG", "IEF": "IEF", "XLY": "Cyclicals", "XLP": "Defensives"}
        for fmp_sym, col_name in fmp_map.items():
            s_fmp = fetch_fmp_stable_eod(fmp_sym, start=start_date)
            if len(s_fmp) > 1000:
                if col_name in df_yf.columns: df_yf[col_name] = s_fmp.reindex(df_yf.index).combine_first(df_yf[col_name])
                else: df_yf[col_name] = s_fmp.reindex(df_yf.index)

    fred_series_list = []
    for name, sid in FRED_SERIES_MAP.items():
        s = fetch_fred_series_api(sid, start_date)
        if sid == "BAMLH0A0HYM2" and len(s.dropna()) < 0.60 * len(df_yf):
            s = fetch_fred_series_api("BAA10Y", start_date)
        fred_series_list.append(s.rename(name))

    df_fred = pd.concat(fred_series_list, axis=1)
    df_macro = df_yf.join(df_fred, how="outer")
    for col in list(FRED_SERIES_MAP.keys()) + ["VIX9D", "VIX3M", "HYG", "IEF", "Cyclicals", "Defensives"]:
        if col in df_macro.columns: df_macro[col] = df_macro[col].ffill(limit=10)
    return df_macro.loc[df_yf.index].dropna(subset=["VIX", "DXY", "SP500"])

def engineer_features(df_macro: pd.DataFrame) -> tuple:
    df = df_macro.copy()
    if "NFCI" in df.columns: df["NFCI"] = df["NFCI"].shift(5); df["NFCI_ROC21"] = df["NFCI"].diff(21)
    if "Fed_BS" in df.columns: df["Fed_BS"] = df["Fed_BS"].shift(5); df["Fed_BS_ROC21"] = df["Fed_BS"].pct_change(21)

    df["VIX_SMA21"] = df["VIX"].rolling(21).mean(); df["VIX_STD21"] = df["VIX"].rolling(21).std()
    df["VIX_ZScore"] = (df["VIX"] - df["VIX_SMA21"]) / df["VIX_STD21"]
    df["VIX_PctRank252"] = df["VIX"].rolling(252, min_periods=63).rank(pct=True)
    df["VIX_ROC5"] = df["VIX"].pct_change(5)

    if "VIX9D" in df.columns: df["VIX9D_VIX_Ratio"] = df["VIX9D"] / df["VIX"]
    if "VIX3M" in df.columns: df["VIX_Term_Spread"] = df["VIX"] - df["VIX3M"]
    df["DXY_ROC5"] = df["DXY"].pct_change(5)
    if "VVIX" in df.columns: df["VVIX_ROC5"] = df["VVIX"].pct_change(5)

    df["SP500_SMA200"] = df["SP500"].rolling(200).mean(); df["SP500_Trend"] = (df["SP500"] / df["SP500_SMA200"]) - 1

    if "Cyclicals" in df.columns and "Defensives" in df.columns:
        df["Cyclical_Defensive_Ratio"] = df["Cyclicals"] / df["Defensives"]
        df["Cyclical_Defensive_ROC21"] = df["Cyclical_Defensive_Ratio"].pct_change(21)

    if "Long_Rate" in df.columns: df["Long_Rate_ROC63"] = df["Long_Rate"].pct_change(63)

    credit_col = "Credit_Spread" if "Credit_Spread" in df.columns else "HY_Spread"
    z_fred, roc_fred = None, None
    if credit_col in df.columns:
        c_sma = df[credit_col].rolling(63).mean(); c_std = df[credit_col].rolling(63).std()
        z_fred = (df[credit_col] - c_sma) / c_std; roc_fred = df[credit_col].pct_change(5)

    if "HYG" in df.columns and "IEF" in df.columns and df["HYG"].notna().sum() > 1000:
        hyg_ief_stress = -np.log(df["HYG"] / df["IEF"])
        z_fmp = (hyg_ief_stress - hyg_ief_stress.rolling(63).mean()) / hyg_ief_stress.rolling(63).std()
        df["Credit_ZScore"] = z_fmp.combine_first(z_fred) if z_fred is not None else z_fmp
        df["Credit_ROC5"] = hyg_ief_stress.diff(5).combine_first(roc_fred) if z_fred is not None else hyg_ief_stress.diff(5)
    elif z_fred is not None:
        df["Credit_ZScore"] = z_fred; df["Credit_ROC5"] = roc_fred

    if "Yield_Curve" in df.columns:
        yc_mean = df["Yield_Curve"].rolling(252, min_periods=63).mean(); yc_std = df["Yield_Curve"].rolling(252, min_periods=63).std()
        df["Yield_Curve_ZScore252"] = (df["Yield_Curve"] - yc_mean) / yc_std
        df["Yield_Curve_Momentum"] = df["Yield_Curve"] - df["Yield_Curve"].shift(63)
        df["Yield_Curve_Inversion_Depth"] = np.where(df["Yield_Curve"] < 0, df["Yield_Curve"], 0.0)

    if "Transports" in df.columns: df["Transports_ROC21"] = (df["Transports"] / df["SP500"]).pct_change(21)

    features_ideales = [
        "VIX_ZScore", "VIX_PctRank252", "VIX_ROC5", "VIX_Term_Spread", "VIX9D_VIX_Ratio",
        "SKEW", "MOVE", "VVIX", "VVIX_ROC5", "Credit_ZScore", "Credit_ROC5",
        "SP500_Trend", "DXY_ROC5", "Transports_ROC21", "Cyclical_Defensive_ROC21", "Long_Rate_ROC63",
        "Yield_Curve_ZScore252", "Yield_Curve_Momentum", "Yield_Curve_Inversion_Depth", "NFCI", "NFCI_ROC21", "Fed_BS_ROC21"
    ]
    features_cols = [c for c in features_ideales if c in df.columns and df[c].notna().mean() >= COVERAGE_THRESHOLD]
    
    features_core = ["VIX_ZScore", "SP500_Trend", "Credit_ZScore"]
    df[features_cols] = df[features_cols].ffill(limit=30)
    df = df.dropna(subset=features_core).copy()
    df[features_cols] = df[features_cols].ffill()
    
    for col in features_cols:
        if df[col].isna().sum() > 0: df[col] = df[col].fillna(df[col].rolling(window=ROLLING_IMP_WINDOW, min_periods=1).median())
        if df[col].isna().sum() > 0: df[col] = df[col].ffill()

    return df, features_cols

def validate_features_for_pca(df: pd.DataFrame, features_cols: list) -> list:
    cleaned = []
    for col in features_cols:
        if df[col].var() < 1e-8: continue
        real_changes = df[col].diff().abs() > 1e-10
        if real_changes.any() and (df.index[-1] - df.index[real_changes][-1]).days <= MAX_STALENESS_DAYS:
            cleaned.append(col)
    return cleaned

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. MODÈLES & WFO                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fit_bgmm(X: np.ndarray) -> BayesianGaussianMixture:
    return BayesianGaussianMixture(n_components=N_REGIMES_MAX, covariance_type="full", weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=0.1, n_init=5, max_iter=250, random_state=42).fit(X)

def compute_cluster_stress_scores(centroids_scaled: np.ndarray, features: list) -> np.ndarray:
    scores = np.zeros(centroids_scaled.shape[0])
    weights = {"VIX_ZScore": 0.20, "VIX_PctRank252": 0.20, "Credit_ZScore": 0.25, "Yield_Curve_Inversion_Depth": 0.15, "Long_Rate_ROC63": 0.10, "Cyclical_Defensive_ROC21": -0.10, "SP500_Trend": -0.15}
    total_w = 0.0
    for feat, w in weights.items():
        if feat in features:
            idx = features.index(feat)
            scores += w * centroids_scaled[:, idx]; total_w += abs(w)
    if total_w > 0: scores /= total_w
    if "SP500_Trend" in features:
        trend_idx = features.index("SP500_Trend")
        scores = np.where(centroids_scaled[:, trend_idx] < 0, scores + 0.35 * abs(centroids_scaled[:, trend_idx]), scores)
    return scores

def get_ordered_reference_centroids(bgmm, pca, scaler, features):
    centroids_scaled = pca.inverse_transform(bgmm.means_)
    return centroids_scaled[np.argsort(compute_cluster_stress_scores(centroids_scaled, features))], scaler

def build_stable_mapping(new_bgmm, new_pca, new_scaler, ref_centroids_scaled, ref_scaler, features):
    new_centroids_in_ref_scale = ref_scaler.transform(new_scaler.inverse_transform(new_pca.inverse_transform(new_bgmm.means_)))
    active_mask = new_bgmm.weights_ >= MIN_CLUSTER_WEIGHT
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0: active_indices = np.arange(N_REGIMES_MAX)
    row_ind, col_ind = linear_sum_assignment(np.linalg.norm(new_centroids_in_ref_scale[active_indices, np.newaxis, :] - ref_centroids_scaled[np.newaxis, :, :], axis=2))
    mapping = np.full(N_REGIMES_MAX, N_LABELS - 1, dtype=int)
    for r, c in zip(row_ind, col_ind): mapping[active_indices[r]] = c
    for idx in np.where(~active_mask)[0]: mapping[idx] = int(np.argmin(np.linalg.norm(new_centroids_in_ref_scale[idx] - ref_centroids_scaled, axis=1)))
    stress = compute_cluster_stress_scores(new_centroids_in_ref_scale, features)
    for k in range(N_REGIMES_MAX):
        if mapping[k] == 3 and stress[k] < CRISIS_STRESS_THRESHOLD: mapping[k] = 2
    return mapping

def fit_temporally_calibrated_classifier(base_estimator, X: pd.DataFrame, y: pd.Series):
    n = len(X); split_idx = int(n * 0.80); train_end = max(100, split_idx - 21)
    if train_end >= split_idx or len(np.unique(y.iloc[:train_end])) < 2:
        return CalibratedClassifierCV(clone(base_estimator), method="sigmoid", cv=TimeSeriesSplit(n_splits=3, gap=21)).fit(X, y)
    est = clone(base_estimator); est.fit(X.iloc[:train_end], y.iloc[:train_end])
    try:
        from sklearn.frozen import FrozenEstimator
        cal_model = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid")
    except ImportError:
        cal_model = CalibratedClassifierCV(est, method="sigmoid", cv="prefit")
    cal_model.fit(X.iloc[split_idx:], y.iloc[split_idx:])
    return cal_model

def generate_oof_ew_predictions(df_results: pd.DataFrame, features_cols: list, horizon: int = 21) -> tuple:
    df_ml = df_results.copy()
    df_ml["Target_RiskOff_21d"] = df_ml["Is_RiskOff"].shift(-horizon)
    df_ml = df_ml.dropna(subset=features_cols + ["Target_RiskOff_21d"]).copy()
    df_ml["Target_RiskOff_21d"] = df_ml["Target_RiskOff_21d"].astype(int)
    base_ew = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42, class_weight="balanced")
    tscv = TimeSeriesSplit(n_splits=5, gap=horizon); oof_preds = pd.Series(np.nan, index=df_results.index, dtype=float)
    for train_idx, test_idx in tscv.split(df_ml[features_cols]):
        X_tr, y_tr = df_ml[features_cols].iloc[train_idx], df_ml["Target_RiskOff_21d"].iloc[train_idx]
        if len(np.unique(y_tr)) < 2: continue
        oof_preds.loc[df_ml.index[test_idx]] = fit_temporally_calibrated_classifier(base_ew, X_tr, y_tr).predict_proba(df_ml[features_cols].iloc[test_idx])[:, 1]
    final_ew = fit_temporally_calibrated_classifier(base_ew, df_ml[features_cols], df_ml["Target_RiskOff_21d"])
    tail = df_results.index[df_results.index > df_ml.index[-1]]
    if len(tail) > 0: oof_preds.loc[tail] = final_ew.predict_proba(df_results.loc[tail, features_cols])[:, 1]
    return final_ew, oof_preds

def walk_forward_optimize_ew(df_results: pd.DataFrame, train_window: int = 756, test_window: int = 126) -> pd.DataFrame:
    n_days = len(df_results)
    if n_days < (train_window + test_window):
        df_results["WFO_Threshold"] = 0.50; df_results["WFO_Signal_RiskOff"] = df_results["EW_Proba_RiskOff"] > 0.50
        return df_results
    wfo_signals = pd.Series(False, index=df_results.index); threshold_history = pd.Series(np.nan, index=df_results.index)
    start_idx = train_window
    while start_idx < n_days:
        end_idx = min(start_idx + test_window, n_days)
        df_train = df_results.iloc[start_idx - train_window : start_idx]
        def objective(trial):
            th = trial.suggest_float("threshold", 0.35, 0.75)
            df_sim = df_train.copy(); df_sim["Cash"] = df_sim["EW_Proba_RiskOff"] > th
            df_sim["Ret"] = np.where(df_sim["Cash"], 0.0, df_sim["SP500"].pct_change(1))
            cum = (1 + df_sim["Ret"].fillna(0)).cumprod()
            max_dd = abs(((cum - cum.cummax()) / cum.cummax()).min())
            ann_ret = df_sim["Ret"].mean() * 252
            return float(ann_ret / (max_dd + 1e-4)) if 0.01 <= df_sim["Cash"].mean() <= 0.40 else -999.0
        study = optuna.create_study(direction="maximize"); study.optimize(objective, n_trials=20, show_progress_bar=False)
        best_th = study.best_params["threshold"]
        df_test = df_results.iloc[start_idx:end_idx]
        wfo_signals.loc[df_test.index] = df_test["EW_Proba_RiskOff"] > best_th
        threshold_history.loc[df_test.index] = best_th
        start_idx = end_idx
    df_results["WFO_Threshold"] = threshold_history.ffill().fillna(0.50)
    df_results["WFO_Signal_RiskOff"] = wfo_signals
    return df_results

def train_regime_forecaster(df_results: pd.DataFrame, features_cols: list, horizon: int = 21):
    df_ml = df_results.copy(); df_ml["Target_Regime"] = df_ml["Regime_IA"].shift(-horizon)
    df_ml = df_ml.dropna(subset=features_cols + ["Target_Regime"])
    df_ml = df_ml[df_ml["Target_Regime"].isin(VALID_REGIMES)]
    return CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42), method="sigmoid", cv=TimeSeriesSplit(n_splits=3, gap=horizon)).fit(df_ml[features_cols], df_ml["Target_Regime"])

def evaluate_market_reality(df_results: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    df_eval = df_results.copy(); sp_ret = df_eval["SP500"].pct_change()
    df_eval["Fwd_Realized_Vol_21d"] = sp_ret.rolling(horizon).std().shift(-horizon) * np.sqrt(252) * 100
    df_eval["Fwd_Return_21d"] = (df_eval["SP500"].shift(-horizon) / df_eval["SP500"] - 1.0) * 100
    fwd_max_dd = np.full(len(df_eval), np.nan); sp_vals = df_eval["SP500"].values
    for idx in range(len(sp_vals) - horizon):
        fwd_max_dd[idx] = ((np.min(sp_vals[idx: idx + horizon + 1]) / sp_vals[idx]) - 1.0) * 100
    df_eval["Fwd_MaxDD_21d"] = fwd_max_dd
    return df_eval[df_eval["Regime_IA"].isin(VALID_REGIMES)].groupby("Regime_IA")[["Fwd_Return_21d", "Fwd_Realized_Vol_21d", "Fwd_MaxDD_21d"]].agg(["count", "mean", "median"])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. EXÉCUTION & ENVOI MAIL                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def send_email_notification(subject, html_content):
    if not EMAIL_USER or not EMAIL_PASS:
        log.warning("⚠️ Identifiants e-mail non configurés. E-mail ignoré.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject; msg["From"] = EMAIL_USER; msg["To"] = EMAIL_USER
    msg.attach(MIMEText(html_content, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string())
        log.info("📧 E-mail de notification envoyé avec succès !")
    except Exception as e:
        log.error(f"❌ Erreur envoi mail : {e}")

def run_pipeline():
    df_macro = collect_data()
    df_model, features_cols = engineer_features(df_macro)
    features_cols = validate_features_for_pca(df_model, features_cols)
    X_all = df_model[features_cols].values

    scaler_ref = StandardScaler(); pca_ref = PCA(n_components=PCA_VARIANCE_TARGET, random_state=42)
    bgmm_ref = fit_bgmm(pca_ref.fit_transform(scaler_ref.fit_transform(X_all[:MIN_TRAIN_DAYS])))
    ref_centroids, ref_scaler = get_ordered_reference_centroids(bgmm_ref, pca_ref, scaler_ref, features_cols)
    current_mapping = build_stable_mapping(bgmm_ref, pca_ref, scaler_ref, ref_centroids, ref_scaler, features_cols)
    current_bgmm, current_scaler, current_pca, last_refit_idx = bgmm_ref, scaler_ref, pca_ref, MIN_TRAIN_DAYS

    total_days = len(df_model); regimes, riskoff, posteriors, p_exp, p_norm, p_trans, p_cris = [], [], [], [], [], [], []
    i = MIN_TRAIN_DAYS
    while i < total_days:
        if (i - last_refit_idx) >= REFIT_FREQ:
            new_scaler = StandardScaler(); new_pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=42)
            new_bgmm = fit_bgmm(new_pca.fit_transform(new_scaler.fit_transform(X_all[max(0, i - WINDOW_SIZE):i])))
            current_mapping = build_stable_mapping(new_bgmm, new_pca, new_scaler, ref_centroids, ref_scaler, features_cols)
            current_bgmm, current_scaler, current_pca, last_refit_idx = new_bgmm, new_scaler, new_pca, i
        end_batch = min(i + REFIT_FREQ - (i - last_refit_idx), total_days); batch_size = end_batch - i
        proba_vecs = current_bgmm.predict_proba(current_pca.transform(current_scaler.transform(X_all[i:end_batch])))
        for j in range(batch_size):
            mapped = np.zeros(N_REGIMES_MAX)
            for raw in range(N_REGIMES_MAX):
                rn = current_mapping[raw]
                if rn < N_REGIMES_MAX: mapped[rn] += proba_vecs[j, raw]
            stable_rank = int(np.argmax(mapped))
            regime = REGIME_LABELS.get(min(stable_rank, N_LABELS - 1), "INDETERMINE")
            regimes.append(regime); riskoff.append(regime == "RISK_OFF_CRISE"); posteriors.append(float(mapped.max()))
            p_exp.append(float(mapped[0])); p_norm.append(float(mapped[1])); p_trans.append(float(mapped[2])); p_cris.append(float(mapped[3]))
        i = end_batch

    df_results = df_model.iloc[MIN_TRAIN_DAYS:].copy()
    df_results["Regime_IA"] = regimes; df_results["Is_RiskOff"] = riskoff; df_results["Posterior_BGMM"] = posteriors
    df_results["Proba_Expansion"] = p_exp; df_results["Proba_Normal"] = p_norm; df_results["Proba_Transition"] = p_trans; df_results["Proba_Crise"] = p_cris

    ew_model, oof_ew = generate_oof_ew_predictions(df_results, features_cols, horizon=21)
    df_results["EW_Proba_RiskOff"] = oof_ew
    df_results = walk_forward_optimize_ew(df_results)
    forecaster = train_regime_forecaster(df_results, features_cols, horizon=21)

    # Résultats du jour
    current_regime = df_results["Regime_IA"].iloc[-1]
    posterior_val = df_results["Posterior_BGMM"].iloc[-1]
    ew_prob = df_results["EW_Proba_RiskOff"].iloc[-1]
    wfo_th = df_results["WFO_Threshold"].iloc[-1]
    is_risk_off = df_results["WFO_Signal_RiskOff"].iloc[-1]
    
    latest_feat = df_results[features_cols].iloc[-1:]
    forecaster_preds = {c: float(p) for c, p in zip(forecaster.classes_, forecaster.predict_proba(latest_feat)[0])}
    forecast_str = " | ".join([f"{c.split('_')[-1]}: {p:.1%}" for c, p in forecaster_preds.items()])

    html_body = f"""
    <h2>🤖 Rapport Oracle Macro-Micro v9.2</h2>
    <p><b>Date :</b> {DATE_STAMP}</p>
    <ul>
        <li>📍 <b>Régime Actuel (Nowcast) :</b> {current_regime} (Posterior : {posterior_val:.1%})</li>
        <li>🚨 <b>Early Warning (21j) :</b> {ew_prob:.1%}</li>
        <li>🎯 <b>Seuil WFO Dynamique :</b> {wfo_th:.2f}</li>
        <li>⚡ <b>Signal Final WFO :</b> <span style="color: {'red' if is_risk_off else 'green'};"><b>{'RISK-OFF' if is_risk_off else 'NORMAL'}</b></span></li>
    </ul>
    <p><b>🔮 Prévisions à 21 jours :</b> {forecast_str}</p>
    """
    
    log.info(html_body.replace("<br>", "\n").replace("<h2>", "").replace("</h2>", ""))
    send_email_notification(f"📊 Alerte Oracle Macro — {DATE_STAMP} : {'RISK-OFF' if is_risk_off else 'NORMAL'}", html_body)

if __name__ == "__main__":
    run_pipeline()
