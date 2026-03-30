"""
Climate & Health Risk Prediction - Final Model with Enhanced Features
Using existing climate_features.csv with advanced feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

DATA_PATH = "data/"
OUTPUT_PATH = "data/submission.csv"

print("=" * 70)
print("CLIMATE & HEALTH - ENHANCED MODEL WITH EXTERNAL DATA")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading Data...")
train = pd.read_csv(f"{DATA_PATH}Train.csv")
test = pd.read_csv(f"{DATA_PATH}Test.csv")
climate_features = pd.read_csv(f"{DATA_PATH}climate_features.csv")

# Check for external data
external_climate_path = f"{DATA_PATH}external/openmeteo_climate.csv"
has_external = False

try:
    external_climate = pd.read_csv(external_climate_path)
    has_external = True
    print(f"Found external climate data: {external_climate.shape}")
except:
    print("No external data found, using provided climate features")
    external_climate = None

ID_COL = "ID"
TARGET = "is_climate_sensitive"

train_ids = train[ID_COL].values
test_ids = test[ID_COL].values
y = train[TARGET].values

print(f"Train: {train.shape}, Test: {test.shape}")

# =============================================================================
# 2. MERGE CLIMATE FEATURES
# =============================================================================
print("\n[2] Merging Climate Features...")
climate_features = climate_features.drop(columns=["deathdate"], errors="ignore")
train = train.merge(climate_features, on=ID_COL, how="left")
test = test.merge(climate_features, on=ID_COL, how="left")
print(f"After merge - Train: {train.shape}, Test: {test.shape}")

# Add external data as new features
if has_external:
    print("\n[2b] Adding External Climate Data...")
    external_climate["date"] = pd.to_datetime(external_climate["date"])
    train["deathdate"] = pd.to_datetime(train["deathdate"])
    test["deathdate"] = pd.to_datetime(test["deathdate"])

    # Create keys for joining
    train["lat_round"] = train["latitude"].round(1)
    train["lon_round"] = train["longitude"].round(1)
    test["lat_round"] = test["latitude"].round(1)
    test["lon_round"] = test["longitude"].round(1)
    train["merge_key"] = (
        train["lat_round"].astype(str) + "_" + train["lon_round"].astype(str)
    )
    test["merge_key"] = (
        test["lat_round"].astype(str) + "_" + test["lon_round"].astype(str)
    )
    external_climate["merge_key"] = (
        external_climate["lat_round"].astype(str)
        + "_"
        + external_climate["lon_round"].astype(str)
    )

    # Aggregate external data by location
    external_loc_agg = (
        external_climate.groupby("merge_key")
        .agg(
            {
                "temp_max": "mean",
                "temp_min": "mean",
                "temp_mean": "mean",
                "precipitation": "mean",
                "rain": "mean",
                "humidity": "mean",
            }
        )
        .reset_index()
    )
    external_loc_agg.columns = [
        "merge_key",
        "ext_temp_max",
        "ext_temp_min",
        "ext_temp_mean",
        "ext_precip",
        "ext_rain",
        "ext_humidity",
    ]

    # Merge location-level external data
    train = train.merge(external_loc_agg, on="merge_key", how="left")
    test = test.merge(external_loc_agg, on="merge_key", how="left")

    # Clean up
    train = train.drop(columns=["lat_round", "lon_round", "merge_key"])
    test = test.drop(columns=["lat_round", "lon_round", "merge_key"])

    print(f"After external data - Train: {train.shape}, Test: {test.shape}")

# =============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n[3] Advanced Feature Engineering...")


def engineer_features(df):
    """Enhanced feature engineering using all available data"""
    df = df.copy()

    # =================================================================
    # TEMPORAL FEATURES
    # =================================================================
    df["deathdate"] = pd.to_datetime(df["deathdate"])
    df["year"] = df["deathdate"].dt.year
    df["month"] = df["deathdate"].dt.month
    df["day"] = df["deathdate"].dt.day
    df["day_of_week"] = df["deathdate"].dt.dayofweek
    df["day_of_year"] = df["deathdate"].dt.dayofyear
    df["week_of_year"] = df["deathdate"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["deathdate"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # =================================================================
    # SEASONAL FEATURES (Critical for tropical climate)
    # =================================================================
    # Uganda has two rainy seasons: Mar-May and Sep-Nov
    df["is_rainy_season"] = df["month"].isin([3, 4, 5, 9, 10, 11]).astype(int)
    df["is_dry_season"] = df["month"].isin([1, 2, 6, 7, 8, 12]).astype(int)
    df["is_long_rain"] = df["month"].isin([3, 4, 5]).astype(int)
    df["is_short_rain"] = df["month"].isin([9, 10, 11]).astype(int)
    df["is_transition"] = df["month"].isin([2, 6, 8]).astype(int)

    # =================================================================
    # TEMPERATURE FEATURES
    # =================================================================
    # Daily range
    df["temp_range"] = df["max_temperature"] - df["min_temperature"]

    # Anomalies vs moving averages
    df["temp_vs_tavg30"] = df["avg_temperature"] - df["tavg_30d"]
    df["temp_vs_tavg90"] = df["avg_temperature"] - df["tavg_90d"]
    df["temp_anomaly_abs"] = df["temp_vs_tavg30"].abs()

    # Temperature extremes
    df["tmax_vs_tavg"] = df["max_temperature"] - df["avg_temperature"]
    df["tmin_vs_tavg"] = df["avg_temperature"] - df["min_temperature"]
    df["tmax_vs_tmax30"] = df["max_temperature"] - df["tmax_30d"]
    df["tmin_vs_tmin30"] = df["min_temperature"] - df["tmin_30d"]

    # 30-day temperature variability
    df["tmax_tmin_range_30d"] = df["tmax_30d"] - df["tmin_30d"]
    df["temp_stability"] = 1 / (df["temp_range_mean_30d"] + 0.1)
    df["temp_volatility"] = df["temp_range_mean_30d"] / (df["tavg_30d"] + 1)

    # Temperature trends
    df["tavg_trend_7d_30d"] = df["tavg_7d"] - df["tavg_30d"]
    df["tavg_trend_30d_90d"] = df["tavg_30d"] - df["tavg_90d"]
    df["tavg_acceleration"] = df["tavg_trend_7d_30d"] - df["tavg_trend_30d_90d"]

    # =================================================================
    # RAINFALL FEATURES
    # =================================================================
    # Ratios at different timescales
    df["rain_ratio_7d_30d"] = df["rain_sum_7d"] / (df["rain_sum_30d"] + 1)
    df["rain_ratio_30d_90d"] = df["rain_sum_30d"] / (df["rain_sum_90d"] + 1)

    # Rainfall intensity
    df["rain_intensity"] = df["rain_sum_30d"] / (df["rain_days_30d"] + 1)
    df["rain_per_day_7d"] = df["rain_sum_7d"] / 7
    df["rain_per_day_30d"] = df["rain_sum_30d"] / 30

    # Rainfall extremes
    df["max_rain_vs_avg"] = df["max_daily_rain_30d"] / (
        df["rain_sum_30d"] / (df["rain_days_30d"] + 1) + 0.1
    )
    df["heavy_rain_event"] = (df["max_daily_rain_30d"] > 20).astype(int)
    df["very_heavy_rain"] = (df["max_daily_rain_30d"] > 30).astype(int)

    # Rainfall trends
    df["rain_trend"] = df["rain_sum_7d"] / 7 - df["rain_sum_30d"] / 30

    # =================================================================
    # VEGETATION FEATURES (NDVI)
    # =================================================================
    df["ndvi_change"] = df["ndvi_30d"] - df["ndvi_90d"]
    df["ndvi_ratio"] = df["ndvi_30d"] / (df["ndvi_90d"] + 0.01)
    df["ndvi_product"] = df["ndvi_30d"] * df["ndvi_90d"]
    df["low_vegetation"] = (df["ndvi_30d"] < 0.4).astype(int)
    df["high_vegetation"] = (df["ndvi_30d"] > 0.7).astype(int)

    # =================================================================
    # INTERACTION FEATURES
    # =================================================================
    df["temp_rain_interaction"] = df["avg_temperature"] * df["precipitation"]
    df["temp_rain_interaction_30d"] = df["tavg_30d"] * df["rain_sum_30d"] / 1000
    df["temp_rain_ratio"] = df["avg_temperature"] / (df["precipitation"] + 0.1)
    df["hot_rain_interaction"] = df["hot_days_30d"] * df["rain_sum_30d"] / 100

    # =================================================================
    # TERRAIN FEATURES
    # =================================================================
    df["elevation_slope_interaction"] = df["elevation"] * df["slope"] / 1000
    df["is_high_elevation"] = (df["elevation"] > 1200).astype(int)
    df["is_steep_terrain"] = (df["slope"] > 1).astype(int)

    # =================================================================
    # DEMOGRAPHIC FEATURES
    # =================================================================
    df["age_squared"] = df["age"] ** 2
    df["age_log"] = np.log1p(df["age"])
    df["age_group"] = pd.cut(
        df["age"], bins=[-1, 1, 5, 14, 25, 45, 65, 200], labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(float)
    df["is_infant"] = (df["age"] <= 1).astype(int)
    df["is_child"] = ((df["age"] > 1) & (df["age"] <= 14)).astype(int)
    df["is_elderly"] = (df["age"] >= 65).astype(int)
    df["is_young_adult"] = ((df["age"] > 14) & (df["age"] <= 35)).astype(int)

    # =================================================================
    # CLIMATE STRESS INDICATORS
    # =================================================================
    # Temperature stress
    df["cold_stress"] = (df["tmin_30d"] < 15).astype(int)
    df["heat_stress"] = (df["tmax_30d"] > 30).astype(int)
    df["extreme_heat"] = (df["hot_days_30d"] > 0).astype(int)
    df["heat_stress_days"] = df["hot_days_30d"]

    # Rainfall stress
    df["drought_indicator"] = (
        df["rain_sum_30d"] < df["rain_sum_30d"].quantile(0.1)
    ).astype(int)
    df["flood_indicator"] = (
        df["rain_sum_30d"] > df["rain_sum_30d"].quantile(0.9)
    ).astype(int)

    # Combined stress
    df["combined_stress"] = df["cold_stress"] + df["heat_stress"] + df["extreme_heat"]

    # =================================================================
    # ENCODING
    # =================================================================
    df["zone_encoded"] = (df["zone"] == "Rural").astype(int)
    df["gender_encoded"] = (df["gender"] == "Male").astype(int)

    # =================================================================
    # IMPORTANT INTERACTIONS
    # =================================================================
    df["infant_rainy"] = df["is_infant"] * df["is_rainy_season"]
    df["elderly_heat"] = df["is_elderly"] * df["heat_stress"]
    df["child_flood"] = df["is_child"] * df["heavy_rain_event"]
    df["temp_age"] = df["avg_temperature"] * df["age"]
    df["rain_age"] = df["rain_sum_30d"] * df["age"] / 1000
    df["zone_precip"] = df["zone_encoded"] * df["precipitation"]

    # =================================================================
    # EXTERNAL DATA FEATURES (if available)
    # =================================================================
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["high_humidity"] = (df["humidity"] > 80).astype(int)

    if "soil_moisture" in df.columns:
        df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
        df["wet_soil"] = (df["soil_moisture"] > 0.5).astype(int)

    return df


train = engineer_features(train)
test = engineer_features(test)

print(f"After feature engineering - Train: {train.shape}, Test: {test.shape}")

# =============================================================================
# 4. PREPARE FEATURES
# =============================================================================
print("\n[4] Preparing Features...")

drop_cols = [ID_COL, "deathdate", "location", "region", "lat_lon_combined", TARGET]

feature_cols = [col for col in train.columns if col not in drop_cols]
print(f"Number of features: {len(feature_cols)}")

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Label encode categorical columns
for col in ["zone", "gender"]:
    if col in X_train.columns:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# Handle any remaining object columns
for col in X_train.columns:
    if X_train[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat(
            [X_train[col].astype(str), X_test[col].astype(str)], axis=0
        )
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Fill missing values with train median (NO LEAKAGE)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# Handle infinite values
X_train = X_train.replace([np.inf, -np.inf], 0)
X_test = X_test.replace([np.inf, -np.inf], 0)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================================
# 5. CROSS-VALIDATION TRAINING
# =============================================================================
print("\n[5] Cross-Validation Training...")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

N_SPLITS = 5
RANDOM_STATE = 42

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_rf = np.zeros(len(X_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))
test_rf = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\n{'=' * 50}")
    print(f"FOLD {fold + 1}/{N_SPLITS}")
    print(f"{'=' * 50}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # LightGBM
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 500,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # XGBoost
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 500,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

    # CatBoost
    model_cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    oof_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / N_SPLITS

    # RandomForest
    model_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model_rf.fit(X_tr, y_tr)

    oof_rf[val_idx] = model_rf.predict_proba(X_val)[:, 1]
    test_rf += model_rf.predict_proba(X_test)[:, 1] / N_SPLITS

    # Print fold results
    f1_lgb = f1_score(y_val, (oof_lgb[val_idx] >= 0.5).astype(int))
    f1_xgb = f1_score(y_val, (oof_xgb[val_idx] >= 0.5).astype(int))
    f1_cat = f1_score(y_val, (oof_cat[val_idx] >= 0.5).astype(int))
    f1_rf = f1_score(y_val, (oof_rf[val_idx] >= 0.5).astype(int))

    print(f"  LGB: F1={f1_lgb:.4f}")
    print(f"  XGB: F1={f1_xgb:.4f}")
    print(f"  CAT: F1={f1_cat:.4f}")
    print(f"  RF:  F1={f1_rf:.4f}")

# =============================================================================
# 6. ENSEMBLE
# =============================================================================
print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

# Weighted ensemble (same weights that worked well)
weights = {"lgb": 0.30, "xgb": 0.25, "cat": 0.30, "rf": 0.15}

ensemble_oof = (
    weights["lgb"] * oof_lgb
    + weights["xgb"] * oof_xgb
    + weights["cat"] * oof_cat
    + weights["rf"] * oof_rf
)
ensemble_test = (
    weights["lgb"] * test_lgb
    + weights["xgb"] * test_xgb
    + weights["cat"] * test_cat
    + weights["rf"] * test_rf
)

# Print individual model scores
for name, oof in [
    ("LGBM", oof_lgb),
    ("XGB", oof_xgb),
    ("CAT", oof_cat),
    ("RF", oof_rf),
]:
    f1 = f1_score(y, (oof >= 0.5).astype(int))
    auc = roc_auc_score(y, oof)
    weighted = 0.6 * f1 + 0.4 * auc
    print(f"{name}: F1={f1:.4f}, AUC={auc:.4f}, Weighted={weighted:.4f}")

# Print ensemble scores
ensemble_f1 = f1_score(y, (ensemble_oof >= 0.5).astype(int))
ensemble_auc = roc_auc_score(y, ensemble_oof)
ensemble_weighted = 0.6 * ensemble_f1 + 0.4 * ensemble_auc

print(
    f"\nENSEMBLE: F1={ensemble_f1:.4f}, AUC={ensemble_auc:.4f}, Weighted={ensemble_weighted:.4f}"
)

# =============================================================================
# 7. GENERATE SUBMISSION
# =============================================================================
print("\n[6] Generating Submission...")

test_preds_binary = (ensemble_test >= 0.5).astype(int)

print(f"Test prediction distribution:")
print(
    f"  Class 0: {sum(test_preds_binary == 0)} ({sum(test_preds_binary == 0) / len(test_preds_binary):.1%})"
)
print(
    f"  Class 1: {sum(test_preds_binary == 1)} ({sum(test_preds_binary == 1) / len(test_preds_binary):.1%})"
)

submission = pd.DataFrame(
    {"ID": test_ids, "TargetF1": test_preds_binary, "TargetRAUC": ensemble_test}
)

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Cross-Validation Weighted Score: {ensemble_weighted:.4f}")
print(f"  - F1-Score (60%): {ensemble_f1:.4f}")
print(f"  - ROC-AUC (40%): {ensemble_auc:.4f}")
print(f"\nFeatures: {X_train.shape[1]}")
print(f"Cross-Validation: StratifiedKFold (5 folds)")
print(f"Models: LightGBM, XGBoost, CatBoost, RandomForest")

# Feature importance
print("\nTop 20 Important Features (LGBM):")
feat_imp = pd.DataFrame(
    {"feature": X_train.columns, "importance": model_lgb.feature_importances_}
).sort_values("importance", ascending=False)
print(feat_imp.head(20).to_string(index=False))
