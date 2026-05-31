"""
Accelerations — Daily Accelerations Prediction.

Target: accelerations
Pipeline: datapipeline.run_pipeline("accelerations")
Model:    XGBoost, RandomizedSearchCV (10-fold KFold), log1p target.
"""


# =============================================================================
# 1. Imports & Setup
# =============================================================================


import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor

from real_madrid_acwr.config import MODEL_ARTIFACTS_DIR
from real_madrid_acwr.modeling.datapipeline import prepare_test, run_pipeline

warnings.filterwarnings("ignore")

TARGET     = "accelerations"
TARGET_DIR = MODEL_ARTIFACTS_DIR / TARGET


def main() -> None:

    # =============================================================================
    # 2. Data Preprocessing Pipeline
    # =============================================================================

    pipeline     = run_pipeline(TARGET)
    X_tr_np      = pipeline["X_tr_np"]
    y_tr_np      = pipeline["y_tr_np"]
    test_base    = pipeline["test_base"]
    scaler       = pipeline["scaler"]
    feature_cols = pipeline["feature_cols"]

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")


    # =============================================================================
    # 4. Model Training
    # =============================================================================

    cv10 = KFold(n_splits=10, shuffle=True, random_state=42)

    def _run_search(estimator, space, n_iter=50):
        s = RandomizedSearchCV(
            estimator=estimator, param_distributions=space,
            n_iter=n_iter, cv=cv10, scoring="neg_mean_squared_error",
            n_jobs=-1, random_state=42, refit=True, error_score=np.nan,
        )
        s.fit(X_tr_np, y_tr_np)
        return s


    # =============================================================================
    # 4a. XGBoost
    # =============================================================================

    xgb_space = {
        "n_estimators":      randint(200, 3000),   "max_depth":         randint(2, 16),
        "max_leaves":        randint(0, 1024),      "min_child_weight":  loguniform(0.1, 500),
        "gamma":             loguniform(1e-6, 100), "max_delta_step":    randint(0, 10),
        "grow_policy":       ["depthwise", "lossguide"],
        "max_bin":           randint(64, 1024),
        "subsample":         uniform(0.3, 0.7),     "colsample_bytree":  uniform(0.3, 0.7),
        "colsample_bylevel": uniform(0.3, 0.7),     "colsample_bynode":  uniform(0.3, 0.7),
        "learning_rate":     loguniform(1e-3, 5e-1),
        "reg_alpha":         loguniform(1e-8, 100), "reg_lambda":        loguniform(1e-8, 100),
    }
    print("Training XGBoost (50 iters, 10-fold CV)…")
    xgb_search = _run_search(XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, tree_method="hist"), xgb_space)
    xgb_best   = xgb_search.best_estimator_
    print(f"  Best CV RMSE (log1p): {np.sqrt(-xgb_search.best_score_):.4f}")
    print(f"  Best params: {xgb_search.best_params_}")


    # =============================================================================
    # 6. Final Evaluation on Test Set
    # =============================================================================

    test      = prepare_test(test_base, TARGET, scaler, feature_cols)
    X_te      = test[feature_cols].to_numpy()
    y_te      = test[TARGET].to_numpy()
    actual_tr = np.expm1(y_tr_np)
    actual_te = np.expm1(y_te)

    p_tr = np.clip(np.expm1(xgb_best.predict(X_tr_np)), 0, None)
    p_te = np.clip(np.expm1(xgb_best.predict(X_te)),    0, None)

    results = pd.DataFrame([{
        "model":      "XGBoost",
        "train_rmse": float(np.sqrt(mean_squared_error(actual_tr, p_tr))),
        "train_mae":  float(mean_absolute_error(actual_tr, p_tr)),
        "train_r2":   float(r2_score(actual_tr, p_tr)),
        "cv_rmse*":   float(np.sqrt(-xgb_search.best_score_)),
        "test_rmse":  float(np.sqrt(mean_squared_error(actual_te, p_te))),
        "test_mae":   float(mean_absolute_error(actual_te, p_te)),
        "test_r2":    float(r2_score(actual_te, p_te)),
    }])

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 120)
    print("\n" + "─"*90)
    print(f"Final results (* cv_rmse on log1p scale; train/test metrics in count)")
    print("─"*90)
    print(results.to_string(index=False))


    # =============================================================================
    # 8. Save bundle for app
    # =============================================================================

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":        xgb_best,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "ewma_spans":   {"acute": 7, "chronic": 28},
    }
    joblib.dump(bundle, TARGET_DIR / "bundle.joblib")
    print(f"\nSaved: {(TARGET_DIR / 'bundle.joblib').relative_to(MODEL_ARTIFACTS_DIR.parent)}")


if __name__ == "__main__":
    main()
