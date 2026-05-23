"""
Diabetes Risk — XGBoost Model Training (v4.0)
Label: HbA1c mezoniga qarab (DB risk_level dan EMAS)
Leakage yo'q: HbA1c, PPG, FBG feature dan olib tashlandi
"""

import os, json, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "diabetes_risk"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

MODEL_PATH    = "model.pkl"
METADATA_PATH = "model_metadata.json"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
HBA1C_PRE     = 5.7
HBA1C_DM      = 6.5
RISK_NAMES    = {0: "healthy", 1: "prediabetes", 2: "diabetes"}
NUM_CLASSES   = 3
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# HbA1c, FBG, PPG OLIB TASHLANDI — leakage oldini olish
FEATURE_COLS = [
    "age", "gender_num", "family_history",
    "fasting_insulin", "homa_ir", "bmi", "waist_cm",
    "systolic_bp", "diastolic_bp", "total_cholesterol",
    "hdl", "ldl", "triglycerides", "tg_hdl_ratio",
    "crp", "creatinine", "egfr", "adiponectin",
]

ML_QUERY = """
SELECT
    p.id,
    DATE_PART('year', AGE(p.date_of_birth))::int            AS age,
    CASE WHEN p.gender = 'male' THEN 1 ELSE 0 END           AS gender_num,
    COALESCE(
        (SELECT MAX(CASE WHEN fh.has_diabetes THEN 1 ELSE 0 END)
         FROM family_history fh WHERE fh.patient_id = p.id), 0
    )                                                        AS family_history,
    MAX(CASE WHEN b.code = 'HBA1C'  THEN tr.value END)      AS hba1c,
    MAX(CASE WHEN b.code = 'FINS'   THEN tr.value END)      AS fasting_insulin,
    MAX(CASE WHEN b.code = 'HOMAIR' THEN tr.value END)      AS homa_ir,
    MAX(CASE WHEN b.code = 'BMI'    THEN tr.value END)      AS bmi,
    MAX(CASE WHEN b.code = 'WAIST'  THEN tr.value END)      AS waist_cm,
    MAX(CASE WHEN b.code = 'SBP'    THEN tr.value END)      AS systolic_bp,
    MAX(CASE WHEN b.code = 'DBP'    THEN tr.value END)      AS diastolic_bp,
    MAX(CASE WHEN b.code = 'TCHOL'  THEN tr.value END)      AS total_cholesterol,
    MAX(CASE WHEN b.code = 'HDL'    THEN tr.value END)      AS hdl,
    MAX(CASE WHEN b.code = 'LDL'    THEN tr.value END)      AS ldl,
    MAX(CASE WHEN b.code = 'TG'     THEN tr.value END)      AS triglycerides,
    MAX(CASE WHEN b.code = 'TGHDL'  THEN tr.value END)      AS tg_hdl_ratio,
    MAX(CASE WHEN b.code = 'CRP'    THEN tr.value END)      AS crp,
    MAX(CASE WHEN b.code = 'CREAT'  THEN tr.value END)      AS creatinine,
    MAX(CASE WHEN b.code = 'EGFR'   THEN tr.value END)      AS egfr,
    MAX(CASE WHEN b.code = 'ADIPO'  THEN tr.value END)      AS adiponectin
FROM patients p
JOIN medical_tests mt ON mt.patient_id      = p.id
JOIN test_results  tr ON tr.medical_test_id = mt.id
JOIN biomarkers    b  ON b.id               = tr.biomarker_id
GROUP BY p.id, p.date_of_birth, p.gender
ORDER BY RANDOM()
"""

def load_data():
    url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
           f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
    engine = create_engine(url)
    print("  DB dan ma'lumot yuklanmoqda...")
    with engine.connect() as conn:
        result = conn.execute(text(ML_QUERY))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    print(f"  {len(df):,} ta yozuv yuklandi")
    return df

def prepare_data(df):
    df["hba1c"] = pd.to_numeric(df["hba1c"], errors="coerce")
    df = df.dropna(subset=["hba1c"]).copy()
    df["label"] = df["hba1c"].apply(
        lambda x: 0 if x < HBA1C_PRE else (1 if x < HBA1C_DM else 2)
    )
    X = df[FEATURE_COLS].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    y = df["label"].astype(int)

    print(f"\n  Feature lar  : {X.shape[1]}")
    print(f"  Satrlar      : {X.shape[0]:,}")
    print(f"\n  Sinf taqsimoti (HbA1c mezoniga qarab):")
    for label in sorted(y.unique()):
        count = (y == label).sum()
        print(f"    {RISK_NAMES[label]:<14} {count:>7,}  ({count/len(y)*100:.1f}%)")
    return X, y

def train_model(X_train, y_train):
    print("\n  XGBoost o'qitilmoqda...")
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model

def evaluate_model(model, X_test, y_test, X, y):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    print("  Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n{'='*52}")
    print(f"  MODEL NATIJALARI")
    print(f"{'='*52}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"\n  Sinf bo'yicha:")
    print(classification_report(y_test, y_pred,
        target_names=[RISK_NAMES[i] for i in range(NUM_CLASSES)], digits=3))
    return {
        "accuracy": round(acc, 4), "f1_score": round(f1, 4),
        "auc_roc":  round(auc, 4), "cv_mean":  round(float(cv_scores.mean()), 4),
        "cv_std":   round(float(cv_scores.std()), 4),
    }

def print_feature_importance(model):
    fi = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importance (top 10):")
    print(f"  {'Feature':<25} {'Score':>8}  Bar")
    print(f"  {'-'*50}")
    for name, score in fi[:10]:
        print(f"  {name:<25} {score:>8.4f}  {'x'*int(score*200)}")
    return {name: round(float(score), 6) for name, score in fi}

def update_db_model(metrics):
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("UPDATE ml_models SET is_active = FALSE")
        cur.execute("""
            INSERT INTO ml_models
              (model_name, version, algorithm, accuracy, precision_score,
               recall, f1_score, auc_roc, feature_list, is_active, trained_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,NOW())
            ON CONFLICT (model_name, version) DO UPDATE SET
              accuracy=EXCLUDED.accuracy, f1_score=EXCLUDED.f1_score,
              auc_roc=EXCLUDED.auc_roc, is_active=TRUE, trained_at=NOW()
        """, ("DiabetesRiskClassifier","v4.0","XGBoost",
              metrics["accuracy"],metrics["accuracy"],
              metrics["f1_score"],metrics["f1_score"],
              metrics["auc_roc"],FEATURE_COLS))
        conn.commit(); cur.close(); conn.close()
        print("  [OK] DB ga model yozildi (v4.0)")
    except Exception as e:
        print(f"  [WARN] {e}")

def save_model(model, metrics, fi):
    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] Model saqlandi: {MODEL_PATH}")
    metadata = {
        "version": "v4.0", "algorithm": "XGBoost",
        "trained_at": datetime.now().isoformat(),
        "feature_cols": FEATURE_COLS,
        "hba1c_thresholds": {"prediabetes": HBA1C_PRE, "diabetes": HBA1C_DM},
        "risk_names": RISK_NAMES, "metrics": metrics,
        "feature_importance": fi,
        "note": "Label HbA1c ADA mezoniga qarab. HbA1c/FBG/PPG feature dan chiqarildi.",
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metadata saqlandi: {METADATA_PATH}")

def main():
    print("="*52)
    print("  Diabetes Risk — XGBoost v4.0")
    print("  Label: HbA1c ADA mezoni (leakage yo'q)")
    print("="*52)

    print("\n[1/5] Ma'lumot yuklanmoqda...")
    df = load_data()

    print("\n[2/5] Ma'lumot tayyorlanmoqda...")
    X, y = prepare_data(df)

    print("\n[3/5] Train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n[4/5] Model o'qitilmoqda...")
    model = train_model(X_train, y_train)

    print("\n[5/5] Baholanmoqda...")
    metrics = evaluate_model(model, X_test, y_test, X, y)
    fi = print_feature_importance(model)

    print(f"\n{'='*52}\n  Saqlanmoqda...")
    save_model(model, metrics, fi)
    update_db_model(metrics)

    # ── MLflow ────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("DiabetesRisk")
        with mlflow.start_run(run_name="XGBoost_v4.0"):
            mlflow.log_params({
                "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 3, "gamma": 0.1,
                "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics({
                "accuracy":  metrics["accuracy"],
                "f1_score":  metrics["f1_score"],
                "auc_roc":   metrics["auc_roc"],
                "cv_mean":   metrics["cv_mean"],
                "cv_std":    metrics["cv_std"],
            })
            for fname, score in fi.items():
                mlflow.log_metric(f"fi_{fname}", score)
            print(f"  [OK] MLflow ga yozildi: {MLFLOW_URI}")
    except Exception as e:
        print(f"  [WARN] MLflow: {e}")

    print(f"\n  Tayyor! {datetime.now().strftime('%H:%M:%S')}")
    print("="*52)
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()
