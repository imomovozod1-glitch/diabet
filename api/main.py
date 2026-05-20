"""
Diabetes Risk API — FastAPI
Endpointlar:
  GET  /patients              — bemorlar ro'yxati (filter, pagination)
  GET  /patients/{id}         — bitta bemor + tahlillari
  GET  /risk/{patient_id}     — bemor risk tarixi
  POST /predict               — yangi bemor → risk bashorat
  GET  /dashboard/stats       — umumiy statistika
  GET  /dashboard/trend       — yillik trend
  GET  /health                — server holati

Ishlatish:
  pip install fastapi uvicorn psycopg2-binary sqlalchemy
  uvicorn main:app --reload --port 8000
"""

import os
import joblib
import numpy as np
from datetime import date, datetime
from typing import Optional
from uuid import UUID

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── ML Model yuklash ─────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
RISK_NAMES = {0: "healthy", 1: "prediabetes", 2: "diabetes"}

# Feature ustunlar — train_model.py dagi FEATURE_COLS bilan bir xil
ML_FEATURE_COLS = [
    "age", "gender_num", "family_history",
    "fasting_insulin", "homa_ir", "bmi", "waist_cm",
    "systolic_bp", "diastolic_bp", "total_cholesterol",
    "hdl", "ldl", "triglycerides", "tg_hdl_ratio",
    "crp", "creatinine", "egfr", "adiponectin",
]

# Feature nomlar — foydalanuvchiga ko'rsatish uchun
FEATURE_LABELS = {
    "age": "Yosh", "gender_num": "Jins", "family_history": "Oilaviy tarix",
    "fasting_insulin": "Insulin", "homa_ir": "HOMA-IR",
    "bmi": "BMI", "waist_cm": "Bel aylanasi",
    "systolic_bp": "Sistolik BP", "diastolic_bp": "Diastolik BP",
    "total_cholesterol": "Xolesterol", "hdl": "HDL",
    "ldl": "LDL", "triglycerides": "Trigliseridlar",
    "tg_hdl_ratio": "TG/HDL", "crp": "CRP",
    "creatinine": "Kreatinin", "egfr": "eGFR", "adiponectin": "Adiponektin",
}

# Median qiymatlar — yo'q fieldlar uchun
FEATURE_MEDIANS = {
    "age": 45, "gender_num": 0, "family_history": 0,
    "hba1c": 5.5, "fasting_glucose": 95.0, "postprandial_glucose": 120.0,
    "fasting_insulin": 10.0, "homa_ir": 1.5,
    "bmi": 24.0, "waist_cm": 82.0,
    "systolic_bp": 115.0, "diastolic_bp": 72.0,
    "total_cholesterol": 175.0, "hdl": 55.0,
    "ldl": 105.0, "triglycerides": 110.0,
    "tg_hdl_ratio": 2.0, "crp": 0.5,
    "creatinine": 0.9, "egfr": 90.0, "adiponectin": 14.0,
}

def load_ml_model():
    """Model.pkl ni yuklaydi. Yo'q bo'lsa None qaytaradi."""
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"[OK] ML model yuklandi: {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"[WARN] Model yuklanmadi: {e}")
    else:
        print(f"[WARN] model.pkl topilmadi: {MODEL_PATH}")
    return None

ML_MODEL = load_ml_model()

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Diabetes Risk API",
    description="Bemorlarning diabet xavfini baholash tizimi",
    version="1.0.0",
)

# React frontend uchun CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DB ulanish ───────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "diabetes_risk"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}


def get_conn():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=psycopg2.extras.RealDictCursor
    )


# ─── Pydantic sxemalar ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, example=45)
    gender: str = Field(..., example="male")
    bmi: float = Field(..., ge=10, le=70, example=28.5)
    homa_ir:          Optional[float] = Field(None, example=3.1)
    fasting_insulin:  Optional[float] = Field(None, example=18.0)
    waist_cm:         Optional[float] = Field(None, example=96.0)
    systolic_bp:      Optional[float] = Field(None, example=128)
    diastolic_bp:     Optional[float] = Field(None, example=82)
    total_cholesterol:Optional[float] = Field(None, example=210)
    hdl:              Optional[float] = Field(None, example=46)
    ldl:              Optional[float] = Field(None, example=128)
    triglycerides:    Optional[float] = Field(None, example=165)
    tg_hdl_ratio:     Optional[float] = Field(None, example=3.0)
    crp:              Optional[float] = Field(None, example=1.8)
    creatinine:       Optional[float] = Field(None, example=0.9)
    egfr:             Optional[float] = Field(None, example=85.0)
    adiponectin:      Optional[float] = Field(None, example=9.0)
    family_history:   Optional[bool]  = Field(None, example=False)


class PredictResponse(BaseModel):
    risk_score:     float
    risk_level:     str
    recommendation: str
    top_factors:    list


# ─── Yordamchi funksiyalar ────────────────────────────────────────────────────

# ─── ML bashorat funksiyasi ───────────────────────────────────────────────────

def calc_risk(data: PredictRequest) -> dict:
    """
    ML model yoki rule-based (model yo'q bo'lsa) risk hisoblash.
    """
    # Feature vektor yasash
    tg_hdl = (
        data.tg_hdl_ratio if data.tg_hdl_ratio
        else (data.triglycerides / data.hdl if data.triglycerides and data.hdl else None)
    )

    feature_values = {
        "age":               data.age,
        "gender_num":        1 if data.gender == "male" else 0,
        "family_history":    1 if data.family_history else 0,
        "fasting_insulin":   data.fasting_insulin,
        "homa_ir":           data.homa_ir,
        "bmi":               data.bmi,
        "waist_cm":          data.waist_cm,
        "systolic_bp":       data.systolic_bp,
        "diastolic_bp":      data.diastolic_bp,
        "total_cholesterol": data.total_cholesterol,
        "hdl":               data.hdl,
        "ldl":               data.ldl,
        "triglycerides":     data.triglycerides,
        "tg_hdl_ratio":      tg_hdl,
        "crp":               data.crp,
        "creatinine":        data.creatinine,
        "egfr":              data.egfr,
        "adiponectin":       data.adiponectin,
    }

    # Yo'q qiymatlarni median bilan to'ldirish
    X = [
        float(feature_values[col]) if feature_values[col] is not None
        else float(FEATURE_MEDIANS[col])
        for col in ML_FEATURE_COLS
    ]

    # ── ML model mavjud bo'lsa ────────────────────────────────────────────────
    if ML_MODEL is not None:
        X_arr   = np.array([X])
        pred    = int(ML_MODEL.predict(X_arr)[0])
        proba   = ML_MODEL.predict_proba(X_arr)[0]
        score   = float(proba[pred])
        level   = RISK_NAMES[pred]

        # Feature importance dan top omillar
        try:
            fi      = ML_MODEL.feature_importances_
            top_idx = np.argsort(fi)[::-1][:5]
            top_factors = [
                {
                    "name":   FEATURE_LABELS.get(ML_FEATURE_COLS[i], ML_FEATURE_COLS[i]),
                    "value":  round(X[i], 2),
                    "impact": "yuqori" if fi[i] > 0.1 else "o'rta" if fi[i] > 0.03 else "past",
                    "weight": round(float(fi[i]), 4),
                }
                for i in top_idx
            ]
        except Exception:
            top_factors = []

    # ── Model yo'q — rule-based ───────────────────────────────────────────────
    else:
        homa   = feature_values["homa_ir"] or FEATURE_MEDIANS["homa_ir"]
        bmi_v  = data.bmi
        crp_v  = feature_values["crp"] or FEATURE_MEDIANS["crp"]
        score  = min(0.98, round(
            (max(0, homa - 1.5) * 0.12) +
            (max(0, bmi_v - 25) * 0.015) +
            (max(0, crp_v - 1.0) * 0.04) +
            (0.05 if data.family_history else 0), 4
        ))
        if score < 0.15:   level = "healthy"
        elif score < 0.40: level = "prediabetes"
        else:              level = "diabetes"
        top_factors = []

    # Tavsiya
    rec_map = {
        "healthy": "Yiliga 1 marta profilaktik tekshiruv tavsiya etiladi.",
        "prediabetes": "Turmush tarzini o'zgartirish va 3 oyda 1 marta tekshiruv.",
        "diabetes": "Zudlik bilan endokrinolog ko'rigi va davolanish zarur.",
    }

    return {
        "risk_score":     round(score, 4),
        "risk_level":     level,
        "recommendation": rec_map.get(level, ""),
        "top_factors":    top_factors,
    }


# ─── Endpointlar ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Server va DB holati."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM patients")
        cnt = cur.fetchone()["cnt"]
        cur.close()
        conn.close()
        return {"status": "ok", "db": "connected", "total_patients": cnt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB xato: {str(e)}")


@app.get("/patients", tags=["Patients"])
def get_patients(
    page:       int = Query(1,    ge=1,    description="Sahifa raqami"),
    limit:      int = Query(20,   ge=1,    le=100, description="Har sahifada nechta"),
    gender:     Optional[str]   = Query(None, description="male | female"),
    risk_level: Optional[str]   = Query(None, description="very_low | low | moderate | high | very_high"),
    search:     Optional[str]   = Query(None, description="Ism yoki familiya"),
):
    """
    Bemorlar ro'yxati.
    Filter: jins, risk darajasi, ism qidirish.
    Pagination: page, limit.
    """
    offset = (page - 1) * limit
    where  = ["1=1"]
    params = []

    if gender:
        where.append("p.gender = %s")
        params.append(gender)

    if risk_level:
        where.append("ra.risk_level = %s")
        params.append(risk_level)

    if search:
        where.append("(p.first_name ILIKE %s OR p.last_name ILIKE %s)")
        params += [f"%{search}%", f"%{search}%"]

    where_sql = " AND ".join(where)

    query = f"""
        SELECT
            p.id,
            p.first_name,
            p.last_name,
            DATE_PART('year', AGE(p.date_of_birth))::int AS age,
            p.gender,
            p.phone,
            p.address,
            p.created_at,
            ra.risk_level,
            ra.risk_score,
            ra.assessed_at
        FROM patients p
        LEFT JOIN LATERAL (
            SELECT risk_level, risk_score, assessed_at
            FROM risk_assessments
            WHERE patient_id = p.id
            ORDER BY assessed_at DESC
            LIMIT 1
        ) ra ON TRUE
        WHERE {where_sql}
        ORDER BY p.created_at DESC
        LIMIT %s OFFSET %s
    """
    params += [limit, offset]

    # Jami soni
    count_query = f"""
        SELECT COUNT(*) as total
        FROM patients p
        LEFT JOIN LATERAL (
            SELECT risk_level FROM risk_assessments
            WHERE patient_id = p.id
            ORDER BY assessed_at DESC LIMIT 1
        ) ra ON TRUE
        WHERE {where_sql}
    """

    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(count_query, params[:-2])
        total = cur.fetchone()["total"]
        cur.execute(query, params)
        rows  = cur.fetchall()
        cur.close()
        conn.close()
        return {
            "total": total,
            "page":  page,
            "limit": limit,
            "pages": (total + limit - 1) // limit,
            "data":  [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}", tags=["Patients"])
def get_patient(patient_id: str):
    """Bitta bemor — shaxsiy ma'lumotlar + so'nggi tahlil natijalari."""
    try:
        conn = get_conn()
        cur  = conn.cursor()

        # Bemor
        cur.execute("""
            SELECT
                p.*,
                DATE_PART('year', AGE(p.date_of_birth))::int AS age
            FROM patients p WHERE p.id = %s
        """, (patient_id,))
        patient = cur.fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Bemor topilmadi")

        # So'nggi tahlil natijalari
        cur.execute("""
            SELECT
                b.name AS biomarker,
                b.code,
                b.unit,
                b.normal_min,
                b.normal_max,
                b.category,
                tr.value,
                tr.status,
                mt.test_date,
                mt.lab_name
            FROM test_results tr
            JOIN biomarkers   b  ON b.id  = tr.biomarker_id
            JOIN medical_tests mt ON mt.id = tr.medical_test_id
            WHERE mt.patient_id = %s
            ORDER BY mt.test_date DESC, b.category
        """, (patient_id,))
        results = cur.fetchall()

        # Risk tarixi
        cur.execute("""
            SELECT risk_level, risk_score, recommendation, assessed_at
            FROM risk_assessments
            WHERE patient_id = %s
            ORDER BY assessed_at DESC
            LIMIT 5
        """, (patient_id,))
        risk_history = cur.fetchall()

        # Oilaviy tarix
        cur.execute("""
            SELECT relation, has_diabetes, diabetes_type
            FROM family_history
            WHERE patient_id = %s
        """, (patient_id,))
        family = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "patient":      dict(patient),
            "test_results": [dict(r) for r in results],
            "risk_history": [dict(r) for r in risk_history],
            "family":       [dict(r) for r in family],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/{patient_id}", tags=["Risk"])
def get_risk(patient_id: str):
    """Bemor risk darajasi va tarixi."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                ra.risk_score,
                ra.risk_level,
                ra.recommendation,
                ra.assessed_at,
                mm.model_name,
                mm.version
            FROM risk_assessments ra
            JOIN ml_models mm ON mm.id = ra.model_id
            WHERE ra.patient_id = %s
            ORDER BY ra.assessed_at DESC
        """, (patient_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            raise HTTPException(status_code=404, detail="Risk ma'lumoti topilmadi")
        return {"patient_id": patient_id, "history": [dict(r) for r in rows]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
def predict(data: PredictRequest):
    """
    Yangi bemor ma'lumotlari asosida diabet xavfini bashorat qiladi.
    Minimal talab: age, gender, hba1c, fasting_glucose, bmi.
    """
    result = calc_risk(data)
    return PredictResponse(**result)


@app.get("/dashboard/stats", tags=["Dashboard"])
def dashboard_stats():
    """Frontend dashboard uchun umumiy statistika."""
    try:
        conn = get_conn()
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM patients")
        total_patients = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as total FROM risk_assessments")
        total_assessments = cur.fetchone()["total"]

        cur.execute("""
            SELECT risk_level, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM risk_assessments
            GROUP BY risk_level
            ORDER BY count DESC
        """)
        risk_dist = cur.fetchall()

        cur.execute("""
            SELECT COUNT(*) as total FROM notifications
            WHERE is_sent = FALSE
        """)
        unsent = cur.fetchone()["total"]

        cur.execute("""
            SELECT
                ROUND(AVG(CASE WHEN b.code='HBA1C' THEN tr.value END)::numeric, 2) AS avg_hba1c,
                ROUND(AVG(CASE WHEN b.code='FBG'   THEN tr.value END)::numeric, 1) AS avg_glucose,
                ROUND(AVG(CASE WHEN b.code='BMI'   THEN tr.value END)::numeric, 1) AS avg_bmi
            FROM test_results tr
            JOIN biomarkers b ON b.id = tr.biomarker_id
        """)
        avgs = cur.fetchone()

        cur.execute("""
            SELECT gender, COUNT(*) as count
            FROM patients
            GROUP BY gender
        """)
        gender_dist = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "total_patients":    total_patients,
            "total_assessments": total_assessments,
            "unsent_notifications": unsent,
            "risk_distribution": [dict(r) for r in risk_dist],
            "averages":          dict(avgs),
            "gender_distribution": [dict(r) for r in gender_dist],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/trend", tags=["Dashboard"])
def dashboard_trend():
    """Yillik trend — har yil nechta bemor, risk taqsimoti."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                EXTRACT(YEAR FROM mt.test_date)::int AS year,
                ra.risk_level,
                COUNT(*) AS count
            FROM medical_tests mt
            JOIN risk_assessments ra ON ra.medical_test_id = mt.id
            GROUP BY year, ra.risk_level
            ORDER BY year, ra.risk_level
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"trend": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
