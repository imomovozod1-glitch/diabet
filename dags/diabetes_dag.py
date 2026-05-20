"""
Diabetes Risk Pipeline — Apache Airflow DAG
Har 1 soatda:
  1. Yangi bemorlar generatsiya qilinadi
  2. Risk modeli ishga tushadi
  3. Natijalar PostgreSQL ga yoziladi
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import sys

# tasks papkasini Python path ga qo'shish
sys.path.insert(0, '/opt/airflow/tasks')

# ─── DB sozlamalari ───────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     os.getenv("DIABETES_DB_HOST",     "postgres"),
    "port":     int(os.getenv("DIABETES_DB_PORT", "5432")),
    "dbname":   os.getenv("DIABETES_DB_NAME",     "diabetes_risk"),
    "user":     os.getenv("DIABETES_DB_USER",     "postgres"),
    "password": os.getenv("DIABETES_DB_PASSWORD", "postgres"),
}

# Har soatda nechta yangi bemor generatsiya qilinadi
PATIENTS_PER_RUN = 1

# ─── Default argumentlar ──────────────────────────────────────────────────────

default_args = {
    "owner":            "diabetes_team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ─── DAG ─────────────────────────────────────────────────────────────────────

dag = DAG(
    dag_id="diabetes_pipeline",
    default_args=default_args,
    description="Har soatda yangi bemorlar generatsiya + risk baholash",
    schedule_interval="0 * * * *",   # har soat boshida
    start_date=days_ago(1),
    catchup=False,
    tags=["diabetes", "ml", "pipeline"],
    max_active_runs=1,               # bir vaqtda faqat 1 ta run
)

# ─── Task 1: Bemorlar generatsiya ────────────────────────────────────────────

def task_generate_patients(**context):
    """
    Har soatda PATIENTS_PER_RUN ta yangi bemor generatsiya qiladi.
    XCom orqali patient_id larni keyingi taskga uzatadi.
    """
    import psycopg2
    import random
    import numpy as np
    from datetime import date, timedelta

    rng = np.random.default_rng()

    UZ_FIRST_MALE = [
        "Akbar","Alisher","Aziz","Bobur","Doniyor","Eldor","Farrux","Hamza",
        "Ibrohim","Jamshid","Kamol","Laziz","Mirzo","Nodir","Otabek","Sardor",
        "Temur","Ulugbek","Ravshan","Jasur","Bekzod","Davron","Firdavs","Sanjar",
    ]
    UZ_FIRST_FEMALE = [
        "Aziza","Barno","Dilorom","Feruza","Gulnora","Iroda","Kamola","Lobar",
        "Malika","Nafisa","Oydin","Sarvinoz","Umida","Yulduz","Zilola","Sevinch",
        "Shahlo","Mohira","Nozima","Dildora","Gavhar","Zuhra","Charos","Bahora",
    ]
    UZ_LAST = [
        "Karimov","Rahimov","Toshmatov","Yusupov","Xasanov","Mirzayev","Qodirov",
        "Abdullayev","Normatov","Holmatov","Ergashev","Ibragimov","Nazarov",
        "Sultonov","Turgunov","Umarov","Valiyev","Zokirov","Askarov","Mamatov",
    ]
    REGIONS = [
        "Toshkent","Samarqand","Farg'ona","Namangan","Andijon",
        "Buxoro","Xorazm","Qashqadaryo","Surxondaryo","Navoiy",
    ]

    CLASS_WEIGHTS = {"healthy": 0.877, "prediabetes": 0.044, "diabetes": 0.079}
    classes = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Doctor ID larini olish
    cur.execute("SELECT id FROM doctors WHERE is_active = TRUE")
    doctor_ids = [r[0] for r in cur.fetchall()]

    LAB_NAMES = [
        "Toshkent Tibbiyot Laboratoriyasi","MDC Lab","Invitro Uzbekistan",
        "Anor Lab","Shifa Lab","Medtest","Genotek Laboratoriya",
    ]

    patient_ids = []

    for _ in range(PATIENTS_PER_RUN):
        gender  = "male" if rng.random() < 0.50 else "female"
        age     = int(rng.integers(20, 80))
        dob     = date.today() - timedelta(days=age * 365)
        fname   = random.choice(UZ_FIRST_MALE if gender == "male" else UZ_FIRST_FEMALE)
        lname_base = random.choice(UZ_LAST)
        lname   = lname_base if gender == "male" else lname_base + "a"
        region  = random.choice(REGIONS)
        phone   = f"+998{rng.integers(90,99)}{rng.integers(1000000,9999999)}"
        email   = f"{fname.lower()}.{lname.lower()}{rng.integers(10,99)}@mail.uz"

        cur.execute("""
            INSERT INTO patients
              (first_name, last_name, date_of_birth, gender, phone, email, address)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (fname, lname, dob, gender, phone, email, f"{region} viloyati"))

        patient_id = cur.fetchone()[0]
        patient_ids.append(str(patient_id))

        # Medical test
        doctor_id = random.choice(doctor_ids) if doctor_ids else None
        cur.execute("""
            INSERT INTO medical_tests
              (patient_id, doctor_id, lab_name, test_date, status)
            VALUES (%s,%s,%s,NOW(),'completed')
            RETURNING id
        """, (patient_id, doctor_id, random.choice(LAB_NAMES)))

        test_id = cur.fetchone()[0]

        # Sinf tanlash
        status = random.choices(classes, weights=weights)[0]

        # BMI
        obese_prob = {"healthy": 0.34, "prediabetes": 0.55, "diabetes": 0.63}
        if rng.random() < obese_prob[status]:
            bmi = float(np.clip(rng.normal(33, 4), 30, 50))
        elif rng.random() < 0.35:
            bmi = float(np.clip(rng.normal(27, 1.5), 25, 29.9))
        else:
            bmi = float(np.clip(rng.normal(22, 2), 17, 24.9))

        # Biomarkerlar
        af = max(0, (age - 40) * 0.01)
        bf = max(0, (bmi - 25) * 0.05)

        def n(mu, sig, lo, hi):
            return float(np.clip(rng.normal(mu, sig), lo, hi))

        if status == "healthy":
            bmarks = {
                "HBA1C": n(5.1,0.35,4.0,5.6), "FBG": n(88,8,70,99),
                "PPG": n(110,15,80,139),       "FINS": n(10,4,2.6,20),
                "HOMAIR": n(1.5,0.6,0.3,2.5), "BMI": round(bmi,1),
                "WAIST": n(82,8,65,93) if gender=="male" else n(75,7,58,87),
                "SBP": n(115,10,90,119),       "DBP": n(72,7,60,79),
                "TCHOL": n(175,25,140,199),    "HDL": n(55,10,40,90),
                "LDL": n(105,22,60,129),       "TG": n(110,30,50,149),
                "CRP": n(0.5,0.3,0.01,0.99),  "CREAT": n(0.9,0.15,0.6,1.19),
                "EGFR": n(90,12,65,120),       "ADIPO": n(14,4,6,28),
            }
        elif status == "prediabetes":
            bmarks = {
                "HBA1C": n(6.0+af+bf,0.22,5.7,6.4), "FBG": n(108,8,100,125),
                "PPG": n(155,15,140,199),             "FINS": n(18,5,8,35),
                "HOMAIR": n(3.0,0.7,2.5,4.5),        "BMI": round(bmi,1),
                "WAIST": n(96,8,80,109) if gender=="male" else n(88,8,74,100),
                "SBP": n(128,10,115,139),             "DBP": n(80,8,75,89),
                "TCHOL": n(210,28,170,239),           "HDL": n(46,9,35,65),
                "LDL": n(128,25,90,159),              "TG": n(165,40,100,249),
                "CRP": n(1.8,0.8,0.5,4.9),           "CREAT": n(1.0,0.18,0.7,1.3),
                "EGFR": n(80,14,55,100),              "ADIPO": n(9,3,4,16),
            }
        else:
            sv = rng.uniform(0, 1)
            bmarks = {
                "HBA1C": n(8.5+sv*2+af+bf,0.8,6.5,14.0), "FBG": n(180+sv*80,30,126,400),
                "PPG": n(280+sv*60,40,200,500),            "FINS": n(8,5,1,30),
                "HOMAIR": n(5.5,1.5,2.5,12),              "BMI": round(bmi,1),
                "WAIST": n(104,10,88,130) if gender=="male" else n(96,9,82,120),
                "SBP": n(142,14,120,180),                 "DBP": n(88,10,80,110),
                "TCHOL": n(235,35,180,320),               "HDL": n(40,9,28,55),
                "LDL": n(145,32,90,220),                  "TG": n(230,70,150,500),
                "CRP": n(4.0,2.0,1.0,12),                "CREAT": n(1.15,0.3,0.8,2.5),
                "EGFR": n(65,18,30,90),                   "ADIPO": n(6,2.5,2,12),
            }

        bmarks["TGHDL"] = round(bmarks["TG"] / bmarks["HDL"], 3)

        # Biomarker ID larini olish
        cur.execute("SELECT code, id FROM biomarkers WHERE is_active = TRUE")
        biomarker_map = {row[0]: row[1] for row in cur.fetchall()}

        NORMALS = {
            "HBA1C":(4.0,5.6),"FBG":(70,100),"PPG":(70,140),
            "FINS":(2.6,24.9),"HOMAIR":(0,2.5),"BMI":(18.5,24.9),
            "WAIST":(0,94),"SBP":(90,120),"DBP":(60,80),
            "TCHOL":(0,200),"HDL":(40,999),"LDL":(0,100),
            "TG":(0,150),"TGHDL":(0,2.0),"CRP":(0,1.0),
            "CREAT":(0.6,1.2),"EGFR":(60,999),"ADIPO":(5,30),
        }

        for code, val in bmarks.items():
            bm_id = biomarker_map.get(code)
            if not bm_id:
                continue
            lo, hi = NORMALS.get(code, (None, None))
            if lo is None:
                st = "normal"
            elif val < lo * 0.7:
                st = "critical_low"
            elif val < lo:
                st = "low"
            elif val > hi * 1.5:
                st = "critical_high"
            elif val > hi:
                st = "high"
            else:
                st = "normal"

            cur.execute("""
                INSERT INTO test_results (medical_test_id, biomarker_id, value, status)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (medical_test_id, biomarker_id) DO NOTHING
            """, (test_id, bm_id, round(val, 4), st))

        conn.commit()

    cur.close()
    conn.close()

    print(f"[OK] {PATIENTS_PER_RUN} ta yangi bemor yaratildi")

    # XCom orqali patient_id larni uzatish
    context["ti"].xcom_push(key="patient_ids", value=patient_ids)
    return patient_ids


# ─── Task 2: Risk modeli ──────────────────────────────────────────────────────

def task_run_risk_model(**context):
    """
    Yangi bemorlar uchun risk darajasini hisoblaydi.
    HbA1c asosida rule-based model (train_model.py tayyor bo'lguncha).
    """
    import psycopg2
    from datetime import datetime

    ti          = context["ti"]
    patient_ids = ti.xcom_pull(task_ids="generate_patients", key="patient_ids")

    if not patient_ids:
        print("[WARN] Patient ID lar topilmadi")
        return []

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Aktiv model
    cur.execute("SELECT id FROM ml_models WHERE is_active = TRUE LIMIT 1")
    row = cur.fetchone()
    if not row:
        raise ValueError("Aktiv ML model topilmadi!")
    model_id = row[0]

    risk_ids = []

    for patient_id in patient_ids:
        # HbA1c qiymatini olish
        cur.execute("""
            SELECT tr.value, mt.id
            FROM test_results tr
            JOIN biomarkers b ON b.id = tr.biomarker_id
            JOIN medical_tests mt ON mt.id = tr.medical_test_id
            WHERE mt.patient_id = %s AND b.code = 'HBA1C'
            ORDER BY mt.test_date DESC
            LIMIT 1
        """, (patient_id,))
        row = cur.fetchone()
        if not row:
            continue

        hba1c, test_id = row
        hba1c = float(hba1c)

        # Rule-based risk hisoblash
        if hba1c < 5.7:
            risk_score = round(0.02 + (hba1c - 4.0) * 0.05, 4)
            risk_level = "very_low" if risk_score < 0.08 else "low"
            rec = "Yiliga 1 marta profilaktik tekshiruv tavsiya etiladi."
        elif hba1c < 6.5:
            risk_score = round(0.35 + (hba1c - 5.7) * 0.30, 4)
            risk_level = "moderate"
            rec = "Turmush tarzini o'zgartirish va 3 oyda 1 marta tekshiruv."
        elif hba1c < 8.0:
            risk_score = round(0.65 + (hba1c - 6.5) * 0.12, 4)
            risk_level = "high"
            rec = "Shifokor bilan maslahatlashish va dori-darmon ko'rib chiqish."
        else:
            risk_score = round(min(0.83 + (hba1c - 8.0) * 0.03, 0.98), 4)
            risk_level = "very_high"
            rec = "Zudlik bilan endokrinolog ko'rigi va davolanish zarur."

        # Risk assessment yozish
        cur.execute("""
            INSERT INTO risk_assessments
              (patient_id, model_id, medical_test_id,
               risk_score, risk_level, recommendation, assessed_by)
            VALUES (%s,%s,%s,%s,%s,%s,'airflow_dag')
            RETURNING id
        """, (patient_id, model_id, test_id,
              risk_score, risk_level, rec))

        risk_id = cur.fetchone()[0]
        risk_ids.append(str(risk_id))

        # Yuqori xavf uchun notification
        if risk_level in ("high", "very_high"):
            cur.execute("""
                INSERT INTO notifications
                  (patient_id, risk_assessment_id, type, message)
                VALUES (%s,%s,'high_risk_alert',%s)
            """, (patient_id, risk_id,
                  f"Diabet xavfi {risk_level.replace('_',' ')} darajada aniqlandi. "
                  f"Shifokorga murojaat qiling."))

    conn.commit()
    cur.close()
    conn.close()

    print(f"[OK] {len(risk_ids)} ta risk baholash yakunlandi")
    context["ti"].xcom_push(key="risk_ids", value=risk_ids)
    return risk_ids


# ─── Task 3: Statistika log ───────────────────────────────────────────────────

def task_log_statistics(**context):
    """
    Har run dan keyin umumiy statistikani log ga yozadi.
    Monitoring va debugging uchun.
    """
    import psycopg2

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM patients")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM risk_assessments")
    total_risks = cur.fetchone()[0]

    cur.execute("""
        SELECT risk_level, COUNT(*)
        FROM risk_assessments
        GROUP BY risk_level
        ORDER BY COUNT(*) DESC
    """)
    risk_dist = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*) FROM notifications
        WHERE is_sent = FALSE
    """)
    unsent_notifs = cur.fetchone()[0]

    cur.close()
    conn.close()

    print("=" * 50)
    print(f"  Run vaqti    : {context['execution_date']}")
    print(f"  Jami bemorlar: {total_patients:,}")
    print(f"  Jami risk    : {total_risks:,}")
    print(f"  Yuborilmagan : {unsent_notifs} notification")
    print("  Risk taqsimoti:")
    for level, count in risk_dist:
        print(f"    {level:<12} {count:>6,}")
    print("=" * 50)


# ─── Tasklar ─────────────────────────────────────────────────────────────────

t1 = PythonOperator(
    task_id="generate_patients",
    python_callable=task_generate_patients,
    dag=dag,
)

t2 = PythonOperator(
    task_id="run_risk_model",
    python_callable=task_run_risk_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id="log_statistics",
    python_callable=task_log_statistics,
    dag=dag,
)

# ─── Task ketma-ketligi ───────────────────────────────────────────────────────

t1 >> t2 >> t3
