"""
Diabetes Risk Assessment — ML Dataset Generator
O'zbekiston statistikasi asosida 50,000 ta realistik bemor ma'lumoti.

O'rnatish:
    pip install faker psycopg2-binary tqdm numpy

Ishlatish:
    python generate_dataset.py

Sozlamalar (.env yoki pastdagi DB_CONFIG):
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import sys
import random
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from faker import Faker
from tqdm import tqdm
from datetime import datetime, date, timedelta

# ─── Sozlamalar ───────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     "postgres",    # localhost → postgres
    "port":     5432,
    "dbname":   "diabetes_risk",
    "user":     "postgres",
    "password": "postgres",    # 1234 → postgres
}

TOTAL_PATIENTS   = 50_000
START_DATE       = date(2020, 1, 1)
END_DATE         = date(2026, 12, 31)
BATCH_SIZE       = 500        # har safar nechta yozuv DB ga yuboriladi
RANDOM_SEED      = 42

# O'zbekiston statistikasi (2021 World Bank / klinik tadqiqotlar)
CLASS_WEIGHTS = {
    "healthy":    0.877,
    "prediabetes": 0.044,
    "diabetes":   0.079,
}

fake = Faker(["uz_UZ", "ru_RU"])
rng  = np.random.default_rng(RANDOM_SEED)

# ─── O'zbek ismlari (Faker uz_UZ yetarli emas, qo'shimcha) ───────────────────

UZ_FIRST_MALE = [
    "Akbar","Alisher","Aziz","Bobur","Doniyor","Eldor","Farrux","Hamza",
    "Ibrohim","Jamshid","Kamol","Laziz","Mirzo","Nodir","Otabek","Pulat",
    "Ravshan","Sardor","Temur","Ulugbek","Vohid","Xurshid","Yorqin","Zafar",
    "Abror","Bekzod","Davron","Erkin","Firdavs","Husan","Jasur","Komil",
    "Lochinbek","Mansur","Nurbek","Oybek","Parviz","Rustam","Sanjar","Tohir",
]
UZ_FIRST_FEMALE = [
    "Aziza","Barno","Dilorom","Feruza","Gulnora","Hulkar","Iroda","Kamola",
    "Lobar","Malika","Nafisa","Oydin","Parizod","Rohila","Sarvinoz","Tabassum",
    "Umida","Venera","Xurmo","Yulduz","Zilola","Adolat","Bahora","Charos",
    "Dildora","Gavhar","Hamida","Mohira","Nozima","Sevinch","Shahlo","Zuhra",
]
UZ_LAST = [
    "Karimov","Rahimov","Toshmatov","Yusupov","Xasanov","Mirzayev","Qodirov",
    "Abdullayev","Normatov","Holmatov","Ergashev","Ibragimov","Nazarov",
    "Sultonov","Turgunov","Umarov","Valiyev","Xoliqov","Zokirov","Askarov",
    "Baxtiyorov","Davlatov","Eshmatov","Fayzullayev","G'aniyev","Haydarov",
    "Ismoilov","Jumayev","Latipov","Mamatov","Ortiqov","Pulatov","Rajabov",
    "Sobirov","To'rayev","Usmonov","Yunusov","Ziyodullayev","Olimov","Qosimov",
]

REGIONS = [
    "Toshkent","Samarqand","Farg'ona","Namangan","Andijon","Buxoro",
    "Xorazm","Qashqadaryo","Surxondaryo","Sirdaryo","Jizzax","Navoiy",
    "Qoraqalpog'iston",
]

LAB_NAMES = [
    "Toshkent Tibbiyot Laboratoriyasi","MDC Lab","Invitro Uzbekistan",
    "Anor Lab","Genotek Laboratoriya","Shifa Lab","Medtest",
    "Respublika Endokrinologiya Markazi","Poliklinika №1","Poliklinika №3",
]

DOCTOR_IDS   = []   # DB dan olinadi
BIOMARKER_IDS = {}  # code → uuid
MODEL_ID      = None


# ─── Realistik biomarker generatsiyasi ───────────────────────────────────────

def _norm(mu, sigma, lo, hi):
    """Kesib qo'yilgan normal taqsimot."""
    v = rng.normal(mu, sigma)
    return float(np.clip(v, lo, hi))


def result_status(code: str, value: float) -> str:
    """Qiymat normal oralig'ida yoki undan chiqganini aniqlaydi."""
    NORMALS = {
        "HBA1C":  (4.0, 5.6),  "FBG":   (70,  100), "PPG":   (70,  140),
        "FINS":   (2.6, 24.9), "HOMAIR":(0,   2.5),  "BMI":   (18.5,24.9),
        "WAIST":  (0,   94),   "SBP":   (90,  120),  "DBP":   (60,  80),
        "TCHOL":  (0,   200),  "HDL":   (40,  999),  "LDL":   (0,   100),
        "TG":     (0,   150),  "TGHDL": (0,   2.0),  "CRP":   (0,   1.0),
        "CREAT":  (0.6, 1.2),  "EGFR":  (60,  999),  "ADIPO": (5,   30),
    }
    lo, hi = NORMALS.get(code, (None, None))
    if lo is None:
        return "normal"
    if value < lo:
        return "critical_low" if value < lo * 0.7 else "low"
    if value > hi:
        return "critical_high" if value > hi * 1.5 else "high"
    return "normal"


def generate_biomarkers(status: str, age: int, gender: str, bmi: float) -> dict:
    """
    Sinf (healthy/prediabetes/diabetes), yosh, jins va BMI ga qarab
    klinik jihatdan izchil biomarker qiymatlari chiqaradi.

    Manbalar:
      - ADA Standards of Medical Care in Diabetes (2024)
      - WHO diagnostic criteria
      - O'zbekiston klinik tadqiqotlari (Ismailov et al.)
    """
    age_factor = (age - 40) * 0.01   # yosh oshgani sayin xavf ortadi
    bmi_factor = max(0, (bmi - 25) * 0.05)
    male_factor = 0.05 if gender == "male" else 0.0

    if status == "healthy":
        hba1c     = _norm(5.1, 0.35, 4.0,  5.6)
        glucose   = _norm(88,  8.0,  70,   99)
        ppg       = _norm(110, 15,   80,   139)
        insulin   = _norm(10,  4.0,  2.6,  20)
        homa_ir   = _norm(1.5, 0.6,  0.3,  2.5)
        hdl       = _norm(55,  10,   40,   90)   if gender=="male" else _norm(62, 10, 45, 95)
        ldl       = _norm(105, 22,   60,   129)
        tg        = _norm(110, 30,   50,   149)
        tchol     = _norm(175, 25,   140,  199)
        crp       = _norm(0.5, 0.3,  0.01, 0.99)
        creat     = _norm(0.9, 0.15, 0.6,  1.19) if gender=="male" else _norm(0.75,0.12,0.5,0.99)
        egfr      = _norm(90,  12,   65,   120)
        sbp       = _norm(115, 10,   90,   119)
        dbp       = _norm(72,  7,    60,   79)
        adipo     = _norm(14,  4,    6,    28)
        waist     = _norm(82,  8,    65,   93)  if gender=="male" else _norm(75,7,58,87)

    elif status == "prediabetes":
        hba1c     = _norm(6.0 + age_factor + bmi_factor, 0.22, 5.7,  6.4)
        glucose   = _norm(108, 8.0,  100,  125)
        ppg       = _norm(155, 15,   140,  199)
        insulin   = _norm(18,  5.0,  8,    35)
        homa_ir   = _norm(3.0, 0.7,  2.5,  4.5)
        hdl       = _norm(46,  9,    35,   65)   if gender=="male" else _norm(52, 9, 38, 70)
        ldl       = _norm(128, 25,   90,   159)
        tg        = _norm(165, 40,   100,  249)
        tchol     = _norm(210, 28,   170,  239)
        crp       = _norm(1.8, 0.8,  0.5,  4.9)
        creat     = _norm(1.0, 0.18, 0.7,  1.3)  if gender=="male" else _norm(0.85,0.15,0.6,1.1)
        egfr      = _norm(80,  14,   55,   100)
        sbp       = _norm(128, 10,   115,  139)
        dbp       = _norm(80,  8,    75,   89)
        adipo     = _norm(9,   3,    4,    16)
        waist     = _norm(96,  8,    80,   109) if gender=="male" else _norm(88,8,74,100)

    else:  # diabetes
        severity  = rng.uniform(0, 1)   # 0=yengil, 1=og'ir
        hba1c     = _norm(8.5 + severity * 2 + age_factor + bmi_factor, 0.8, 6.5, 14.0)
        glucose   = _norm(180 + severity * 80, 30, 126, 400)
        ppg       = _norm(280 + severity * 60, 40, 200, 500)
        insulin   = _norm(8,   5.0,  1,    30)   # insulin qarshiligi tufayli past
        homa_ir   = _norm(5.5, 1.5,  2.5,  12)
        hdl       = _norm(40,  9,    28,   55)   if gender=="male" else _norm(46, 8, 32, 60)
        ldl       = _norm(145, 32,   90,   220)
        tg        = _norm(230, 70,   150,  500)
        tchol     = _norm(235, 35,   180,  320)
        crp       = _norm(4.0, 2.0,  1.0,  12)
        creat     = _norm(1.15,0.3,  0.8,  2.5)  if gender=="male" else _norm(0.95,0.25,0.7,2.0)
        egfr      = _norm(65,  18,   30,   90)
        sbp       = _norm(142, 14,   120,  180)
        dbp       = _norm(88,  10,   80,   110)
        adipo     = _norm(6,   2.5,  2,    12)
        waist     = _norm(104, 10,   88,   130) if gender=="male" else _norm(96,9,82,120)

    tg_hdl = round(tg / hdl, 3)

    bmarks = {
        "HBA1C": round(hba1c, 2),
        "FBG": round(glucose, 1),
        "PPG": round(ppg, 1),
        "FINS": round(insulin, 2),
        "HOMAIR": round(homa_ir, 3),
        "BMI": round(bmi, 1),
        "WAIST": round(waist, 1),
        "SBP": round(sbp, 1),
        "DBP": round(dbp, 1),
        "TCHOL": round(tchol, 1),
        "HDL": round(hdl, 1),
        "LDL": round(ldl, 1),
        "TG": round(tg, 1),
        "TGHDL": round(tg_hdl, 3),
        "CRP": round(crp, 3),
        "CREAT": round(creat, 3),
        "EGFR": round(egfr, 1),
        "ADIPO": round(adipo, 2),
    }

    # Shovqin qo'shish
    noise_pct = {
        "HBA1C": 0.08, "FBG": 0.10, "PPG": 0.12,
        "FINS": 0.15, "HOMAIR": 0.18, "BMI": 0.04,
        "WAIST": 0.05, "SBP": 0.06, "DBP": 0.06,
        "TCHOL": 0.08, "HDL": 0.10, "LDL": 0.10,
        "TG": 0.12, "TGHDL": 0.12, "CRP": 0.20,
        "CREAT": 0.08, "EGFR": 0.10, "ADIPO": 0.15,
    }
    for code in list(bmarks.keys()):
        pct = noise_pct.get(code, 0.10)
        bmarks[code] = round(bmarks[code] * (1 + rng.uniform(-pct, pct)), 2)

    return bmarks


def risk_level_from_status(status: str, bmarks: dict) -> tuple[str, float]:
    score = 0.0

    homa = bmarks.get("HOMAIR", 1.5)
    if homa >= 4.0:   score += 0.20
    elif homa >= 2.5: score += 0.10

    crp = bmarks.get("CRP", 0.5)
    if crp >= 3.0:   score += 0.12
    elif crp >= 1.0: score += 0.06

    sbp = bmarks.get("SBP", 115)
    if sbp >= 140:   score += 0.12
    elif sbp >= 130: score += 0.06

    bmi = bmarks.get("BMI", 22)
    if bmi >= 30:    score += 0.10
    elif bmi >= 25:  score += 0.05

    tg = bmarks.get("TG", 110)
    if tg >= 200:    score += 0.08
    elif tg >= 150:  score += 0.04

    fins = bmarks.get("FINS", 10)
    if fins >= 25:   score += 0.06
    elif fins >= 15: score += 0.03

    egfr = bmarks.get("EGFR", 90)
    if egfr < 45:    score += 0.08
    elif egfr < 60:  score += 0.04

    # Base — kichikroq, ko'proq overlap
    base = {"healthy": 0.05, "prediabetes": 0.20, "diabetes": 0.35}
    score += base[status]

    # Kuchli shovqin — labellar aralashsin
    score += float(rng.uniform(-0.20, 0.20))
    score = float(np.clip(score, 0.01, 0.98))

    if score < 0.25:   level = "very_low"
    elif score < 0.45: level = "low"
    elif score < 0.62: level = "moderate"
    elif score < 0.78: level = "high"
    else:              level = "very_high"

    return level, round(score, 4)


def risk_level_from_hba1c(hba1c: float) -> tuple[str, float]:
    """HbA1c asosida risk darajasi va skori."""
    if hba1c < 5.7:
        score = round(rng.uniform(0.02, 0.15), 4)
        level = "very_low" if score < 0.08 else "low"
    elif hba1c < 6.5:
        score = round(rng.uniform(0.35, 0.60), 4)
        level = "moderate"
    elif hba1c < 8.0:
        score = round(rng.uniform(0.65, 0.82), 4)
        level = "high"
    else:
        score = round(rng.uniform(0.83, 0.98), 4)
        level = "very_high"
    return level, score


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta)))


def generate_bmi(status: str) -> float:
    """Klinik tadqiqot nisbatlari asosida BMI generatsiya."""
    # yangi diabet: 63% semiz, prediabet: 55%, sog'lom: 34%
    obese_prob = {"healthy": 0.34, "prediabetes": 0.55, "diabetes": 0.63}
    if rng.random() < obese_prob[status]:
        return _norm(33, 4, 30, 50)   # semiz
    elif rng.random() < 0.35:
        return _norm(27, 1.5, 25, 29.9)  # ortiqcha vazn
    else:
        return _norm(22, 2, 17, 24.9)    # normal


# ─── DB yordamchi funksiyalar ─────────────────────────────────────────────────

def fetch_ids(conn):
    """Mavjud doctor, biomarker va model ID larini oladi."""
    global DOCTOR_IDS, BIOMARKER_IDS, MODEL_ID
    cur = conn.cursor()

    cur.execute("SELECT id FROM doctors WHERE is_active = TRUE")
    DOCTOR_IDS = [r[0] for r in cur.fetchall()]
    if not DOCTOR_IDS:
        raise RuntimeError(
            "Doctors jadvali bo'sh. Avval diabetes_db_migration.py ni ishga tushiring."
        )

    cur.execute("SELECT code, id FROM biomarkers WHERE is_active = TRUE")
    BIOMARKER_IDS = {row[0]: row[1] for row in cur.fetchall()}
    if not BIOMARKER_IDS:
        raise RuntimeError("Biomarkers jadvali bo'sh.")

    cur.execute("SELECT id FROM ml_models WHERE is_active = TRUE LIMIT 1")
    row = cur.fetchone()
    if not row:
        raise RuntimeError("Aktiv ML model topilmadi.")
    MODEL_ID = row[0]
    cur.close()


def insert_batch(conn, table: str, cols: list, rows: list):
    if not rows:
        return
    cur = conn.cursor()
    query = f"INSERT INTO {table} ({','.join(cols)}) VALUES %s ON CONFLICT DO NOTHING"
    execute_values(cur, query, rows, page_size=BATCH_SIZE)
    conn.commit()
    cur.close()


# ─── Asosiy generatsiya ───────────────────────────────────────────────────────

def generate_all(conn):
    classes = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())

    # Har bir sinf uchun aniq sonlar
    counts = {
        "healthy":     int(TOTAL_PATIENTS * CLASS_WEIGHTS["healthy"]),
        "prediabetes": int(TOTAL_PATIENTS * CLASS_WEIGHTS["prediabetes"]),
        "diabetes":    TOTAL_PATIENTS
                       - int(TOTAL_PATIENTS * CLASS_WEIGHTS["healthy"])
                       - int(TOTAL_PATIENTS * CLASS_WEIGHTS["prediabetes"]),
    }

    # Status ro'yxati
    statuses = (
        ["healthy"]     * counts["healthy"] +
        ["prediabetes"] * counts["prediabetes"] +
        ["diabetes"]    * counts["diabetes"]
    )
    random.shuffle(statuses)

    print(f"\n  Sinf taqsimoti:")
    for k, v in counts.items():
        print(f"    {k:<14} {v:>6} ta ({v/TOTAL_PATIENTS*100:.1f}%)")
    print()

    # ── Batchlarga bo'lib yozish ──────────────────────────────────────────────
    pat_rows,  test_rows,  res_rows  = [], [], []
    risk_rows, notif_rows            = [], []

    pbar = tqdm(total=TOTAL_PATIENTS, unit="bemor", ncols=72,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i, status in enumerate(statuses):

        # ── Bemor ────────────────────────────────────────────────────────────
        gender  = "male" if rng.random() < 0.50 else "female"
        age     = int(rng.integers(20, 80))
        dob     = date.today().replace(year=date.today().year - age)
        fname   = random.choice(UZ_FIRST_MALE if gender=="male" else UZ_FIRST_FEMALE)
        lname_base = random.choice(UZ_LAST)
        lname   = lname_base if gender == "male" else lname_base + "a"
        region  = random.choice(REGIONS)
        email   = f"{fname.lower()}.{lname.lower()}{rng.integers(10,99)}@mail.uz"
        phone   = f"+998{rng.integers(90,99)}{rng.integers(1000000,9999999)}"

        pat_id = f"pat-{i:07d}"   # UUID o'rniga placeholder (DB gen_random_uuid ishlatadi)

        pat_rows.append((
            fname, lname, dob, gender, phone, email,
            f"{region} viloyati", datetime.now(), datetime.now()
        ))

        # ── Tekshiruv ─────────────────────────────────────────────────────────
        test_date  = random_date(START_DATE, END_DATE)
        doctor_id  = random.choice(DOCTOR_IDS)
        lab        = random.choice(LAB_NAMES)

        test_rows.append((doctor_id, lab, test_date, "completed", datetime.now()))

        # ── Biomarkerlar ──────────────────────────────────────────────────────
        bmi    = generate_bmi(status)
        bmarks = generate_biomarkers(status, age, gender, bmi)

        for code, val in bmarks.items():
            bm_id = BIOMARKER_IDS.get(code)
            if bm_id:
                res_rows.append((bm_id, val, result_status(code, val)))

        # ── Risk baholash ─────────────────────────────────────────────────────
        rl, rs = risk_level_from_status(status, bmarks)
        rec = {
            "very_low": "Yiliga 1 marta profilaktik tekshiruv tavsiya etiladi.",
            "low":      "6 oyda 1 marta qon shakari nazorati.",
            "moderate": "Turmush tarzini o'zgartirish va 3 oyda 1 marta tekshiruv.",
            "high":     "Shifokor bilan maslahatlashish va dori-darmon ko'rib chiqish.",
            "very_high":"Zudlik bilan endokrinolog ko'rigi va davolanish zarur.",
        }[rl]

        risk_rows.append((MODEL_ID, rs, rl, rec, datetime.now()))

        # ── Bildirgi (faqat yuqori xavfda) ───────────────────────────────────
        if rl in ("high", "very_high"):
            notif_rows.append((
                "high_risk_alert",
                f"Diabet xavfi {rl.replace('_',' ')} darajada aniqlandi. Shifokorga murojaat qiling.",
                False, datetime.now()
            ))
        else:
            notif_rows.append(None)   # placeholder

        pbar.update(1)

        # ── Batch yozish ──────────────────────────────────────────────────────
        if len(pat_rows) >= BATCH_SIZE:
            flush_batch(conn, pat_rows, test_rows, res_rows, risk_rows, notif_rows)
            pat_rows, test_rows, res_rows, risk_rows, notif_rows = [], [], [], [], []

    # ── Qolganlarini yozish ───────────────────────────────────────────────────
    if pat_rows:
        flush_batch(conn, pat_rows, test_rows, res_rows, risk_rows, notif_rows)

    pbar.close()


def flush_batch(conn, pat_rows, test_rows, res_rows, risk_rows, notif_rows):
    """
    Bir batch ni tranzaksiyada yozadi.
    Har bir yozuvni bog'lash uchun avval patients, keyin qolganlar.
    """
    cur = conn.cursor()

    # 1. Patients → ID larni olamiz
    pat_ids = []
    for row in pat_rows:
        cur.execute("""
            INSERT INTO patients
              (first_name,last_name,date_of_birth,gender,phone,email,address,created_at,updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, row)
        pat_ids.append(cur.fetchone()[0])

    # 2. Medical tests → ID larni olamiz
    test_ids = []
    for pid, trow in zip(pat_ids, test_rows):
        doctor_id, lab, test_date, status, created_at = trow
        cur.execute("""
            INSERT INTO medical_tests
              (patient_id,doctor_id,lab_name,test_date,status,created_at,updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (pid, doctor_id, lab, test_date, status, created_at, created_at))
        test_ids.append(cur.fetchone()[0])

    # 3. Test results (batch)
    res_flat = []
    res_idx  = 0
    biomarker_count = len(BIOMARKER_IDS)
    for tid in test_ids:
        for _ in range(biomarker_count):
            if res_idx < len(res_rows):
                bm_id, val, st = res_rows[res_idx]
                res_flat.append((tid, bm_id, val, st))
                res_idx += 1

    execute_values(cur, """
        INSERT INTO test_results (medical_test_id, biomarker_id, value, status)
        VALUES %s
    """, res_flat, page_size=1000)

    # 4. Risk assessments → ID larni olamiz
    risk_ids = []
    for pid, tid, rrow in zip(pat_ids, test_ids, risk_rows):
        model_id, rs, rl, rec, assessed_at = rrow
        cur.execute("""
            INSERT INTO risk_assessments
              (patient_id,model_id,medical_test_id,risk_score,risk_level,recommendation,assessed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (pid, model_id, tid, rs, rl, rec, assessed_at))
        risk_ids.append(cur.fetchone()[0])

    # 5. Notifications (faqat xavf yuqori bo'lganlar)
    notif_flat = []
    for pid, rid, nrow in zip(pat_ids, risk_ids, notif_rows):
        if nrow is not None:
            ntype, msg, sent, created = nrow
            notif_flat.append((pid, rid, ntype, msg, sent, created))

    if notif_flat:
        execute_values(cur, """
            INSERT INTO notifications
              (patient_id,risk_assessment_id,type,message,is_sent,created_at)
            VALUES %s
        """, notif_flat, page_size=500)

    conn.commit()
    cur.close()

# ─── Xulosa statistikasi ──────────────────────────────────────────────────────

def print_stats(conn):
    cur = conn.cursor()
    print("\n" + "=" * 52)
    print("  Yakuniy statistika:")
    print("=" * 52)

    tables = [
        "patients", "medical_tests", "test_results",
        "risk_assessments", "notifications",
    ]
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        print(f"  {t:<25} {n:>8,} yozuv")

    print()
    cur.execute("""
        SELECT risk_level, COUNT(*) as cnt,
               ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 1) as pct
        FROM risk_assessments
        GROUP BY risk_level
        ORDER BY cnt DESC
    """)
    print("  Risk darajalari:")
    for rl, cnt, pct in cur.fetchall():
        print(f"    {rl:<15} {cnt:>7,}  ({pct}%)")

    print()
    cur.execute("""
        SELECT EXTRACT(YEAR FROM test_date)::int AS yr, COUNT(*)
        FROM medical_tests GROUP BY yr ORDER BY yr
    """)
    print("  Yillar bo'yicha:")
    for yr, cnt in cur.fetchall():
        print(f"    {yr}    {cnt:>7,} ta tekshiruv")

    cur.close()
    print("=" * 52)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  Diabetes Dataset Generator")
    print(f"  {TOTAL_PATIENTS:,} bemor | 2020–2026")
    print("=" * 52)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as e:
        print(f"\n[XATO] DB ga ulanib bo'lmadi: {e}")
        print("  Avval diabetes_db_migration.py ni ishga tushiring.")
        sys.exit(1)

    print("\n[1/4] Eski ma'lumotlar tozalanmoqda...")
    cur = conn.cursor()
    cur.execute("""
        TRUNCATE TABLE
            notifications,
            feature_contributions,
            model_run_logs,
            risk_assessments,
            test_results,
            medical_tests,
            family_history,
            patients
        RESTART IDENTITY CASCADE;
    """)
    conn.commit()
    cur.close()
    print("  [OK] Jadvallar tozalandi.")

    print("\n[2/4] ID lar yuklanmoqda...")
    fetch_ids(conn)
    print(f"  {len(DOCTOR_IDS)} ta shifokor, {len(BIOMARKER_IDS)} ta biomarker topildi.")

    print(f"\n[3/4] {TOTAL_PATIENTS:,} ta bemor generatsiya qilinmoqda...")
    t0 = datetime.now()
    generate_all(conn)
    elapsed = (datetime.now() - t0).seconds

    print(f"\n[4/4] Bajarildi — {elapsed} soniyada")
    print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()