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


def generate_biomarkers(status: str, gender: str, bmi: float) -> dict:
    """
    HbA1c/FBG/PPG — status bo'yicha (label manbai).
    Secondary featurelar — HbA1c dan olingan haqiqiy labelga mos generatsiya.
    """
    # ── Label manbai ──────────────────────────────────────────────────────────
    if status == "healthy":
        hba1c   = _norm(5.2, 0.55, 4.0,  7.2)
        glucose = _norm(90,  14,   64,  138)
        ppg     = _norm(112, 26,   68,  208)
    elif status == "prediabetes":
        hba1c   = _norm(6.1, 0.52, 5.2,  7.8)
        glucose = _norm(112, 14,   90,  158)
        ppg     = _norm(162, 30,  118,  258)
    else:  # diabetes
        hba1c   = _norm(8.2, 1.10, 6.3, 14.0)
        glucose = _norm(182, 40,  118,  400)
        ppg     = _norm(275, 55,  172,  500)

    # HbA1c dan haqiqiy labelni aniqlash — secondary featurelar shunga mos bo'lsin
    if hba1c < 5.7:
        eff = "healthy"
    elif hba1c < 6.5:
        eff = "prediabetes"
    else:
        eff = "diabetes"

    # 20% ehtimollik bilan boshqa klass secondary featurelari (realistik overlap)
    if rng.random() < 0.10:
        other = [c for c in ["healthy", "prediabetes", "diabetes"] if c != eff]
        eff = other[int(rng.integers(0, 2))]

    # ── Secondary features — haqiqiy label (eff) ga mos ─────────────────────
    if eff == "healthy":
        insulin = _norm(10,  3.0,  3,   20)
        homa_ir = _norm(1.5, 0.7,  0.3,  3.5)
        hdl     = _norm(58,   8,  38,   82) if gender=="male" else _norm(65,  8, 42, 90)
        ldl     = _norm(102, 16,  65,  140)
        tg      = _norm(105, 24,  55,  168)
        tchol   = _norm(172, 22, 130,  220)
        crp     = _norm(0.5, 0.4, 0.01, 2.5)
        creat   = _norm(0.88,0.12,0.60, 1.18) if gender=="male" else _norm(0.74,0.10,0.52,1.02)
        egfr    = _norm(92,   9,  68,  118)
        sbp     = _norm(113,  9,  92,  136)
        dbp     = _norm(72,   7,  58,   90)
        adipo   = _norm(15,   3,  7,   24)
        waist   = _norm(82,   8,  62,  102) if gender=="male" else _norm(75, 7, 56, 95)

    elif eff == "prediabetes":
        insulin = _norm(19,  4.5,  8,   34)
        homa_ir = _norm(3.4, 1.0,  1.5, 6.5)
        hdl     = _norm(46,   8,  28,   66) if gender=="male" else _norm(52,  8, 32, 72)
        ldl     = _norm(130, 18,  90,  172)
        tg      = _norm(172, 35, 100,  272)
        tchol   = _norm(215, 26, 165,  268)
        crp     = _norm(2.2, 0.8, 0.8,  5)
        creat   = _norm(1.04,0.14,0.72, 1.42) if gender=="male" else _norm(0.88,0.12,0.62,1.18)
        egfr    = _norm(79,  10,  52,  105)
        sbp     = _norm(130,  9, 108,  152)
        dbp     = _norm(82,   8,  68,  100)
        adipo   = _norm(9.5,  2.5, 4,  15)
        waist   = _norm(97,   9,  78,  120) if gender=="male" else _norm(89, 8, 70, 110)

    else:  # diabetes
        insulin = _norm(8.0,  4.0,  2,   22)
        homa_ir = _norm(5.8,  1.4,  2.5, 10)
        hdl     = _norm(39,   7,  22,   58) if gender=="male" else _norm(44,  7, 26, 62)
        ldl     = _norm(146, 20,  95,  200)
        tg      = _norm(238, 42, 148,  380)
        tchol   = _norm(238, 28, 180,  310)
        crp     = _norm(5.2,  1.4, 2.0,  9)
        creat   = _norm(1.25,0.22,0.82, 2.0) if gender=="male" else _norm(1.05,0.20,0.68,1.8)
        egfr    = _norm(62,  12,  28,   88)
        sbp     = _norm(144, 10, 120,  170)
        dbp     = _norm(89,   8,  72,  108)
        adipo   = _norm(5.5,  2.0, 1.5, 10)
        waist   = _norm(105, 10,  84,  132) if gender=="male" else _norm(97, 9, 76, 122)

    tg_hdl = round(tg / hdl, 3)

    bmarks = {
        "HBA1C":  round(hba1c,   2),
        "FBG":    round(glucose, 1),
        "PPG":    round(ppg,     1),
        "FINS":   round(insulin, 2),
        "HOMAIR": round(homa_ir, 3),
        "BMI":    round(bmi,     1),
        "WAIST":  round(waist,   1),
        "SBP":    round(sbp,     1),
        "DBP":    round(dbp,     1),
        "TCHOL":  round(tchol,   1),
        "HDL":    round(hdl,     1),
        "LDL":    round(ldl,     1),
        "TG":     round(tg,      1),
        "TGHDL":  round(tg_hdl,  3),
        "CRP":    round(crp,     3),
        "CREAT":  round(creat,   3),
        "EGFR":   round(egfr,    1),
        "ADIPO":  round(adipo,   2),
    }

    # Label manbalariga kichik noise, secondary featurelarga kattaroq
    label_sources = {"HBA1C", "FBG", "PPG"}
    for code in list(bmarks.keys()):
        if code in label_sources:
            bmarks[code] = round(bmarks[code] * (1 + rng.uniform(-0.06, 0.06)), 2)
        else:
            bmarks[code] = round(bmarks[code] * (1 + rng.uniform(-0.15, 0.15)), 2)

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
    score += float(rng.uniform(-0.32, 0.32))
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


def generate_bmi(status: str = "healthy") -> float:
    """BMI — klass bo'yicha kichik farq."""
    obese_prob = {"healthy": 0.38, "prediabetes": 0.44, "diabetes": 0.52}
    if rng.random() < obese_prob.get(status, 0.44):
        return _norm(32, 5, 27, 52)
    elif rng.random() < 0.38:
        return _norm(27, 2.5, 22, 32)
    else:
        return _norm(22, 3, 15, 27)


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
        bmarks = generate_biomarkers(status, gender, bmi)

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