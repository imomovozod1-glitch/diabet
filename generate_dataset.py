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
        hba1c     = _norm(5.3, 0.90, 4.0,  7.5)
        glucose   = _norm(92,  22.0, 65,   145)
        ppg       = _norm(118, 35,   72,   175)
        insulin   = _norm(13,  9.0,  2.0,  45)
        homa_ir   = _norm(2.2, 2.0,  0.2,  8.0)
        hdl       = _norm(49,  18,   20,   100)  if gender=="male" else _norm(55, 18, 22, 105)
        ldl       = _norm(115, 45,   40,   185)
        tg        = _norm(138, 85,   35,   320)
        tchol     = _norm(185, 55,   105,  295)
        crp       = _norm(1.5, 1.8,  0.01, 9.0)
        creat     = _norm(0.93,0.35, 0.45, 1.80) if gender=="male" else _norm(0.79,0.30,0.38,1.50)
        egfr      = _norm(84,  25,   38,   132)
        sbp       = _norm(121, 25,   80,   165)
        dbp       = _norm(77,  16,   50,   108)
        adipo     = _norm(10,  8,    2,    32)
        waist     = _norm(90,  18,   52,   128) if gender=="male" else _norm(83,17,45,120)

    elif status == "prediabetes":
        hba1c     = _norm(6.0 + age_factor + bmi_factor, 1.0, 4.5,  8.5)
        glucose   = _norm(108, 26.0, 72,   168)
        ppg       = _norm(152, 42,   105,  230)
        insulin   = _norm(15,  9.5,  2,    52)
        homa_ir   = _norm(2.9, 2.2,  0.3,  9.0)
        hdl       = _norm(46,  18,   18,   86)   if gender=="male" else _norm(52, 18, 20, 92)
        ldl       = _norm(126, 46,   55,   210)
        tg        = _norm(162, 92,   55,   400)
        tchol     = _norm(210, 58,   128,  315)
        crp       = _norm(2.3, 2.2,  0.05, 11.0)
        creat     = _norm(1.03,0.38, 0.50, 1.92) if gender=="male" else _norm(0.89,0.34,0.40,1.62)
        egfr      = _norm(76,  27,   28,   122)
        sbp       = _norm(126, 25,   88,   172)
        dbp       = _norm(81,  18,   52,   112)
        adipo     = _norm(8.5, 7,    1.5,  26)
        waist     = _norm(95,  20,   58,   138) if gender=="male" else _norm(87,19,50,128)

    else:  # diabetes
        severity  = rng.uniform(0, 1)
        hba1c     = _norm(7.2 + severity * 2 + age_factor + bmi_factor, 1.8, 4.5, 14.0)
        glucose   = _norm(148 + severity * 75, 55, 82, 400)
        ppg       = _norm(240 + severity * 65, 72, 145, 500)
        insulin   = _norm(11,  10.0, 1,    52)
        homa_ir   = _norm(4.0, 2.8,  0.3,  15)
        hdl       = _norm(43,  18,   15,   78)   if gender=="male" else _norm(49, 17, 17, 84)
        ldl       = _norm(135, 52,   52,   258)
        tg        = _norm(198, 108,  70,   500)
        tchol     = _norm(222, 62,   128,  368)
        crp       = _norm(4.2, 3.5,  0.1,  16)
        creat     = _norm(1.24,0.55, 0.52, 3.5)  if gender=="male" else _norm(1.02,0.50,0.44,2.8)
        egfr      = _norm(70,  30,   15,   112)
        sbp       = _norm(133, 28,   92,   195)
        dbp       = _norm(84,  20,   55,   125)
        adipo     = _norm(8.0, 6.5,  0.8,  22)
        waist     = _norm(98,  22,   65,   148) if gender=="male" else _norm(90,21,58,138)

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

    # 1-bosqich: nisbiy shovqin
    for code in list(bmarks.keys()):
        bmarks[code] = round(bmarks[code] * (1 + rng.uniform(-0.45, 0.45)), 2)

    # 2-bosqich: klasslar farqidan 2-3x katta qo'shimcha absolut shovqin —
    # secondary featurelarning signal/noise nisbatini nolga yaqinlashtiradi.
    # HbA1c/FBG/PPG shu ro'yxatda yo'q: ular label manbai, o'zgartirmaymiz.
    WASHOUT = {
        "FINS":   20.0,   # class diff ~2,  noise ~10x
        "HOMAIR":  5.0,   # class diff ~1.8, noise ~2.8x
        "BMI":    10.0,   # class diff ~3,  noise ~3x
        "WAIST":  24.0,   # class diff ~8,  noise ~3x
        "SBP":    32.0,   # class diff ~12, noise ~2.7x
        "DBP":    24.0,   # class diff ~7,  noise ~3.4x
        "TCHOL":  80.0,   # class diff ~37, noise ~2.2x
        "HDL":    25.0,   # class diff ~6,  noise ~4x
        "LDL":    58.0,   # class diff ~20, noise ~2.9x
        "TG":    115.0,   # class diff ~60, noise ~1.9x
        "TGHDL":   2.8,   # class diff ~0.5, noise ~5.6x
        "CRP":     7.5,   # class diff ~2.7, noise ~2.8x
        "CREAT":   0.6,   # class diff ~0.3, noise ~2x
        "EGFR":   35.0,   # class diff ~14, noise ~2.5x
        "ADIPO":  10.0,   # class diff ~2,  noise ~5x
    }
    for code, w in WASHOUT.items():
        if code in bmarks:
            bmarks[code] = round(bmarks[code] + float(rng.uniform(-w, w)), 2)

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


def generate_bmi(status: str) -> float:
    """BMI — klasslar o'rtasida kichik farq, katta overlap."""
    obese_prob = {"healthy": 0.42, "prediabetes": 0.46, "diabetes": 0.50}
    if rng.random() < obese_prob[status]:
        return _norm(32, 5, 28, 52)
    elif rng.random() < 0.38:
        return _norm(27, 2.5, 23, 32)
    else:
        return _norm(23, 3, 16, 28)


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

        # Haqiqiy label noise:
        #   - HbA1c/FBG/PPG → original statusdan (train_model.py label shu 3 tadan oladi)
        #   - Barcha boshqa featurelar (BMI ham!) → qo'shni statusdan
        # Natijada model ko'radigan featurelar label bilan mos kelmaydi.
        _adj = {"healthy": "prediabetes", "prediabetes": "diabetes", "diabetes": "prediabetes"}
        if rng.random() < 0.55:
            bmi_adj         = generate_bmi(_adj[status])
            bmarks_label    = generate_biomarkers(status,       age, gender, bmi)
            bmarks_features = generate_biomarkers(_adj[status], age, gender, bmi_adj)
            bmarks = bmarks_features
            bmarks["HBA1C"] = bmarks_label["HBA1C"]
            bmarks["FBG"]   = bmarks_label["FBG"]
            bmarks["PPG"]   = bmarks_label["PPG"]
        else:
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