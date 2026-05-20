#!/bin/bash
# PostgreSQL ichida ikki xil DB yaratadi:
# 1. airflow  — Airflow metama'lumotlari uchun
# 2. diabetes_risk — Bizning asosiy DB

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  SELECT 'CREATE DATABASE airflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

  SELECT 'CREATE DATABASE diabetes_risk'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'diabetes_risk')\gexec

  ALTER USER postgres WITH PASSWORD 'postgres';
  GRANT ALL PRIVILEGES ON DATABASE airflow TO postgres;
  GRANT ALL PRIVILEGES ON DATABASE diabetes_risk TO postgres;
EOSQL

echo "Databases created: airflow, diabetes_risk"

# diabetes_risk sxemasini yaratish
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname="diabetes_risk" <<-EOSQL
  CREATE EXTENSION IF NOT EXISTS "pgcrypto";

  -- Enum turlari
  DO \$\$ BEGIN
    CREATE TYPE gender_type AS ENUM ('male', 'female', 'other');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE diabetes_type AS ENUM ('type1', 'type2', 'gestational', 'prediabetes', 'unknown');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE family_relation AS ENUM ('parent', 'sibling', 'grandparent', 'uncle_aunt', 'other');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE biomarker_category AS ENUM ('blood_sugar', 'lipid', 'anthropometric', 'inflammatory', 'hormonal', 'renal', 'other');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE test_status AS ENUM ('pending', 'completed', 'cancelled', 'invalid');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE result_status AS ENUM ('normal', 'high', 'low', 'critical_high', 'critical_low');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE risk_level AS ENUM ('very_low', 'low', 'moderate', 'high', 'very_high');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE feature_direction AS ENUM ('increases_risk', 'decreases_risk', 'neutral');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE notification_type AS ENUM ('high_risk_alert', 'follow_up', 'test_reminder', 'result_ready', 'system');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  DO \$\$ BEGIN
    CREATE TYPE run_status AS ENUM ('success', 'failed', 'partial');
  EXCEPTION WHEN duplicate_object THEN NULL; END \$\$;

  -- Jadvallar
  CREATE TABLE IF NOT EXISTS doctors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(150) NOT NULL,
    specialty VARCHAR(100),
    license_no VARCHAR(50) UNIQUE NOT NULL,
    phone VARCHAR(20),
    email VARCHAR(150) UNIQUE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender gender_type NOT NULL,
    phone VARCHAR(20),
    email VARCHAR(150),
    address TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS family_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    relation family_relation NOT NULL,
    has_diabetes BOOLEAN NOT NULL DEFAULT FALSE,
    diabetes_type diabetes_type,
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS biomarkers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    code VARCHAR(30) UNIQUE NOT NULL,
    description TEXT,
    normal_min NUMERIC(10,4),
    normal_max NUMERIC(10,4),
    unit VARCHAR(30),
    category biomarker_category NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS medical_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,
    test_date TIMESTAMP NOT NULL DEFAULT NOW(),
    lab_name VARCHAR(150),
    status test_status NOT NULL DEFAULT 'pending',
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    medical_test_id UUID NOT NULL REFERENCES medical_tests(id) ON DELETE CASCADE,
    biomarker_id UUID NOT NULL REFERENCES biomarkers(id),
    value NUMERIC(12,4) NOT NULL,
    unit VARCHAR(30),
    status result_status NOT NULL DEFAULT 'normal',
    lab_note TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (medical_test_id, biomarker_id)
  );

  CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(100),
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc_roc FLOAT,
    feature_list TEXT[],
    model_path TEXT,
    trained_at TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
  );

  CREATE TABLE IF NOT EXISTS risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES ml_models(id),
    medical_test_id UUID REFERENCES medical_tests(id) ON DELETE SET NULL,
    risk_score FLOAT NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    risk_level risk_level NOT NULL,
    recommendation TEXT,
    assessed_by VARCHAR(100),
    assessed_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS model_run_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_assessment_id UUID NOT NULL REFERENCES risk_assessments(id) ON DELETE CASCADE,
    run_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    input_snapshot JSONB NOT NULL,
    execution_time_ms INTEGER,
    model_version VARCHAR(20),
    status run_status NOT NULL DEFAULT 'success',
    error_message TEXT
  );

  CREATE TABLE IF NOT EXISTS feature_contributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_log_id UUID NOT NULL REFERENCES model_run_logs(id) ON DELETE CASCADE,
    biomarker_id UUID NOT NULL REFERENCES biomarkers(id),
    input_value NUMERIC(12,4),
    importance_score FLOAT NOT NULL,
    direction feature_direction NOT NULL DEFAULT 'neutral',
    shap_value FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    risk_assessment_id UUID REFERENCES risk_assessments(id) ON DELETE SET NULL,
    type notification_type NOT NULL,
    message TEXT NOT NULL,
    is_sent BOOLEAN NOT NULL DEFAULT FALSE,
    sent_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
  );

  -- Indekslar
  CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email);
  CREATE INDEX IF NOT EXISTS idx_medical_tests_patient ON medical_tests(patient_id);
  CREATE INDEX IF NOT EXISTS idx_medical_tests_date ON medical_tests(test_date DESC);
  CREATE INDEX IF NOT EXISTS idx_test_results_test ON test_results(medical_test_id);
  CREATE INDEX IF NOT EXISTS idx_risk_patient ON risk_assessments(patient_id);
  CREATE INDEX IF NOT EXISTS idx_risk_assessed_at ON risk_assessments(assessed_at DESC);

  -- Seed: Shifokorlar
  INSERT INTO doctors (full_name, specialty, license_no, phone, email) VALUES
    ('Dr. Aziz Karimov', 'Endocrinology', 'UZ-END-001', '+998901234567', 'a.karimov@clinic.uz'),
    ('Dr. Malika Tosheva', 'General Practice', 'UZ-GP-002', '+998907654321', 'm.tosheva@clinic.uz')
  ON CONFLICT (license_no) DO NOTHING;

  -- Seed: Biomarkerlar
  INSERT INTO biomarkers (name, code, description, normal_min, normal_max, unit, category) VALUES
    ('HbA1c', 'HBA1C', 'Glikozillangan gemoglobin', 4.0, 5.6, '%', 'blood_sugar'),
    ('Fasting glucose', 'FBG', 'Och qoringa qon shakari', 70, 100, 'mg/dL', 'blood_sugar'),
    ('Postprandial glucose', 'PPG', '2 soat keyin qon shakari', 70, 140, 'mg/dL', 'blood_sugar'),
    ('Fasting insulin', 'FINS', 'Och qoringa insulin', 2.6, 24.9, 'uIU/mL', 'hormonal'),
    ('HOMA-IR', 'HOMAIR', 'Insulin qarshilik indeksi', 0, 2.5, 'index', 'hormonal'),
    ('BMI', 'BMI', 'Tana massasi indeksi', 18.5, 24.9, 'kg/m2', 'anthropometric'),
    ('Waist circumference', 'WAIST', 'Bel aylanasi', 0, 94, 'cm', 'anthropometric'),
    ('Systolic BP', 'SBP', 'Sistolik qon bosimi', 90, 120, 'mmHg', 'other'),
    ('Diastolic BP', 'DBP', 'Diastolik qon bosimi', 60, 80, 'mmHg', 'other'),
    ('Total cholesterol', 'TCHOL', 'Umumiy xolesterol', 0, 200, 'mg/dL', 'lipid'),
    ('HDL cholesterol', 'HDL', 'Yaxshi xolesterol', 40, 999, 'mg/dL', 'lipid'),
    ('LDL cholesterol', 'LDL', 'Yomon xolesterol', 0, 100, 'mg/dL', 'lipid'),
    ('Triglycerides', 'TG', 'Trigliseridlar', 0, 150, 'mg/dL', 'lipid'),
    ('TG/HDL ratio', 'TGHDL', 'Trigliserid/HDL nisbati', 0, 2.0, 'ratio', 'lipid'),
    ('C-reactive protein', 'CRP', 'Yalliglanish markeri', 0, 1.0, 'mg/L', 'inflammatory'),
    ('Creatinine', 'CREAT', 'Buyrak funksiyasi', 0.6, 1.2, 'mg/dL', 'renal'),
    ('eGFR', 'EGFR', 'Glomerular filtrasiya tezligi', 60, 999, 'mL/min', 'renal'),
    ('Adiponectin', 'ADIPO', 'Adipokin gormoni', 5, 30, 'ug/mL', 'hormonal')
  ON CONFLICT (code) DO NOTHING;

  -- Seed: ML model
  INSERT INTO ml_models (model_name, version, algorithm, accuracy, precision_score, recall, f1_score, auc_roc, feature_list, is_active, trained_at)
  VALUES (
    'DiabetesRiskClassifier', 'v1.0', 'XGBoost',
    0.924, 0.911, 0.887, 0.899, 0.963,
    ARRAY['HBA1C','FBG','BMI','HOMAIR','TGHDL','WAIST','AGE','FAMILY_HISTORY'],
    TRUE, NOW()
  ) ON CONFLICT (model_name, version) DO NOTHING;
EOSQL

echo "diabetes_risk schema created successfully"
