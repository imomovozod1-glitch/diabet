"""
Temp setup: migration + 5k dataset with localhost postgres
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

PATCH = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "diabetes_risk",
    "user":     "postgres",
    "password": "postgres",
}

# ── Migration ─────────────────────────────────────────────
import diabetes_db_migration as mig
mig.DB_CONFIG.update(PATCH)

print("\n=== Running migration ===")
import psycopg2
conn = psycopg2.connect(**PATCH)
conn.autocommit = False

mig.run_migration(conn, "pgcrypto", 'CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
mig.run_migration(conn, "Enums",    mig.ENUM_TYPES)
mig.run_migration(conn, "Tables",   mig.TABLES)
mig.run_migration(conn, "Indexes",  mig.INDEXES)
mig.run_migration(conn, "Seed",     mig.SEED_DATA)
conn.close()
print("Migration done.\n")

# ── Dataset ───────────────────────────────────────────────
import generate_dataset as gd
gd.DB_CONFIG.update(PATCH)
gd.TOTAL_PATIENTS = 3_000  # fast run
gd.BATCH_SIZE     = 300

print("=== Generating 3,000 patients ===")
gd.main()
print("Dataset done.\n")
