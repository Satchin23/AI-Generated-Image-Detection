import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), "forensics_database.sqlite3")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
tables = [r[0] for r in cur.fetchall()]

print("=" * 70)
print(f"{'TABLE NAME':<25} | {'COLUMNS':<8} | {'ROWS':<8}")
print("=" * 70)

for t in tables:
    col_count = len(cur.execute(f"PRAGMA table_info({t});").fetchall())
    row_count = cur.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
    print(f"{t:<25} | {col_count:<8} | {row_count:<8}")

print("=" * 70)
conn.close()

