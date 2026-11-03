"""Database SQLite untuk riwayat prediksi."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "app.db"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            longitude REAL, latitude REAL, housing_median_age REAL,
            total_rooms REAL, total_bedrooms REAL, population REAL,
            households REAL, median_income REAL, ocean_proximity TEXT,
            predicted_value REAL
        )""")


def save_prediction(d: dict, value: float):
    with conn() as c:
        c.execute("""INSERT INTO predictions
            (longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
             population, households, median_income, ocean_proximity, predicted_value)
            VALUES (:longitude,:latitude,:housing_median_age,:total_rooms,:total_bedrooms,
                    :population,:households,:median_income,:ocean_proximity,:v)""",
                  {**d, "v": value})


def list_predictions(limit: int = 20):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]


def clear_predictions():
    with conn() as c:
        c.execute("DELETE FROM predictions")
