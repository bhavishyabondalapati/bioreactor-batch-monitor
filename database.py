"""Step 2: store the simulated data in SQLite and pull it back out with SQL."""
import sqlite3

import pandas as pd

SCHEMA = """
DROP TABLE IF EXISTS sensor_readings;
DROP TABLE IF EXISTS batches;

CREATE TABLE batches (
    batch_id         TEXT PRIMARY KEY,
    is_reference     INTEGER NOT NULL,   -- 1 = known-good historical batch used to set limits
    injected_fault   TEXT NOT NULL,      -- ground truth for validation ONLY; the monitor never reads it
    fault_start_hour INTEGER,
    fault_end_hour   INTEGER,
    data_source      TEXT NOT NULL       -- always 'SIMULATED' in this project
);

CREATE TABLE sensor_readings (
    batch_id      TEXT NOT NULL REFERENCES batches(batch_id),
    hour          INTEGER NOT NULL,
    ph            REAL,
    temp_c        REAL,
    do_pct        REAL,
    agitation_rpm REAL,
    PRIMARY KEY (batch_id, hour)         -- one reading per batch per hour
);
"""


def build_database(batches, readings, path):
    """Create the tables and load the data. path=':memory:' keeps it in RAM (used by tests)."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)           # runs several SQL statements in one go
    batches.to_sql("batches", conn, if_exists="append", index=False)
    readings.to_sql("sensor_readings", conn, if_exists="append", index=False)
    conn.commit()
    return conn


def load_batches(conn):
    return pd.read_sql("SELECT * FROM batches", conn).set_index("batch_id")


def load_readings(conn, reference):
    """JOIN: readings for reference (known-good) batches, or for the new batches to monitor."""
    sql = """
        SELECT r.*
        FROM sensor_readings r
        JOIN batches b ON r.batch_id = b.batch_id
        WHERE b.is_reference = ?
        ORDER BY r.batch_id, r.hour
    """
    # "?" is a placeholder: sqlite fills it in safely (avoids SQL injection)
    return pd.read_sql(sql, conn, params=(int(reference),))


def batch_summary(conn):
    """JOIN + GROUP BY: one summary row per batch."""
    sql = """
        SELECT b.batch_id,
               b.is_reference,
               ROUND(AVG(r.ph), 3)     AS avg_ph,
               ROUND(MAX(r.ph), 3)     AS max_ph,
               ROUND(MAX(r.temp_c), 2) AS max_temp,
               ROUND(MIN(r.do_pct), 1) AS min_do,
               COUNT(*)                AS n_readings
        FROM batches b
        JOIN sensor_readings r ON r.batch_id = b.batch_id
        GROUP BY b.batch_id
        ORDER BY b.batch_id
    """
    return pd.read_sql(sql, conn)


def biggest_ph_jumps(conn, n=5):
    """Window function LAG: compare each pH reading with the previous hour in the SAME batch."""
    sql = """
        SELECT * FROM (
            SELECT batch_id, hour, ROUND(ph, 3) AS ph,
                   -- LAG(ph) = the pH one row earlier; PARTITION BY restarts it for each batch
                   ROUND(ph - LAG(ph) OVER (PARTITION BY batch_id ORDER BY hour), 3) AS ph_change
            FROM sensor_readings
        )
        WHERE ph_change IS NOT NULL      -- the first hour of each batch has no previous reading
        ORDER BY ABS(ph_change) DESC
        LIMIT ?
    """
    return pd.read_sql(sql, conn, params=(n,))
