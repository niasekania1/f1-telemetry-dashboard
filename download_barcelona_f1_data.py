#!/usr/bin/env python3
"""
Download Barcelona (Spanish GP) 2024 race data and extract lap stats + derived telemetry metrics.

IMPORTANT:
- FastF1 provides `Brake` only as a boolean (on/off). There is NO real brake pressure channel.
- "Brake intensity" style metrics below are derived from speed/time (deceleration), not pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import fastf1
from datetime import datetime
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

YEAR = 2024
EVENT = "Spanish Grand Prix"   # Barcelona
SESSION = "R"                  # Race


def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def compute_telemetry_metrics(car_data: pd.DataFrame | None) -> dict:
    """
    Compute per-lap summary metrics from car telemetry (car_data).
    Uses Date for time deltas (recommended), then derives accel/decel.

    Returns keys that are safe to merge into the lap record.
    """
    metrics = {
        # basic
        "Avg_Speed": None,
        "Max_Speed": None,
        "Min_Speed": None,
        "Avg_RPM": None,
        "Max_RPM": None,
        "Avg_Throttle": None,
        "Avg_Brake_OnPct": None,          # % samples with brake applied (time-proxy)
        "Avg_Gear": None,
        "Avg_DRS": None,

        # time/usage
        "Brake_TimePct": None,
        "FullThrottle_TimePct": None,
        "Coast_TimePct": None,
        "DRS_TimePct": None,

        # events
        "Braking_Events": None,
        "Gear_Shifts": None,

        # dynamics (derived)
        "Max_Accel_g": None,
        "Max_Decel_g": None,
        "Avg_Decel_g_When_Braking": None,
    }

    if car_data is None or len(car_data) < 5:
        return metrics

    tel = car_data.copy()

    # Basic averages
    if "Speed" in tel.columns:
        metrics["Avg_Speed"] = _safe_float(tel["Speed"].mean())
        metrics["Max_Speed"] = _safe_float(tel["Speed"].max())
        metrics["Min_Speed"] = _safe_float(tel["Speed"].min())

    if "RPM" in tel.columns:
        metrics["Avg_RPM"] = _safe_float(tel["RPM"].mean())
        metrics["Max_RPM"] = _safe_float(tel["RPM"].max())

    if "Throttle" in tel.columns:
        metrics["Avg_Throttle"] = _safe_float(tel["Throttle"].mean())

    if "nGear" in tel.columns:
        metrics["Avg_Gear"] = _safe_float(tel["nGear"].mean())
    elif "Gear" in tel.columns:
        metrics["Avg_Gear"] = _safe_float(tel["Gear"].mean())

    if "DRS" in tel.columns:
        metrics["Avg_DRS"] = _safe_float(tel["DRS"].mean())

    # Brake is boolean on/off
    brake = tel.get("Brake", pd.Series(False, index=tel.index)).astype(bool)
    metrics["Avg_Brake_OnPct"] = _safe_float(100.0 * brake.mean())

    # Use Date-based timing for dt (Time can have duplicates)
    if "Date" not in tel.columns or "Speed" not in tel.columns:
        return metrics

    tel = tel.dropna(subset=["Date", "Speed"])
    if len(tel) < 5:
        return metrics

    t = (tel["Date"] - tel["Date"].iloc[0]).dt.total_seconds()
    dt = t.diff().replace(0, np.nan)

    total_time = dt.sum(skipna=True)
    if total_time and total_time > 0:
        throttle = tel.get("Throttle", pd.Series(np.nan, index=tel.index))

        metrics["Brake_TimePct"] = _safe_float(100.0 * dt[brake].sum(skipna=True) / total_time)

        full_throttle = (throttle >= 98)
        metrics["FullThrottle_TimePct"] = _safe_float(100.0 * dt[full_throttle].sum(skipna=True) / total_time)

        coast = (throttle <= 5) & (~brake)
        metrics["Coast_TimePct"] = _safe_float(100.0 * dt[coast].sum(skipna=True) / total_time)

        # DRS "on" heuristic: codes 10/12/14 commonly represent open
        if "DRS" in tel.columns:
            drs_on = tel["DRS"].isin([10, 12, 14])
            metrics["DRS_TimePct"] = _safe_float(100.0 * dt[drs_on].sum(skipna=True) / total_time)

    # Braking events = rising edges
    metrics["Braking_Events"] = int((brake.astype(int).diff() == 1).sum())

    # Gear shifts
    if "nGear" in tel.columns:
        g = tel["nGear"]
        metrics["Gear_Shifts"] = int((g.diff() != 0).sum(skipna=True))

    # Longitudinal accel/decel from speed derivative
    speed_ms = tel["Speed"] / 3.6
    dv = speed_ms.diff()
    accel = (dv / dt).replace([np.inf, -np.inf], np.nan)
    accel_g = accel / 9.80665

    metrics["Max_Accel_g"] = _safe_float(accel_g.max(skipna=True))

    decel_g = (-accel_g).where(accel_g < 0, 0.0)
    metrics["Max_Decel_g"] = _safe_float(decel_g.max(skipna=True))

    brake_decel = (-accel_g).where(brake & (accel_g < 0))
    metrics["Avg_Decel_g_When_Braking"] = _safe_float(brake_decel.mean(skipna=True))

    return metrics


def download_barcelona_data():
    print(f"Loading {YEAR} {EVENT} ({SESSION})...")
    session = fastf1.get_session(YEAR, EVENT, SESSION)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    print("Session loaded!")

    # Build driver mapping
    driver_info = {}
    for drv_num in session.drivers:
        info = session.get_driver(drv_num)
        rec = {
            "Abbreviation": info.get("Abbreviation", str(drv_num)),
            "FirstName": info.get("FirstName", ""),
            "LastName": info.get("LastName", ""),
            "FullName": (f"{info.get('FirstName','')} {info.get('LastName','')}").strip(),
            "TeamName": info.get("TeamName", ""),
        }
        driver_info[str(drv_num)] = rec
        driver_info[rec["Abbreviation"]] = rec

    rows = []
    telemetry_errors = 0

    for _, lap in session.laps.iterrows():
        driver_id = lap.get("Driver")
        info = driver_info.get(driver_id, None)
        if info is None:
            continue

        # Skip laps without a real laptime
        if pd.isna(lap.get("LapTime")):
            continue

        # Telemetry-derived metrics
        try:
            car_data = lap.get_car_data()
            tel_metrics = compute_telemetry_metrics(car_data)
        except Exception:
            telemetry_errors += 1
            tel_metrics = compute_telemetry_metrics(None)

        # FIX: your original code used `is not pd.NaT` which is incorrect.
        is_pit_lap = bool(pd.notna(lap.get("PitInTime")) or pd.notna(lap.get("PitOutTime")))

        record = {
            "Year": YEAR,
            "Event": session.event.get("EventName", EVENT),
            "Session": SESSION,

            "Driver_ID": driver_id,
            "Driver_Abbreviation": info["Abbreviation"],
            "Driver_Full_Name": info["FullName"],
            "Team": info["TeamName"],

            "Lap_Number": int(lap.get("LapNumber")),
            "Lap_Time": str(lap.get("LapTime")),
            "Lap_Time_Seconds": lap.get("LapTime").total_seconds() if pd.notna(lap.get("LapTime")) else None,

            "Sector1_Time": str(lap.get("Sector1Time")) if pd.notna(lap.get("Sector1Time")) else None,
            "Sector2_Time": str(lap.get("Sector2Time")) if pd.notna(lap.get("Sector2Time")) else None,
            "Sector3_Time": str(lap.get("Sector3Time")) if pd.notna(lap.get("Sector3Time")) else None,

            "Compound": lap.get("Compound"),
            "Tyre_Life": lap.get("TyreLife"),
            "Fresh_Tyre": lap.get("FreshTyre"),
            "Stint": lap.get("Stint"),
            "Lap_In_Stint": None,  # filled after sorting

            "Track_Status": lap.get("TrackStatus"),
            "Is_Pit_Lap": is_pit_lap,
            "Pit_In_Time": str(lap.get("PitInTime")) if pd.notna(lap.get("PitInTime")) else None,
            "Pit_Out_Time": str(lap.get("PitOutTime")) if pd.notna(lap.get("PitOutTime")) else None,

            "Speed_I1": lap.get("SpeedI1") if pd.notna(lap.get("SpeedI1")) else None,
            "Speed_I2": lap.get("SpeedI2") if pd.notna(lap.get("SpeedI2")) else None,
            "Speed_FL": lap.get("SpeedFL") if pd.notna(lap.get("SpeedFL")) else None,
            "Speed_ST": lap.get("SpeedST") if pd.notna(lap.get("SpeedST")) else None,

            "Lap_Start_Time": str(lap.get("LapStartTime")) if pd.notna(lap.get("LapStartTime")) else None,
            "Time_from_Leader": str(lap.get("Time")) if pd.notna(lap.get("Time")) else None,
        }

        record.update(tel_metrics)
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data extracted!")
        return None

    # Clean + sort + compute Lap_In_Stint
    df["Stint"] = df["Stint"].fillna(-1).astype(int)
    df = df.sort_values(["Driver_Abbreviation", "Lap_Number"])
    df["Lap_In_Stint"] = df.groupby(["Driver_Abbreviation", "Stint"]).cumcount() + 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"barcelona_f1_2024_complete_stats_{ts}.csv"
    df.to_csv(filename, index=False)

    print(f"\nSUCCESS: saved {len(df)} laps -> {filename}")
    if telemetry_errors:
        print(f"Telemetry errors: {telemetry_errors}")

    return filename


if __name__ == "__main__":
    download_barcelona_data()
