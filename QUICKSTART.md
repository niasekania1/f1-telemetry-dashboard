# Quick Start Guide

## 🚀 Run the Telemetry Dashboard

```bash
uv run f1_telemetry_dashboard.py
```

Then open your browser to: **http://127.0.0.1:8050**

## 📊 What You'll See

The dashboard provides:
- **Driver Selection**: Choose from 20 F1 drivers (Albon, Alonso, Bottas, etc.)
- **Lap Selection**: Pick any lap from the race
- **Telemetry Types**:
  - Full Telemetry (Gear, RPM, Speed, Throttle/Brake, DRS)
  - Speed & RPM Only
  - Gears & Throttle

## 📁 Files Overview

- `f1_telemetry_dashboard.py` - Interactive dashboard
- `download_barcelona_f1_data.py` - Download script (already run)
- `barcelona_f1_2024_complete_stats_*.csv` - Race data (490KB, 1,310 laps)
- `pyproject.toml` - Dependencies (FastF1, Dash, Plotly, Pandas)

## 🎯 Example Usage

1. **Select Driver**: Choose "Max Verstappen" (VER)
2. **Select Lap**: Choose Lap 25
3. **View Telemetry**: See gear shifts, RPM, speed, throttle/brake traces

## 🔧 Manual Data Analysis

```python
import pandas as pd
df = pd.read_csv("barcelona_f1_2024_complete_stats_20251221_115840.csv")

# Get Verstappen's fastest lap
ver_laps = df[df['Driver_Abbreviation'] == 'VER']
fastest = ver_laps.loc[ver_laps['Lap_Time_Seconds'].idxmin()]
print(f"Fastest lap: {fastest['Lap_Number']} - {fastest['Lap_Time']}")
```

## 📈 Data Columns Available

- **Driver Info**: ID, Abbreviation, Full Name, Team
- **Lap Data**: Number, Time, Sector Times
- **Tyre Data**: Compound, Life, Freshness, Stint
- **Speed**: Sector speeds, Avg/Max Speed
- **Telemetry**: RPM, Throttle, Brake, Gear, DRS
- **Timing**: Lap Start, Time from Leader

## ⚡ Performance

- Dashboard loads in <5 seconds
- Real-time interactive updates
- Smooth scrolling through 1,310 laps

## 🎓 Research Applications

- Compare driver performance
- Analyze tyre degradation
- Study corner behavior (throttle/brake)
- Investigate DRS effectiveness
- Race strategy analysis