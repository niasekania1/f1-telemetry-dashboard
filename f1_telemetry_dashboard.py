#!/usr/bin/env python3
"""
F1 Telemetry Dashboard - Barcelona (Spanish GP) 2024

What’s new vs your version:
- FIX: driver dropdown pairing was wrong (you zipped two independently sorted arrays)
- FIX: avoid double session loading on Dash debug reload (use_reloader=False)
- Adds:
  * "Brake Intensity (estimated)" from longitudinal deceleration (NOT real brake pressure)
  * Longitudinal acceleration (g)
  * Track map (X/Y) colored by Speed/Throttle/Brake Intensity/Accel
  * Corner markers (CircuitInfo)
  * Braking point markers (brake onsets)
  * Summary metric cards (top speed, brake %, full throttle %, coasting %, max decel, gear shifts, etc.)
"""

from __future__ import annotations

import os
import glob
from collections import OrderedDict

import numpy as np
import pandas as pd
import fastf1

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


# ---------------------------
# Config
# ---------------------------
CACHE_DIR = "./cache"
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2024
EVENT = "Spanish Grand Prix"
SESSION_NAME = "R"

# Optional: auto-pick newest CSV if present (not required for dashboard)
CSV_GLOB = "barcelona_f1_2024_complete_stats_*.csv"


TRACK_STATUS_MAPPING = {
    "1": "Track Clear",
    "2": "Yellow Flag",
    "3": "Unknown (3)",
    "4": "Safety Car",
    "5": "Red Flag",
    "6": "Virtual Safety Car",
    "7": "VSC Ending",
}


def format_lap_time(t) -> str:
    """Format LapTime to M:SS.mmm"""
    if t is None or (isinstance(t, float) and np.isnan(t)) or pd.isna(t):
        return "N/A"

    # If it's a pandas Timedelta / datetime.timedelta
    if hasattr(t, "total_seconds"):
        total_ms = t.total_seconds() * 1000.0
        minutes = int(total_ms // 60000)
        seconds = (total_ms % 60000) / 1000.0
        return f"{minutes}:{seconds:06.3f}"

    # If it's a string (e.g. from CSV)
    s = str(t)
    if "days" in s:
        s = s.split("days")[-1].strip()
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        if float(h) > 0:
            return f"{int(h)}:{m}:{sec[:6]}"
        return f"{int(m)}:{sec[:6]}"
    return s[:10]


def get_track_status_text(status_code) -> str:
    s = str(status_code) if status_code is not None else "1"

    if s in TRACK_STATUS_MAPPING:
        return TRACK_STATUS_MAPPING[s]

    # Compounded statuses can appear; pick most severe-ish
    if "5" in s:
        return "Red Flag"
    if "4" in s:
        return "Safety Car"
    if "6" in s:
        return "Virtual Safety Car"
    if "2" in s:
        return "Yellow Flag"
    if "1" in s:
        return "Track Clear"
    return f"Status: {s}"


def status_color(status_text: str) -> str:
    if "Yellow" in status_text:
        return "#ffc107"
    if "Safety" in status_text:
        return "#fd7e14"
    if "Red" in status_text:
        return "#dc3545"
    if "Virtual" in status_text:
        return "#17a2b8"
    return "#28a745"


def latest_csv_path() -> str | None:
    files = sorted(glob.glob(CSV_GLOB), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


# ---------------------------
# Load Session (once)
# ---------------------------
print("Loading FastF1 session...")
session = fastf1.get_session(YEAR, EVENT, SESSION_NAME)
session.load(laps=True, telemetry=True, weather=True, messages=True)
print("Session loaded!")

# Circuit info (corners etc.)
circuit_info = session.get_circuit_info()  # may be None for some events

# Optional CSV
csv_path = latest_csv_path()
csv_df = pd.read_csv(csv_path) if csv_path else pd.DataFrame()

# Driver options (use official session results)
results = session.results
driver_rows = []
for _, r in results.iterrows():
    abbr = r.get("Abbreviation") or r.get("Tla") or r.get("Driver")
    name = r.get("FullName") or r.get("BroadcastName") or abbr
    team = r.get("TeamName") or r.get("Team") or ""
    label = f"{name} ({abbr})" + (f" - {team}" if team else "")
    driver_rows.append((name, abbr, label))

driver_rows = sorted(driver_rows, key=lambda x: x[0])
driver_options = [{"label": lbl, "value": abbr} for _, abbr, lbl in driver_rows]
default_driver = driver_options[0]["value"] if driver_options else None


# ---------------------------
# Telemetry caching
# ---------------------------
TEL_CACHE: OrderedDict[tuple[str, int], tuple[pd.DataFrame, pd.Series]] = OrderedDict()
TEL_CACHE_MAX = 60


def get_lap_and_tel(driver: str, lap_number: int) -> tuple[pd.Series, pd.DataFrame]:
    key = (driver, int(lap_number))
    if key in TEL_CACHE:
        TEL_CACHE.move_to_end(key)
        lap, tel = TEL_CACHE[key]
        return lap, tel

    driver_laps = session.laps.pick_driver(driver)
    lap_obj = driver_laps.pick_lap(int(lap_number)).iloc[0]

    tel = lap_obj.get_telemetry()
    # Ensure distance exists
    try:
        tel = tel.add_distance()
    except Exception:
        # If already has Distance or can't add, keep as-is
        pass

    # Derived timing axis (seconds)
    if "Date" in tel.columns and len(tel) > 1:
        t = (tel["Date"] - tel["Date"].iloc[0]).dt.total_seconds()
    else:
        t = tel["Time"].dt.total_seconds() if "Time" in tel.columns else pd.Series(np.arange(len(tel)) * 0.2)
    tel["t"] = t

    # Longitudinal accel (g) from speed derivative
    if "Speed" in tel.columns and len(tel) > 3:
        speed_ms = tel["Speed"] / 3.6
        dt = tel["t"].diff().replace(0, np.nan)
        accel = (speed_ms.diff() / dt).replace([np.inf, -np.inf], np.nan)
        tel["LongAccelG"] = accel / 9.80665
        tel["LongDecelG"] = (-tel["LongAccelG"]).where(tel["LongAccelG"] < 0, 0.0)
    else:
        tel["LongAccelG"] = 0.0
        tel["LongDecelG"] = 0.0

    # Brake boolean + estimated intensity (0..100)
    tel["Brake"] = tel.get("Brake", False).astype(bool)
    brakedecel = tel["LongDecelG"].where(tel["Brake"], 0.0)
    max_bd = float(np.nanmax(brakedecel.values)) if len(brakedecel) else 0.0
    if max_bd > 0:
        tel["BrakeIntensity"] = (100.0 * brakedecel / max_bd).clip(0, 100)
    else:
        tel["BrakeIntensity"] = 0.0

    # DRS on heuristic
    if "DRS" in tel.columns:
        tel["DRS_on"] = tel["DRS"].isin([10, 12, 14]).astype(int)
    else:
        tel["DRS_on"] = 0

    # Cache
    TEL_CACHE[key] = (lap_obj, tel)
    if len(TEL_CACHE) > TEL_CACHE_MAX:
        TEL_CACHE.popitem(last=False)

    return lap_obj, tel


def lap_metrics(tel: pd.DataFrame) -> dict:
    if tel is None or len(tel) < 5:
        return {}

    dt = tel["t"].diff()
    total_time = dt.sum(skipna=True)
    if not total_time or total_time <= 0:
        total_time = np.nan

    throttle = tel.get("Throttle", pd.Series(np.nan, index=tel.index))
    brake = tel["Brake"].astype(bool)

    full_throttle = (throttle >= 98)
    coast = (throttle <= 5) & (~brake)

    metrics = {
        "Top Speed": f"{np.nanmax(tel['Speed']):.0f} km/h" if "Speed" in tel.columns else "N/A",
        "Brake (time%)": f"{(100.0 * dt[brake].sum(skipna=True) / total_time):.1f}%" if np.isfinite(total_time) else "N/A",
        "Full Throttle (time%)": f"{(100.0 * dt[full_throttle].sum(skipna=True) / total_time):.1f}%" if np.isfinite(total_time) else "N/A",
        "Coast (time%)": f"{(100.0 * dt[coast].sum(skipna=True) / total_time):.1f}%" if np.isfinite(total_time) else "N/A",
        "Max Decel": f"{np.nanmax(tel['LongDecelG']):.2f} g" if "LongDecelG" in tel.columns else "N/A",
        "Gear Shifts": str(int((tel.get("nGear", pd.Series(np.nan)).diff() != 0).sum(skipna=True))) if "nGear" in tel.columns else "N/A",
        "Braking Events": str(int((brake.astype(int).diff() == 1).sum())),
        "DRS (time%)": f"{(100.0 * dt[tel['DRS_on'] == 1].sum(skipna=True) / total_time):.1f}%" if ("DRS_on" in tel.columns and np.isfinite(total_time)) else "N/A",
    }
    return metrics


def metric_cards(metrics: dict) -> dbc.Row:
    cols = []
    for k, v in metrics.items():
        cols.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(k, className="text-muted", style={"fontSize": "0.85rem"}),
                        html.Div(v, style={"fontSize": "1.2rem", "fontWeight": "bold"}),
                    ]),
                    className="mb-2",
                ),
                md=3,
            )
        )
    # wrap into rows of 4
    rows = []
    for i in range(0, len(cols), 4):
        rows.append(dbc.Row(cols[i:i+4], className="g-2"))
    return html.Div(rows)


# ---------------------------
# Dash App
# ---------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("F1 Telemetry Dashboard — Barcelona 2024", className="text-center my-3"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Driver"),
            dcc.Dropdown(
                id="driver-dropdown",
                options=driver_options,
                value=default_driver,
                clearable=False,
                style={"color": "#000", "backgroundColor": "#fff"},
            ),
        ], md=3),

        dbc.Col([
            html.Label("Lap"),
            dcc.Dropdown(
                id="lap-dropdown",
                clearable=False,
                style={"color": "#000", "backgroundColor": "#fff"},
            ),
        ], md=4),

        dbc.Col([
            html.Label("X-Axis"),
            dcc.Dropdown(
                id="x-axis-mode",
                options=[
                    {"label": "Distance (m)", "value": "distance"},
                    {"label": "Time (s)", "value": "time"},
                ],
                value="distance",
                clearable=False,
                style={"color": "#000", "backgroundColor": "#fff"},
            ),
        ], md=2),

        dbc.Col([
            html.Label("Map coloring"),
            dcc.Dropdown(
                id="map-color",
                options=[
                    {"label": "Speed", "value": "Speed"},
                    {"label": "Throttle", "value": "Throttle"},
                    {"label": "Brake Intensity (est.)", "value": "BrakeIntensity"},
                    {"label": "Longitudinal Accel (g)", "value": "LongAccelG"},
                ],
                value="Speed",
                clearable=False,
                style={"color": "#000", "backgroundColor": "#fff"},
            ),
        ], md=3),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(html.Div(id="metrics-row"), width=12),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="telemetry-graph", style={"height": "78vh"}), md=8),
        dbc.Col(dcc.Graph(id="track-map", style={"height": "78vh"}), md=4),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(html.Div(id="lap-info", className="text-center mt-2", style={"fontSize": "1.05rem"}), width=12),
    ]),
], fluid=True)


@app.callback(
    Output("lap-dropdown", "options"),
    Output("lap-dropdown", "value"),
    Input("driver-dropdown", "value"),
)
def update_lap_dropdown(driver: str):
    if not driver:
        return [], None

    driver_laps = session.laps.pick_driver(driver).sort_values("LapNumber")

    opts = []
    for _, r in driver_laps.iterrows():
        ln = int(r["LapNumber"])
        lt = format_lap_time(r.get("LapTime"))
        comp = r.get("Compound", "")
        tl = r.get("TyreLife", "")
        label = f"Lap {ln:>2} | {lt} | {comp} | TyreLife: {tl}"
        opts.append({"label": label, "value": ln})

    # default: fastest lap if possible
    try:
        fast = driver_laps.pick_fastest()
        default_lap = int(fast["LapNumber"])
    except Exception:
        default_lap = opts[0]["value"] if opts else None

    return opts, default_lap


@app.callback(
    Output("telemetry-graph", "figure"),
    Output("track-map", "figure"),
    Output("lap-info", "children"),
    Output("metrics-row", "children"),
    Input("driver-dropdown", "value"),
    Input("lap-dropdown", "value"),
    Input("x-axis-mode", "value"),
    Input("map-color", "value"),
)
def update_dashboard(driver: str, lap_number: int, x_mode: str, map_color: str):
    if not driver or lap_number is None:
        return go.Figure(), go.Figure(), "Select a driver and lap.", ""

    try:
        lap_obj, tel = get_lap_and_tel(driver, int(lap_number))

        # Lap info (from FastF1 laps)
        lap_time = lap_obj.get("LapTime")
        compound = lap_obj.get("Compound", "N/A")
        tyre_life = lap_obj.get("TyreLife", None)
        stint = lap_obj.get("Stint", None)
        track_status = lap_obj.get("TrackStatus", "1")
        status_txt = get_track_status_text(track_status)
        is_pit_lap = bool(pd.notna(lap_obj.get("PitInTime")) or pd.notna(lap_obj.get("PitOutTime")))

        # Weather at lap (optional)
        track_temp = air_temp = wind = None
        try:
            w = lap_obj.get_weather_data()
            track_temp = w.get("TrackTemp", None)
            air_temp = w.get("AirTemp", None)
            wind = w.get("WindSpeed", None)
        except Exception:
            pass

        # X axis
        if x_mode == "time":
            x = tel["t"]
            x_title = "Time (s)"
        else:
            x = tel["Distance"] if "Distance" in tel.columns else tel["t"]
            x_title = "Distance (m)" if "Distance" in tel.columns else "Time (s)"

        # Braking points (rising edges)
        brake_onsets = tel.index[(tel["Brake"].astype(int).diff() == 1).fillna(False)]
        onset_x = x.loc[brake_onsets] if len(brake_onsets) else pd.Series([], dtype=float)

        # --- Telemetry figure
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Gear", "RPM", "Speed", "Throttle / Brake / Brake Intensity (est.)", "Longitudinal Accel (g)", "DRS")
        )

        if "nGear" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["nGear"], mode="lines",
                line=dict(width=2, shape="hv"),
                name="Gear",
                hovertemplate="Gear: %{y}<extra></extra>"
            ), row=1, col=1)

        if "RPM" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["RPM"], mode="lines",
                line=dict(width=2),
                name="RPM",
                hovertemplate="RPM: %{y:.0f}<extra></extra>"
            ), row=2, col=1)

        if "Speed" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["Speed"], mode="lines",
                line=dict(width=2),
                name="Speed",
                hovertemplate="Speed: %{y:.0f} km/h<extra></extra>"
            ), row=3, col=1)

        if "Throttle" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["Throttle"], mode="lines",
                line=dict(width=2),
                name="Throttle",
                hovertemplate="Throttle: %{y:.0f}%<extra></extra>"
            ), row=4, col=1)

        # Brake binary
        fig.add_trace(go.Scatter(
            x=x, y=tel["Brake"].astype(int) * 100, mode="lines",
            line=dict(width=2, shape="hv"),
            name="Brake (on/off)",
            hovertemplate="Brake: %{y:.0f}%<extra></extra>"
        ), row=4, col=1)

        # Brake intensity (estimated)
        fig.add_trace(go.Scatter(
            x=x, y=tel["BrakeIntensity"], mode="lines",
            line=dict(width=2),
            name="Brake Intensity (est.)",
            hovertemplate="Brake intensity (est.): %{y:.0f}<extra></extra>"
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=tel["LongAccelG"], mode="lines",
            line=dict(width=2),
            name="LongAccelG",
            hovertemplate="Long accel: %{y:.2f} g<extra></extra>"
        ), row=5, col=1)

        if "DRS" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["DRS_on"], mode="lines",
                line=dict(width=2, shape="hv"),
                name="DRS (on)",
                hovertemplate="DRS on: %{y}<extra></extra>"
            ), row=6, col=1)

        # Corner markers (distance mode only)
        if circuit_info is not None and x_mode == "distance" and "Distance" in tel.columns:
            try:
                corners = circuit_info.corners.dropna(subset=["Distance"])
                for _, c in corners.iterrows():
                    fig.add_vline(x=float(c["Distance"]), line_dash="dot", line_width=1, opacity=0.25, row=3, col=1)
            except Exception:
                pass

        # Braking onset markers as vertical lines on Speed row
        if len(onset_x):
            for bx in onset_x.iloc[:200]:  # safety cap
                fig.add_vline(x=float(bx), line_dash="dash", line_width=1, opacity=0.25, row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=55, r=10, t=55, b=40),
            title=f"{driver} — Lap {lap_number}",
            title_x=0.5,
            hovermode="closest",
            showlegend=False,
            plot_bgcolor="black",
            paper_bgcolor="black",
        )

        fig.update_xaxes(
            title_text=x_title, row=6, col=1,
            showgrid=True, gridwidth=1, gridcolor="#333",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikedash="solid", spikethickness=1,
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#333", zeroline=False)

        # --- Track map figure
        map_fig = go.Figure()
        if "X" in tel.columns and "Y" in tel.columns:
            color_series = tel[map_color] if map_color in tel.columns else tel.get("Speed", pd.Series(np.nan, index=tel.index))

            map_fig.add_trace(go.Scattergl(
                x=tel["X"], y=tel["Y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_series,
                    colorbar=dict(title=map_color),
                ),
                hovertemplate="X:%{x:.0f}<br>Y:%{y:.0f}<extra></extra>",
                name="Track"
            ))

            # Braking onsets on map
            if len(brake_onsets):
                map_fig.add_trace(go.Scattergl(
                    x=tel.loc[brake_onsets, "X"],
                    y=tel.loc[brake_onsets, "Y"],
                    mode="markers",
                    marker=dict(size=8, color="#ff0000"), # Red for braking points
                    name="Brake onsets"
                ))

            # Corner markers
            if circuit_info is not None:
                try:
                    corners = circuit_info.corners.copy()
                    corners["Label"] = corners["Number"].astype(str) + corners["Letter"].fillna("")
                    map_fig.add_trace(go.Scattergl(
                        x=corners["X"], y=corners["Y"],
                        mode="markers+text",
                        text=corners["Label"],
                        textposition="top center",
                        marker=dict(size=7),
                        name="Corners"
                    ))
                except Exception:
                    pass

        map_fig.update_layout(
            template="plotly_dark",
            title="Track Map",
            title_x=0.5,
            margin=dict(l=10, r=10, t=55, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor="black",
            paper_bgcolor="black",
        )

        # Metrics cards
        m = lap_metrics(tel)
        # Add weather if available
        if track_temp is not None:
            m["Track Temp"] = f"{track_temp:.1f} °C"
        if air_temp is not None:
            m["Air Temp"] = f"{air_temp:.1f} °C"
        if wind is not None:
            m["Wind"] = f"{wind:.1f} m/s"

        metrics_component = metric_cards(m)

        # Lap info line
        info = [
            html.Span(f"Lap time: {format_lap_time(lap_time)}", className="mx-2"),
            html.Span("|", className="mx-2 text-muted"),
            html.Span(f"Compound: {compound}", className="mx-2"),
            html.Span("|", className="mx-2 text-muted"),
            html.Span(f"Tyre life: {int(tyre_life) if pd.notna(tyre_life) else 'N/A'}", className="mx-2"),
            html.Span("|", className="mx-2 text-muted"),
            html.Span(f"Stint: {int(stint) if pd.notna(stint) else 'N/A'}", className="mx-2"),
            html.Span("|", className="mx-2 text-muted"),
            html.Span(f"Track: {status_txt}", className="mx-2",
                      style={"color": status_color(status_txt), "fontWeight": "bold"}),
        ]
        if is_pit_lap:
            info += [html.Span("|", className="mx-2 text-muted"),
                     html.Span("🛑 PIT LAP", className="mx-2 text-danger", style={"fontWeight": "bold"})]

        return fig, map_fig, info, metrics_component

    except Exception as e:
        import traceback
        traceback.print_exc()
        return go.Figure(), go.Figure(), f"Error: {e}", ""


if __name__ == "__main__":
    print("Starting dashboard: http://127.0.0.1:8050")
    # IMPORTANT: use_reloader=False prevents loading the FastF1 session twice in debug mode
    app.run(debug=True, port=8050, use_reloader=False)
