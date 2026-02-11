#!/usr/bin/env python3
"""
ATIS — Automated Telemetry Insight System
F1 Telemetry Dashboard — Barcelona (Spanish GP) 2024

Rebuilt with:
- Clean, responsive dark-themed layout with F1-inspired branding
- Horizontal scrolling metric cards with color-coded accents
- Unified hover crosshair across all telemetry subplots
- Floating AI Race Engineer chat bubble (mock responses, ready for LLM integration)
- Track map drawn as continuous colored line with Turbo colorscale

All telemetry processing logic (FastF1, caching, metrics) preserved from original.
"""

from __future__ import annotations

import os
import glob
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import fastf1

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, Patch
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


# ---------------------------
# Config
# ---------------------------
CACHE_DIR = "./cache"
fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2024
EVENT = "Spanish Grand Prix"
SESSION_NAME = "R"

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

    if hasattr(t, "total_seconds"):
        total_ms = t.total_seconds() * 1000.0
        minutes = int(total_ms // 60000)
        seconds = (total_ms % 60000) / 1000.0
        return f"{minutes}:{seconds:06.3f}"

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


def status_badge_color(status_text: str) -> str:
    """Map track status to a dbc.Badge color name."""
    if "Yellow" in status_text:
        return "warning"
    if "Safety" in status_text or "Virtual" in status_text:
        return "info"
    if "Red" in status_text:
        return "danger"
    return "success"


COMPOUND_COLORS = {
    "SOFT": "danger",
    "MEDIUM": "warning",
    "HARD": "light",
    "INTERMEDIATE": "success",
    "WET": "info",
}


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

circuit_info = session.get_circuit_info()

csv_path = latest_csv_path()
csv_df = pd.read_csv(csv_path) if csv_path else pd.DataFrame()

# Driver options
results = session.results
driver_rows = []
for _, r in results.iterrows():
    abbr = r.get("Abbreviation") or r.get("Tla") or r.get("Driver")
    name = r.get("FullName") or r.get("BroadcastName") or abbr
    team = r.get("TeamName") or r.get("Team") or ""
    label = f"{abbr} — {team}" if team else abbr
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
    try:
        tel = tel.add_distance()
    except Exception:
        pass

    if "Date" in tel.columns and len(tel) > 1:
        t = (tel["Date"] - tel["Date"].iloc[0]).dt.total_seconds()
    else:
        t = tel["Time"].dt.total_seconds() if "Time" in tel.columns else pd.Series(np.arange(len(tel)) * 0.2)
    tel["t"] = t

    if "Speed" in tel.columns and len(tel) > 3:
        speed_ms = tel["Speed"] / 3.6
        dt = tel["t"].diff().replace(0, np.nan)
        accel = (speed_ms.diff() / dt).replace([np.inf, -np.inf], np.nan)
        tel["LongAccelG"] = accel / 9.80665
        tel["LongDecelG"] = (-tel["LongAccelG"]).where(tel["LongAccelG"] < 0, 0.0)
    else:
        tel["LongAccelG"] = 0.0
        tel["LongDecelG"] = 0.0

    tel["Brake"] = tel.get("Brake", False).astype(bool)
    brakedecel = tel["LongDecelG"].where(tel["Brake"], 0.0)
    max_bd = float(np.nanmax(brakedecel.values)) if len(brakedecel) else 0.0
    if max_bd > 0:
        tel["BrakeIntensity"] = (100.0 * brakedecel / max_bd).clip(0, 100)
    else:
        tel["BrakeIntensity"] = 0.0

    if "DRS" in tel.columns:
        tel["DRS_on"] = tel["DRS"].isin([10, 12, 14]).astype(int)
    else:
        tel["DRS_on"] = 0

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


# ---------------------------
# Metric Cards (redesigned)
# ---------------------------
METRIC_CONFIG = {
    "Top Speed":              {"icon": "\u2191",  "accent": "#00d2ff"},
    "Brake (time%)":          {"icon": "\u25A0",  "accent": "#e10600"},
    "Full Throttle (time%)":  {"icon": "\u25B6",  "accent": "#00ff87"},
    "Coast (time%)":          {"icon": "\u25CB",  "accent": "#ffc107"},
    "Max Decel":              {"icon": "\u25BC",  "accent": "#ff6b6b"},
    "Gear Shifts":            {"icon": "\u21C5",  "accent": "#a78bfa"},
    "Braking Events":         {"icon": "\u2759",  "accent": "#ff8c42"},
    "DRS (time%)":            {"icon": "\u2195",  "accent": "#38bdf8"},
    "Track Temp":             {"icon": "\u2600",  "accent": "#fbbf24"},
    "Air Temp":               {"icon": "\u2601",  "accent": "#94a3b8"},
    "Wind":                   {"icon": "\u2634",  "accent": "#67e8f9"},
}


def metric_cards(metrics: dict) -> html.Div:
    cards = []
    for key, value in metrics.items():
        cfg = METRIC_CONFIG.get(key, {"icon": "\u2022", "accent": "#6b7280"})
        cards.append(
            html.Div([
                html.Div(cfg["icon"], className="metric-icon",
                         style={"color": cfg["accent"]}),
                html.Div([
                    html.Div(value, className="metric-value"),
                    html.Div(key, className="metric-label"),
                ], className="metric-text"),
            ], className="metric-card",
               style={"borderLeft": f"3px solid {cfg['accent']}"}),
        )
    return html.Div(cards, className="metrics-strip")


# ---------------------------
# Chart styling constants
# ---------------------------
CHART_BG = "#0f0f1a"

TRACE_COLORS = {
    "Gear":                "#a78bfa",
    "RPM":                 "#fbbf24",
    "Speed":               "#00d2ff",
    "Throttle":            "#00ff87",
    "Brake (on/off)":      "#e10600",
    "Brake Intensity":     "#ff6b6b",
    "LongAccelG":          "#38bdf8",
    "DRS":                 "#c084fc",
}


# ---------------------------
# AI Chat — Mock Response System
# ---------------------------
MOCK_GREETINGS = [
    "Copy {driver}. Looking at lap {lap} now. What do you need?",
    "Roger. I've got {driver}'s lap {lap} telemetry pulled up. Fire away.",
    "Copy that. {driver}, lap {lap} — I'm ready. What are we looking at?",
]

MOCK_RESPONSES = {
    "braking": [
        "Looking at the brake trace, {driver} is hitting the brakes {braking_events} times this lap. "
        "Max deceleration is {max_decel}. The heavy braking zones into Turn 1, Turn 5, and the chicane "
        "are where we'd normally look for time. Check if the brake onset is consistent across those zones.",
        "Brake application accounts for {brake_pct} of the lap. That's reasonable for Barcelona. "
        "The Turn 10 braking point is the one to watch — a late-brake there can find a couple of tenths.",
    ],
    "speed": [
        "Top speed this lap is {top_speed}. {driver} is getting a solid run through the main straight. "
        "DRS activation is at {drs_pct} of the lap. The mid-corner speeds through Turns 3 and 9 are "
        "where the real time is made at this circuit.",
        "Speed trace looks clean. Peak of {top_speed} on the main straight. If you want to find "
        "more pace, look at the minimum speeds in the slow corners — that's where the car balance matters most.",
    ],
    "throttle": [
        "Full throttle at {throttle_pct} of the lap, coasting at {coast_pct}. Barcelona rewards early "
        "throttle application through the long corners. The brake-to-throttle transition in the Turn 5 "
        "complex is critical — any hesitation there costs time.",
        "{throttle_pct} full throttle, {coast_pct} coasting. The coasting percentage is worth investigating — "
        "ideally you want to be either on the brakes or on the throttle, minimal time in between.",
    ],
    "tyre": [
        "Tyre management is crucial at Barcelona — the long Turn 3 right-hander is particularly hard on "
        "the front-left. If you see the speed traces degrading through that corner across laps, that's "
        "thermal degradation kicking in.",
        "Keep an eye on the tyre life relative to the stint. Barcelona is one of the hardest circuits on "
        "tyres. If the throttle application is getting more tentative in the high-speed corners, it's "
        "usually the rear starting to go.",
    ],
    "general": [
        "Solid lap from {driver}. {top_speed} top speed, {brake_pct} braking, {throttle_pct} full throttle. "
        "The main areas to focus on at Barcelona are the long Turn 3 right-hander and getting a clean "
        "exit out of the final chicane onto the main straight.",
        "Copy. The data shows {gear_shifts} gear shifts and {braking_events} braking events — consistent "
        "with a clean lap. Ask me about a specific corner or zone and I can give you more detail.",
        "Looking at the overall picture: {top_speed} peak speed, {max_decel} max decel, {drs_pct} DRS usage. "
        "If you want to deep-dive, tell me which sector or corner you're interested in.",
    ],
}

MOCK_HELP = (
    "I can help you analyze the current lap. Try asking about:\n"
    "- **Braking** — brake zones, deceleration, brake events\n"
    "- **Speed** — top speed, DRS, speed trace\n"
    "- **Throttle** — full throttle %, coasting, pedal application\n"
    "- **Tyres** — degradation, compounds, stint management\n"
    "- Or just ask a general question about the lap!"
)


def generate_mock_response(message: str, driver: str | None, lap: int | None) -> str:
    """Generate a mock race engineer response using current telemetry context."""
    metrics_data = {}
    if driver and lap:
        try:
            _, tel = get_lap_and_tel(driver, int(lap))
            metrics_data = lap_metrics(tel)
        except Exception:
            pass

    template_vars = {
        "driver": driver or "driver",
        "lap": lap or "N/A",
        "top_speed": metrics_data.get("Top Speed", "N/A"),
        "brake_pct": metrics_data.get("Brake (time%)", "N/A"),
        "throttle_pct": metrics_data.get("Full Throttle (time%)", "N/A"),
        "coast_pct": metrics_data.get("Coast (time%)", "N/A"),
        "max_decel": metrics_data.get("Max Decel", "N/A"),
        "gear_shifts": metrics_data.get("Gear Shifts", "N/A"),
        "braking_events": metrics_data.get("Braking Events", "N/A"),
        "drs_pct": metrics_data.get("DRS (time%)", "N/A"),
    }

    msg_lower = message.lower()
    if any(w in msg_lower for w in ["help", "what can", "how do"]):
        return MOCK_HELP
    if any(w in msg_lower for w in ["hello", "hi", "hey", "sup"]):
        return random.choice(MOCK_GREETINGS).format(**template_vars)
    if any(w in msg_lower for w in ["brake", "braking", "decel", "stop"]):
        category = "braking"
    elif any(w in msg_lower for w in ["speed", "fast", "top", "straight", "drs"]):
        category = "speed"
    elif any(w in msg_lower for w in ["throttle", "gas", "power", "coast", "pedal"]):
        category = "throttle"
    elif any(w in msg_lower for w in ["tyre", "tire", "compound", "deg", "wear", "stint"]):
        category = "tyre"
    else:
        category = "general"

    return random.choice(MOCK_RESPONSES[category]).format(**template_vars)


# ---------------------------
# Dash App
# ---------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "ATIS — F1 Telemetry"

# --- Layout Components ---

header = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            # Logo / Title
            dbc.Col([
                html.Div([
                    html.Span("ATIS", style={
                        "fontWeight": "800",
                        "fontSize": "1.4rem",
                        "color": "#e10600",
                        "letterSpacing": "0.1em",
                    }),
                    html.Span(" Telemetry", style={
                        "fontWeight": "300",
                        "fontSize": "1.4rem",
                        "color": "#ffffff",
                    }),
                    html.Div("Barcelona 2024 — Race", style={
                        "fontSize": "0.72rem",
                        "color": "#6b7280",
                        "marginTop": "-2px",
                        "letterSpacing": "0.03em",
                    }),
                ])
            ], width="auto"),

            # Driver + Lap dropdowns
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Driver", className="control-label"),
                        dcc.Dropdown(
                            id="driver-dropdown",
                            options=driver_options,
                            value=default_driver,
                            clearable=False,
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Label("Lap", className="control-label"),
                        dcc.Dropdown(
                            id="lap-dropdown",
                            clearable=False,
                        ),
                    ], md=6),
                ], className="g-2"),
            ], md=6, lg=5, className="ms-auto"),
        ], align="center", className="w-100"),
    ], fluid=True),
    dark=True,
    className="px-3 py-2",
    style={
        "borderBottom": "2px solid #e10600",
        "background": "linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)",
    },
)

control_strip = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("X-Axis", className="control-label"),
                dbc.Select(
                    id="x-axis-mode",
                    options=[
                        {"label": "Distance (m)", "value": "distance"},
                        {"label": "Time (s)", "value": "time"},
                    ],
                    value="distance",
                    className="custom-dark-select",
                ),
            ], xs=6, md=2),
            dbc.Col([
                html.Label("Map Color", className="control-label"),
                dbc.Select(
                    id="map-color",
                    options=[
                        {"label": "Speed", "value": "Speed"},
                        {"label": "Throttle", "value": "Throttle"},
                        {"label": "Brake Intensity", "value": "BrakeIntensity"},
                        {"label": "Long. Accel (g)", "value": "LongAccelG"},
                    ],
                    value="Speed",
                    className="custom-dark-select",
                ),
            ], xs=6, md=2),
            dbc.Col([
                html.Div(id="lap-info", className="lap-info-strip"),
            ], md=8),
        ], align="center", className="g-2"),
    ], fluid=True),
], className="control-strip")

metrics_area = html.Div(id="metrics-row")

# --- Playback Bar ---
playback_bar = html.Div([
    # Left: speed controls
    html.Div([
        html.Button("0.25x", id="speed-025", className="speed-btn", n_clicks=0),
        html.Button("0.5x",  id="speed-05",  className="speed-btn", n_clicks=0),
        html.Button("1x",    id="speed-1",   className="speed-btn speed-btn-active", n_clicks=0),
        html.Button("2x",    id="speed-2",   className="speed-btn", n_clicks=0),
        html.Button("4x",    id="speed-4",   className="speed-btn", n_clicks=0),
    ], className="speed-controls"),

    # Center: play/pause + progress bar
    html.Div([
        html.Button("\u25B6", id="play-btn", className="play-btn", n_clicks=0),
        html.Div([
            html.Span("0.0s", id="replay-time-label", className="replay-time-label"),
            dcc.Slider(
                id="replay-slider",
                min=0, max=100, step=0.1, value=0,
                marks=None,
                updatemode="drag",
                tooltip={"placement": "top", "always_visible": False},
                className="replay-slider",
            ),
            html.Span("--s", id="replay-total-label", className="replay-time-label"),
        ], className="progress-bar-area"),
    ], className="transport-controls"),

    # Right: live readout
    html.Div(id="live-readout", className="live-readout"),

    # Hidden state stores
    dcc.Store(id="replay-playing", data=False),
    dcc.Store(id="replay-speed", data=1.0),
    dcc.Store(id="replay-index", data=0),
    dcc.Store(id="replay-tel-json", data=None),
    dcc.Interval(id="replay-interval", interval=50, disabled=True),
], id="playback-bar", className="playback-bar")

content_area = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div(
                dcc.Graph(id="telemetry-graph", style={"height": "calc(100vh - 260px)", "minHeight": "400px"},
                          config={"displayModeBar": False}),
                className="chart-wrapper",
            ),
            lg=8, md=12, className="mb-2",
        ),
        dbc.Col(
            html.Div(
                dcc.Graph(id="track-map", style={"height": "calc(100vh - 260px)", "minHeight": "400px"},
                          config={"displayModeBar": False}),
                className="chart-wrapper",
            ),
            lg=4, md=12, className="mb-2",
        ),
    ], className="g-2"),
], fluid=True, className="mt-1")

# --- Chat Bubble ---
chat_bubble = html.Div([
    # Floating action button
    html.Button(
        "\U0001F3CE",
        id="chat-toggle-btn",
        className="chat-fab",
        n_clicks=0,
    ),

    # Chat panel
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("\U0001F3CE", style={"marginRight": "8px", "fontSize": "1.1rem"}),
                html.Span("Race Engineer", style={"fontWeight": "700"}),
            ]),
            html.Button("\u2715", id="chat-close-btn",
                        className="chat-close-btn", n_clicks=0),
        ], className="chat-header"),

        # Messages area
        html.Div([
            html.Div(
                "Hey! I'm your Race Engineer assistant. I can see the telemetry "
                "you're looking at right now. Ask me about braking zones, speed traces, "
                "throttle application — or type 'help' to see what I can do.",
                className="chat-msg chat-msg-ai",
            ),
        ], id="chat-messages", className="chat-messages"),

        # Input area
        html.Div([
            dbc.Input(
                id="chat-input",
                placeholder="Ask about this lap...",
                type="text",
                className="chat-text-input",
                debounce=True,
            ),
            html.Button("Send", id="chat-send-btn",
                        className="chat-send-btn", n_clicks=0),
        ], className="chat-input-area"),
    ], id="chat-panel", className="chat-panel", style={"display": "none"}),

    # State stores
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="chat-panel-open", data=False),
], className="chat-container")

# --- Full Layout ---
app.layout = html.Div([
    header,
    control_strip,
    metrics_area,
    playback_bar,
    content_area,
    chat_bubble,
])


# ---------------------------
# Callbacks
# ---------------------------

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
        tl_str = str(int(tl)) if pd.notna(tl) else ""
        comp_short = str(comp)[0] if comp else ""
        label = f"L{ln:>2}  {lt}  {comp_short}  T{tl_str}"
        opts.append({"label": label, "value": ln})

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
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
        )
        return empty_fig, empty_fig, "Select a driver and lap.", ""

    try:
        lap_obj, tel = get_lap_and_tel(driver, int(lap_number))

        # Lap info
        lap_time = lap_obj.get("LapTime")
        compound = lap_obj.get("Compound", "N/A")
        tyre_life = lap_obj.get("TyreLife", None)
        stint = lap_obj.get("Stint", None)
        track_status = lap_obj.get("TrackStatus", "1")
        status_txt = get_track_status_text(track_status)
        is_pit_lap = bool(pd.notna(lap_obj.get("PitInTime")) or pd.notna(lap_obj.get("PitOutTime")))

        # Weather
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

        # ---- Telemetry figure ----
        fig = make_subplots(
            rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=("Gear", "RPM", "Speed (km/h)",
                            "Throttle / Brake / Brake Intensity",
                            "Longitudinal Accel (g)", "DRS"),
        )

        if "nGear" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["nGear"], mode="lines",
                line=dict(width=1.5, shape="hv", color=TRACE_COLORS["Gear"]),
                name="Gear",
                hovertemplate="Gear: %{y}<extra></extra>",
            ), row=1, col=1)

        if "RPM" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["RPM"], mode="lines",
                line=dict(width=1.5, color=TRACE_COLORS["RPM"]),
                name="RPM",
                hovertemplate="RPM: %{y:,.0f}<extra></extra>",
            ), row=2, col=1)

        if "Speed" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["Speed"], mode="lines",
                line=dict(width=1.5, color=TRACE_COLORS["Speed"]),
                name="Speed",
                hovertemplate="Speed: %{y:.0f} km/h<extra></extra>",
            ), row=3, col=1)

        if "Throttle" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["Throttle"], mode="lines",
                line=dict(width=1.5, color=TRACE_COLORS["Throttle"]),
                name="Throttle",
                hovertemplate="Throttle: %{y:.0f}%<extra></extra>",
            ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=tel["Brake"].astype(int) * 100, mode="lines",
            line=dict(width=1.5, shape="hv", color=TRACE_COLORS["Brake (on/off)"]),
            name="Brake (on/off)",
            hovertemplate="Brake: %{y:.0f}%<extra></extra>",
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=tel["BrakeIntensity"], mode="lines",
            line=dict(width=1.5, color=TRACE_COLORS["Brake Intensity"]),
            name="Brake Intensity (est.)",
            hovertemplate="Brake Intensity: %{y:.0f}<extra></extra>",
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=tel["LongAccelG"], mode="lines",
            line=dict(width=1.5, color=TRACE_COLORS["LongAccelG"]),
            name="Long. Accel (g)",
            hovertemplate="Accel: %{y:.2f} g<extra></extra>",
        ), row=5, col=1)

        if "DRS" in tel.columns:
            fig.add_trace(go.Scatter(
                x=x, y=tel["DRS_on"], mode="lines",
                line=dict(width=1.5, shape="hv", color=TRACE_COLORS["DRS"]),
                name="DRS (on)",
                hovertemplate="DRS: %{y}<extra></extra>",
            ), row=6, col=1)

        # Corner markers (distance mode)
        if circuit_info is not None and x_mode == "distance" and "Distance" in tel.columns:
            try:
                corners = circuit_info.corners.dropna(subset=["Distance"])
                for _, c in corners.iterrows():
                    fig.add_vline(
                        x=float(c["Distance"]),
                        line_dash="dot", line_width=1, opacity=0.2,
                        row=3, col=1,
                    )
            except Exception:
                pass

        # Braking onset markers on speed row
        onset_x = x.loc[brake_onsets] if len(brake_onsets) else pd.Series([], dtype=float)
        if len(onset_x):
            for bx in onset_x.iloc[:200]:
                fig.add_vline(
                    x=float(bx),
                    line_dash="dash", line_width=1, opacity=0.15,
                    line_color="#e10600",
                    row=3, col=1,
                )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            font=dict(family="system-ui, -apple-system, sans-serif", color="#9ca3af", size=11),
            margin=dict(l=50, r=15, t=35, b=35),
            title=dict(text=f"{driver} — Lap {lap_number}", x=0.5, font=dict(size=14, color="#fff")),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#333", font_size=12),
            showlegend=False,
        )

        # Style subplot titles
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=10, color="#6b7280"))

        fig.update_xaxes(
            title_text=x_title, row=6, col=1,
            showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.05)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikedash="solid", spikethickness=1, spikecolor="#555",
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
        )

        # ---- Track map figure ----
        map_fig = go.Figure()
        if "X" in tel.columns and "Y" in tel.columns:
            color_series = tel[map_color] if map_color in tel.columns else tel.get("Speed", pd.Series(np.nan, index=tel.index))

            map_fig.add_trace(go.Scattergl(
                x=tel["X"], y=tel["Y"],
                mode="lines+markers",
                marker=dict(
                    size=3,
                    color=color_series,
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=map_color, font=dict(size=11, color="#9ca3af")),
                        len=0.75,
                        thickness=12,
                        tickfont=dict(size=10, color="#6b7280"),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    ),
                ),
                line=dict(width=3, color="rgba(100,100,100,0.3)"),
                hovertemplate=f"{map_color}: " + "%{marker.color:.1f}<extra></extra>",
                name="Track",
            ))

            # Braking onsets on map
            if len(brake_onsets):
                map_fig.add_trace(go.Scattergl(
                    x=tel.loc[brake_onsets, "X"],
                    y=tel.loc[brake_onsets, "Y"],
                    mode="markers",
                    marker=dict(size=7, color="#e10600", symbol="diamond"),
                    name="Brake onsets",
                    hovertemplate="Brake point<extra></extra>",
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
                        textfont=dict(size=10, color="#9ca3af"),
                        marker=dict(size=6, color="#ffffff", opacity=0.7),
                        name="Corners",
                        hovertemplate="Turn %{text}<extra></extra>",
                    ))
                except Exception:
                    pass

        # Replay cursor dot (last trace — animation will update this)
        if "X" in tel.columns and "Y" in tel.columns:
            map_fig.add_trace(go.Scattergl(
                x=[tel["X"].iloc[0]],
                y=[tel["Y"].iloc[0]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#e10600",
                    line=dict(width=2, color="#ffffff"),
                    symbol="circle",
                ),
                name="Cursor",
                hoverinfo="skip",
            ))

        map_fig.update_layout(
            template="plotly_dark",
            title=dict(text="Track Map", x=0.5, font=dict(size=14, color="#fff")),
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            font=dict(family="system-ui, -apple-system, sans-serif"),
        )

        # ---- Metrics ----
        m = lap_metrics(tel)
        if track_temp is not None:
            m["Track Temp"] = f"{track_temp:.1f} \u00b0C"
        if air_temp is not None:
            m["Air Temp"] = f"{air_temp:.1f} \u00b0C"
        if wind is not None:
            m["Wind"] = f"{wind:.1f} m/s"

        metrics_component = metric_cards(m)

        # ---- Lap info badges ----
        compound_color = COMPOUND_COLORS.get(str(compound).upper(), "secondary")

        badges = [
            dbc.Badge(format_lap_time(lap_time), color="light", text_color="dark",
                      className="info-badge", style={"fontWeight": "700", "fontSize": "0.85rem"}),
            dbc.Badge(str(compound).upper(), color=compound_color, className="info-badge"),
            dbc.Badge(f"Tyre: {int(tyre_life) if pd.notna(tyre_life) else 'N/A'} laps",
                      color="secondary", className="info-badge"),
            dbc.Badge(f"Stint {int(stint) if pd.notna(stint) else 'N/A'}",
                      color="secondary", className="info-badge"),
            dbc.Badge(status_txt, color=status_badge_color(status_txt), className="info-badge"),
        ]
        if is_pit_lap:
            badges.append(dbc.Badge("PIT", color="danger", className="info-badge",
                                    style={"fontWeight": "700"}))
        if track_temp is not None:
            badges.append(dbc.Badge(f"Track {track_temp:.0f}\u00b0C", color="dark",
                                    className="info-badge"))

        info_component = html.Div(badges, className="d-flex flex-wrap gap-1 align-items-center justify-content-end")

        return fig, map_fig, info_component, metrics_component

    except Exception as e:
        import traceback
        traceback.print_exc()
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
        )
        return empty_fig, empty_fig, f"Error: {e}", ""


# ---- Replay callbacks ----

SPEED_MAP = {
    "speed-025": 0.25,
    "speed-05":  0.5,
    "speed-1":   1.0,
    "speed-2":   2.0,
    "speed-4":   4.0,
}


@app.callback(
    Output("replay-tel-json", "data"),
    Output("replay-slider", "max"),
    Output("replay-total-label", "children"),
    Output("replay-index", "data", allow_duplicate=True),
    Output("replay-playing", "data", allow_duplicate=True),
    Output("replay-interval", "disabled", allow_duplicate=True),
    Output("play-btn", "children", allow_duplicate=True),
    Input("driver-dropdown", "value"),
    Input("lap-dropdown", "value"),
    State("x-axis-mode", "value"),
    prevent_initial_call=True,
)
def cache_replay_data(driver, lap_number, x_mode):
    """Serialize telemetry for the replay system whenever lap changes. Also resets playback."""
    if not driver or lap_number is None:
        return None, 100, "--s", 0, False, True, "\u25B6"

    try:
        _, tel = get_lap_and_tel(driver, int(lap_number))

        if x_mode == "time":
            x_vals = tel["t"].tolist()
        else:
            x_vals = tel["Distance"].tolist() if "Distance" in tel.columns else tel["t"].tolist()

        replay_data = {
            "x": x_vals,
            "t": tel["t"].tolist(),
            "X": tel["X"].tolist() if "X" in tel.columns else [],
            "Y": tel["Y"].tolist() if "Y" in tel.columns else [],
            "Speed": tel["Speed"].tolist() if "Speed" in tel.columns else [],
            "nGear": tel["nGear"].tolist() if "nGear" in tel.columns else [],
            "Throttle": tel["Throttle"].tolist() if "Throttle" in tel.columns else [],
            "Brake": tel["Brake"].astype(int).tolist(),
            "n_points": len(tel),
        }

        max_t = tel["t"].max()
        return replay_data, len(tel) - 1, f"{max_t:.1f}s", 0, False, True, "\u25B6"

    except Exception:
        return None, 100, "--s", 0, False, True, "\u25B6"


@app.callback(
    Output("replay-playing", "data"),
    Output("replay-interval", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("replay-playing", "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, is_playing):
    new_state = not is_playing
    icon = "\u23F8" if new_state else "\u25B6"
    return new_state, not new_state, icon


@app.callback(
    Output("replay-speed", "data"),
    Output("speed-025", "className"),
    Output("speed-05", "className"),
    Output("speed-1", "className"),
    Output("speed-2", "className"),
    Output("speed-4", "className"),
    [Input(sid, "n_clicks") for sid in SPEED_MAP],
    prevent_initial_call=True,
)
def set_speed(*n_clicks):
    trigger = ctx.triggered_id
    speed = SPEED_MAP.get(trigger, 1.0)

    classes = []
    for sid in SPEED_MAP:
        if sid == trigger:
            classes.append("speed-btn speed-btn-active")
        else:
            classes.append("speed-btn")

    return speed, *classes


@app.callback(
    Output("replay-index", "data"),
    Output("replay-slider", "value"),
    Output("replay-time-label", "children"),
    Input("replay-interval", "n_intervals"),
    State("replay-index", "data"),
    State("replay-tel-json", "data"),
    State("replay-speed", "data"),
    prevent_initial_call=True,
)
def animation_tick(n_intervals, current_index, tel_data, speed):
    if tel_data is None:
        raise PreventUpdate

    n_points = tel_data["n_points"]
    t_values = tel_data["t"]

    tick_duration = 0.05 * speed
    current_t = t_values[current_index]
    target_t = current_t + tick_duration

    new_index = current_index
    while new_index < n_points - 1 and t_values[new_index] < target_t:
        new_index += 1

    if new_index >= n_points - 1:
        new_index = 0

    new_t = t_values[new_index]
    return new_index, new_index, f"{new_t:.1f}s"


@app.callback(
    Output("telemetry-graph", "figure", allow_duplicate=True),
    Output("track-map", "figure", allow_duplicate=True),
    Output("live-readout", "children"),
    Input("replay-index", "data"),
    State("replay-tel-json", "data"),
    State("replay-playing", "data"),
    prevent_initial_call=True,
)
def update_cursor(index, tel_data, is_playing):
    if tel_data is None:
        raise PreventUpdate

    x_val = tel_data["x"][index]
    map_x = tel_data["X"][index] if tel_data["X"] else None
    map_y = tel_data["Y"][index] if tel_data["Y"] else None

    # Telemetry cursor: vertical line across all subplots
    tel_patch = Patch()
    tel_patch["layout"]["shapes"] = [{
        "type": "line",
        "x0": x_val, "x1": x_val,
        "y0": 0, "y1": 1,
        "xref": "x6",
        "yref": "paper",
        "line": {"color": "#e10600", "width": 2, "dash": "solid"},
        "opacity": 0.8,
    }]

    # Track map cursor: update the last trace (cursor dot)
    map_patch = Patch()
    if map_x is not None and map_y is not None:
        map_patch["data"][-1]["x"] = [map_x]
        map_patch["data"][-1]["y"] = [map_y]

    # Live readout
    speed = tel_data["Speed"][index] if tel_data["Speed"] else 0
    gear = tel_data["nGear"][index] if tel_data["nGear"] else 0
    throttle = tel_data["Throttle"][index] if tel_data["Throttle"] else 0
    brake = tel_data["Brake"][index]

    readout = html.Div([
        html.Div([html.Div(f"{speed:.0f}", className="readout-value"),
                  html.Div("km/h", className="readout-label")], className="readout-item"),
        html.Div([html.Div(f"{int(gear)}", className="readout-value"),
                  html.Div("gear", className="readout-label")], className="readout-item"),
        html.Div([html.Div(f"{throttle:.0f}%", className="readout-value"),
                  html.Div("throttle", className="readout-label")], className="readout-item"),
        html.Div([html.Div("ON" if brake else "OFF", className="readout-value",
                           style={"color": "#e10600"} if brake else {}),
                  html.Div("brake", className="readout-label")], className="readout-item"),
    ], className="live-readout")

    return tel_patch, map_patch, readout


@app.callback(
    Output("replay-index", "data", allow_duplicate=True),
    Input("replay-slider", "value"),
    State("replay-playing", "data"),
    prevent_initial_call=True,
)
def slider_seek(slider_value, is_playing):
    if ctx.triggered_id != "replay-slider":
        raise PreventUpdate
    return int(slider_value)


# ---- Chat callbacks ----

@app.callback(
    Output("chat-panel", "style"),
    Output("chat-panel-open", "data"),
    Input("chat-toggle-btn", "n_clicks"),
    Input("chat-close-btn", "n_clicks"),
    State("chat-panel-open", "data"),
    prevent_initial_call=True,
)
def toggle_chat(toggle_clicks, close_clicks, is_open):
    trigger = ctx.triggered_id
    if trigger == "chat-close-btn":
        return {"display": "none"}, False
    new_state = not is_open
    if new_state:
        return {"display": "flex", "flexDirection": "column"}, True
    return {"display": "none"}, False


@app.callback(
    Output("chat-messages", "children"),
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("driver-dropdown", "value"),
    State("lap-dropdown", "value"),
    prevent_initial_call=True,
)
def handle_chat_message(send_clicks, enter_submit, message, history, driver, lap):
    if not message or not message.strip():
        raise PreventUpdate

    history = history or []
    history.append({"role": "user", "text": message.strip()})

    response = generate_mock_response(message.strip(), driver, lap)
    history.append({"role": "ai", "text": response})

    # Build message components — include welcome message first
    msg_components = [
        html.Div(
            "Hey! I'm your Race Engineer assistant. I can see the telemetry "
            "you're looking at right now. Ask me about braking zones, speed traces, "
            "throttle application — or type 'help' to see what I can do.",
            className="chat-msg chat-msg-ai",
        ),
    ]
    for msg in history:
        cls = "chat-msg chat-msg-user" if msg["role"] == "user" else "chat-msg chat-msg-ai"
        msg_components.append(html.Div(msg["text"], className=cls))

    return msg_components, history, ""


if __name__ == "__main__":
    print("Starting dashboard: http://127.0.0.1:8050")
    # IMPORTANT: use_reloader=False prevents loading the FastF1 session twice in debug mode
    app.run(debug=True, port=8050, use_reloader=False)
