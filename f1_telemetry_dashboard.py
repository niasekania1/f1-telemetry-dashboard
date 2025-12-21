#!/usr/bin/env python3
"""
F1 Telemetry Dashboard - Interactive visualization of Barcelona F1 race data
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
import fastf1

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

# File path - CSV is in the same directory
CSV_FILE = "barcelona_f1_2024_complete_stats_20251221_115840.csv"

def load_data():
    """Load and prepare the F1 telemetry data"""
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return pd.DataFrame()
    
    df = pd.read_csv(CSV_FILE)
    # Convert Lap_Number to int for better sorting
    df['Lap_Number'] = df['Lap_Number'].astype(int)
    return df

# Load data
df = load_data()

if df.empty:
    print("No data available. Please run download_barcelona_f1_data.py first.")
    exit(1)

print(f"Loaded {len(df)} rows of telemetry data")
print(f"Drivers: {sorted(df['Driver_Abbreviation'].unique())}")

# Load the session for detailed telemetry
print("Loading session data...")
session = fastf1.get_session(2024, 'Spanish Grand Prix', 'R')
session.load()
print("Session data loaded!")

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("F1 Telemetry Visualizer - Barcelona 2024", className="text-center my-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Driver:"),
            dcc.Dropdown(
                id='driver-dropdown',
                options=[{'label': name, 'value': abbr} for name, abbr in zip(sorted(df['Driver_Full_Name'].unique()), sorted(df['Driver_Abbreviation'].unique()))],
                value=df['Driver_Abbreviation'].unique()[0] if not df.empty else None,
                clearable=False,
                style={'color': '#000', 'backgroundColor': '#fff'}
            )
        ], width=4),
        dbc.Col([
            html.Label("Select Lap:"),
            dcc.Dropdown(
                id='lap-dropdown',
                clearable=False,
                style={'color': '#000', 'backgroundColor': '#fff'}
            )
        ], width=4),
        dbc.Col([
            html.Label("Telemetry Type:"),
            dcc.Dropdown(
                id='telemetry-type',
                options=[
                    {'label': 'Full Telemetry', 'value': 'full'},
                    {'label': 'Speed & RPM Only', 'value': 'speed_rpm'},
                    {'label': 'Gears & Throttle', 'value': 'gears_throttle'}
                ],
                value='full',
                clearable=False,
                style={'color': '#000', 'backgroundColor': '#fff'}
            )
        ], width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='telemetry-graph', style={'height': '85vh'}), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='lap-info', className="text-center mt-3")
        ], width=12)
    ])
], fluid=True)

@app.callback(
    Output('lap-dropdown', 'options'),
    Output('lap-dropdown', 'value'),
    Input('driver-dropdown', 'value')
)
def update_lap_dropdown(selected_driver):
    """Update lap dropdown based on selected driver"""
    if not selected_driver:
        return [], None
    
    driver_laps = df[df['Driver_Abbreviation'] == selected_driver]['Lap_Number'].unique()
    options = [{'label': f"Lap {int(lap)}", 'value': lap} for lap in sorted(driver_laps)]
    
    return options, options[0]['value'] if options else None

@app.callback(
    Output('telemetry-graph', 'figure'),
    Output('lap-info', 'children'),
    Input('driver-dropdown', 'value'),
    Input('lap-dropdown', 'value'),
    Input('telemetry-type', 'value')
)
def update_graph(selected_driver, selected_lap, telemetry_type):
    """Update the telemetry graph based on selections"""
    if not selected_driver or selected_lap is None:
        return go.Figure(), "Select a driver and lap to view telemetry"

    try:
        # Get the lap telemetry data from FastF1
        driver_laps = session.laps.pick_driver(selected_driver)
        lap = driver_laps[driver_laps['LapNumber'] == selected_lap].iloc[0]
        telemetry = lap.get_telemetry()
        
        if len(telemetry) == 0:
            return go.Figure(), "No telemetry data available for this lap"
        
        # Get lap info from CSV
        lap_info_data = df[(df['Driver_Abbreviation'] == selected_driver) & (df['Lap_Number'] == selected_lap)]
        if len(lap_info_data) > 0:
            lap_time = lap_info_data['Lap_Time'].iloc[0]
            compound = lap_info_data['Compound'].iloc[0]
            tyre_life = lap_info_data['Tyre_Life'].iloc[0]
        else:
            lap_time = lap['LapTime']
            compound = lap['Compound']
            tyre_life = lap['TyreLife']
        
        # Create subplots based on telemetry type
        if telemetry_type == 'full':
            rows = 5
            titles = ("Gear", "RPM", "Speed (km/h)", "Throttle & Brake", "DRS")
            row_heights = [0.1, 0.2, 0.3, 0.3, 0.1]
        elif telemetry_type == 'speed_rpm':
            rows = 2
            titles = ("RPM", "Speed (km/h)")
            row_heights = [0.4, 0.6]
        else:  # gears_throttle
            rows = 3
            titles = ("Gear", "Throttle & Brake", "DRS")
            row_heights = [0.2, 0.6, 0.2]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=row_heights,
            subplot_titles=titles
        )

        # X-axis (Distance in meters)
        x_axis = telemetry['Distance']

        # Add traces based on telemetry type
        if telemetry_type == 'full':
            # Gear
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['nGear'], name="Gear", 
                                    line=dict(color='#00ffff', width=2, shape='hv'), mode='lines'), row=1, col=1)
            
            # RPM
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['RPM'], name="RPM", 
                                    line=dict(color='#ff0000', width=2), mode='lines'), row=2, col=1)
            
            # Speed
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['Speed'], name="Speed", 
                                    line=dict(color='#ffffff', width=2), mode='lines'), row=3, col=1)
            
            # Throttle & Brake - scale brake values to be visible
            brake_scaled = telemetry['Brake'] * 100 if telemetry['Brake'].max() <= 1 else telemetry['Brake']
            
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['Throttle'], name="Throttle",
                                    line=dict(color='#00ff00', width=2), mode='lines'), row=4, col=1)
            fig.add_trace(go.Scatter(x=x_axis, y=brake_scaled, name="Brake",
                                    line=dict(color='#ff0000', width=2), mode='lines'), row=4, col=1)
            
            # DRS
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['DRS'], name="DRS", 
                                    line=dict(color='#ffff00', width=2, shape='hv'), mode='lines'), row=5, col=1)
            
        elif telemetry_type == 'speed_rpm':
            # RPM
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['RPM'], name="RPM", 
                                    line=dict(color='#ff0000', width=2), mode='lines'), row=1, col=1)
            
            # Speed
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['Speed'], name="Speed", 
                                    line=dict(color='#ffffff', width=2), mode='lines'), row=2, col=1)
            
        else:  # gears_throttle
            # Gear
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['nGear'], name="Gear", 
                                    line=dict(color='#00ffff', width=2, shape='hv'), mode='lines'), row=1, col=1)
            
            # Throttle & Brake - scale brake values to be visible
            brake_scaled = telemetry['Brake'] * 100 if telemetry['Brake'].max() <= 1 else telemetry['Brake']
            
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['Throttle'], name="Throttle",
                                    line=dict(color='#00ff00', width=2), mode='lines'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_axis, y=brake_scaled, name="Brake",
                                    line=dict(color='#ff0000', width=2), mode='lines'), row=2, col=1)
            
            # DRS
            fig.add_trace(go.Scatter(x=x_axis, y=telemetry['DRS'], name="DRS", 
                                    line=dict(color='#ffff00', width=2, shape='hv'), mode='lines'), row=3, col=1)

        # Update layout to match Wintax style
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=60, r=20, t=60, b=40),
            hovermode="x unified",
            title=f"Telemetry: {selected_driver} - Lap {selected_lap}",
            title_x=0.5,
            plot_bgcolor='black',
            paper_bgcolor='black'
        )

        # Update grid styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333', zeroline=False)
        
        # Add x-axis label only to the bottom subplot
        fig.update_xaxes(title_text="Distance (m)", row=rows, col=1)
        
        # Update subplot title positioning to avoid overlap
        for annotation in fig['layout']['annotations']:
            annotation['y'] = annotation['y'] + 0.01
        
        # Update y-axis labels
        if telemetry_type == 'full':
            fig.update_yaxes(title_text="Gear", row=1, col=1)
            fig.update_yaxes(title_text="RPM", row=2, col=1)
            fig.update_yaxes(title_text="km/h", row=3, col=1)
            fig.update_yaxes(title_text="%", row=4, col=1)
            fig.update_yaxes(title_text="DRS", row=5, col=1)
        elif telemetry_type == 'speed_rpm':
            fig.update_yaxes(title_text="RPM", row=1, col=1)
            fig.update_yaxes(title_text="km/h", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Gear", row=1, col=1)
            fig.update_yaxes(title_text="%", row=2, col=1)
            fig.update_yaxes(title_text="DRS", row=3, col=1)

        # Lap info text
        info_text = f"Lap Time: {lap_time} | Compound: {compound} | Tyre Life: {int(tyre_life)} laps"
        
        return fig, info_text
        
    except Exception as e:
        return go.Figure(), f"Error loading telemetry: {str(e)}"

if __name__ == '__main__':
    print("Starting F1 Telemetry Dashboard...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    app.run(debug=True, port=8050)
