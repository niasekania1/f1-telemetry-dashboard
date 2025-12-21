#!/usr/bin/env python3
"""
Download Barcelona F1 race data and extract all car stats to CSV
"""

import fastf1
import pandas as pd
from datetime import datetime

# Configure FastF1
fastf1.Cache.enable_cache('./cache')

def download_barcelona_data():
    """
    Download Barcelona F1 race data for 2024
    """
    print("Starting Barcelona F1 data download...")
    
    year = 2024
    
    try:
        print(f"Loading {year} Spanish Grand Prix...")
        
        session = fastf1.get_session(year, 'Spanish', 'R')
        session.load()
        
        print("Session loaded successfully!")
        print(f"Event: {session.event['EventName']}")
        
        # Get all laps data
        laps = session.laps
        
        # Get driver information
        drivers = session.drivers
        driver_info_dict = {}
        
        for driver in drivers:
            info = session.get_driver(driver)
            # Map both the driver number and abbreviation to the same info
            driver_info_dict[driver] = {
                'Abbreviation': info['Abbreviation'],
                'FirstName': info['FirstName'],
                'LastName': info['LastName'],
                'FullName': f"{info['FirstName']} {info['LastName']}",
                'TeamName': info['TeamName']
            }
            # Also add by abbreviation for easy lookup
            driver_info_dict[info['Abbreviation']] = driver_info_dict[driver]
        
        print(f"Found {len(drivers)} drivers")
        
        # Extract comprehensive car statistics
        car_data = []
        telemetry_errors = 0
        
        for idx, lap in laps.iterrows():
            driver_id = lap['Driver']
            if driver_id not in driver_info_dict:
                continue
                
            info = driver_info_dict[driver_id]
            
            # Skip laps without valid lap times
            if pd.isna(lap['LapTime']):
                continue
            
            # Get telemetry data for this lap
            avg_speed = avg_rpm = avg_throttle = avg_brake = avg_gear = avg_drs = max_speed = max_rpm = None
            
            try:
                telemetry = lap.get_car_data()
                if len(telemetry) > 0:
                    # Calculate averages from telemetry
                    if 'Speed' in telemetry.columns:
                        avg_speed = telemetry['Speed'].mean()
                        max_speed = telemetry['Speed'].max()
                    if 'RPM' in telemetry.columns:
                        avg_rpm = telemetry['RPM'].mean()
                        max_rpm = telemetry['RPM'].max()
                    if 'Throttle' in telemetry.columns:
                        avg_throttle = telemetry['Throttle'].mean()
                    if 'Brake' in telemetry.columns:
                        avg_brake = telemetry['Brake'].mean()
                    if 'nGear' in telemetry.columns:
                        avg_gear = telemetry['nGear'].mean()
                    if 'DRS' in telemetry.columns:
                        avg_drs = telemetry['DRS'].mean()
            except Exception as e:
                telemetry_errors += 1
            
            # Build comprehensive data record
            record = {
                # Basic Info
                'Year': year,
                'Event': session.event['EventName'],
                
                # Driver Info
                'Driver_ID': driver_id,
                'Driver_Abbreviation': info['Abbreviation'],
                'Driver_Full_Name': info['FullName'],
                'Team': info['TeamName'],
                
                # Lap Info
                'Lap_Number': lap['LapNumber'],
                'Lap_Time': str(lap['LapTime']),
                'Lap_Time_Seconds': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
                
                # Sector Times
                'Sector1_Time': str(lap['Sector1Time']) if pd.notna(lap['Sector1Time']) else None,
                'Sector2_Time': str(lap['Sector2Time']) if pd.notna(lap['Sector2Time']) else None,
                'Sector3_Time': str(lap['Sector3Time']) if pd.notna(lap['Sector3Time']) else None,
                
                # Tyre Data
                'Compound': lap['Compound'],
                'Tyre_Life': lap['TyreLife'],
                'Fresh_Tyre': lap['FreshTyre'],
                'Stint': lap['Stint'],
                'Lap_In_Stint': None,
                
                # Track Status
                'Track_Status': lap['TrackStatus'],
                'Is_Pit_Lap': lap['PitInTime'] is not pd.NaT or lap['PitOutTime'] is not pd.NaT,
                'Pit_In_Time': str(lap['PitInTime']) if pd.notna(lap['PitInTime']) else None,
                'Pit_Out_Time': str(lap['PitOutTime']) if pd.notna(lap['PitOutTime']) else None,
                
                # Speed Data
                'Speed_I1': lap['SpeedI1'] if pd.notna(lap['SpeedI1']) else None,
                'Speed_I2': lap['SpeedI2'] if pd.notna(lap['SpeedI2']) else None,
                'Speed_FL': lap['SpeedFL'] if pd.notna(lap['SpeedFL']) else None,
                'Speed_ST': lap['SpeedST'] if pd.notna(lap['SpeedST']) else None,
                
                # Telemetry Averages
                'Avg_Speed': avg_speed,
                'Avg_RPM': avg_rpm,
                'Avg_Throttle': avg_throttle,
                'Avg_Brake': avg_brake,
                'Avg_Gear': avg_gear,
                'Avg_DRS': avg_drs,
                
                # Telemetry Max
                'Max_Speed': max_speed,
                'Max_RPM': max_rpm,
                
                # Timing
                'Lap_Start_Time': str(lap['LapStartTime']) if pd.notna(lap['LapStartTime']) else None,
                'Time_from_Leader': str(lap['Time']) if pd.notna(lap['Time']) else None,
            }
            
            car_data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(car_data)
        
        if len(df) == 0:
            print("No data extracted!")
            return None
        
        # Sort by driver and lap number
        df = df.sort_values(['Driver_Abbreviation', 'Lap_Number'])
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'barcelona_f1_2024_complete_stats_{timestamp}.csv'
        
        df.to_csv(filename, index=False)
        
        # Print summary
        print(f"\n✅ SUCCESS! Saved {len(df)} rows to {filename}")
        print(f"\n📊 SUMMARY:")
        print(f"   • Total laps: {len(df)}")
        print(f"   • Unique drivers: {len(df['Driver_Abbreviation'].unique())}")
        print(f"   • Teams: {len(df['Team'].unique())}")
        print(f"   • Event: {df['Event'].iloc[0]}")
        if telemetry_errors > 0:
            print(f"   • Telemetry errors: {telemetry_errors}")
        
        print(f"\n📋 DRIVERS:")
        for driver in sorted(df['Driver_Abbreviation'].unique()):
            laps_count = len(df[df['Driver_Abbreviation'] == driver])
            team = df[df['Driver_Abbreviation'] == driver]['Team'].iloc[0]
            print(f"   • {driver} ({team}): {laps_count} laps")
        
        print(f"\n📄 COLUMNS ({len(df.columns)} total):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    download_barcelona_data()