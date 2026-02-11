# Project Documentation: Automated Telemetry Insight System (ATIS)

**Objective:** To solve the "Data Illiteracy" problem in junior motorsport (Formula 4) by using Python and Generative AI to translate complex telemetry signals into actionable coaching feedback.

## 1. Current Status: Proof of Concept (Phase 1)

I have successfully built a functional telemetry dashboard using public Formula 1 data as a proxy for the target Formula 4 environment.

- **Data Source:** FastF1 API (2024 Barcelona Grand Prix data).
- **Tech Stack:**
  - **Backend:** Python, Pandas, NumPy.
  - **Frontend/Viz:** Plotly Dash, Dash Bootstrap Components (running on local server 127.0.0.1:8050).
  - **Signal Processing:** Initial implementation of **Dynamic Time Warping (DTW)** to mathematically align laps of different lengths/durations for accurate comparison.
- **Current Features:**
  - **Multi-Channel Visualization:** Synchronization of Gear, RPM, Speed, Throttle, Brake Intensity, and Longitudinal G-forces.
  - **Interactive Track Map:** GPS-based map with speed-gradient coloring and corner numbering.
  - **Automated Statistics:** Instant calculation of Top Speed, Max Deceleration (G), Full Throttle %, and Coasting %.
  - **Distance-Based Analysis:** Graphs are plotted against distance (meters) rather than time, allowing for direct corner-by-corner comparison.

## 2. The Development Gap (The "Goal")

While the current system shows *what* happened, it does not yet explain *how* to improve. The next phase involves shifting from **Data Visualization** to **Automated Coaching.**

### A. The Generative AI Pipeline (LLM Integration)

- **The Problem:** Drivers can see they are slower in a corner but don't know why (e.g., is it braking too early, or a lazy throttle application?).
- **The Solution:** I will feed the "Vector Differences" (the gap between a Student lap and a Reference lap) into a Large Language Model (LLM).
- **Translation Layer:** The system will convert numerical data (e.g., *Speed Delta: -5km/h at 1200m*) into Natural Language (e.g., *"You are over-braking in Turn 3; try to carry 5km/h more entry speed to stabilize the car"*).

### B. Transition to Proprietary F4 Data (WinTax Integration)

- **Data Ingestion:** Move away from FastF1 and develop a custom parser for **Magneti Marelli WinTax** ASCII/CSV exports.
- **Normalization:** Mapping professional-grade F4 sensor data (Brake Pressure in Bar, Steering Angle in Degrees) to a standardized format for the AI to process.

### C. UX Refinement

- **"Deep-Dive" Interaction:** Enhancing the dashboard so that hovering or clicking a specific corner triggers a "Coaching Pop-up" powered by the AI.

## 3. Immediate Roadmap & Research Questions

To complete the thesis, I will focus on the following milestones:

1. **Refine the Comparative Model:** Build the "Two-Driver Overlay" in the dashboard to visualize the delta between a Pro and a Junior.
2. **Prompt Engineering:** Develop a system prompt that ensures the AI speaks like a "Race Engineer" (using terms like *Trail-braking, Apex, and Exit-phase*) rather than a generic chatbot.
3. **Field Validation (February 2025):** Test the tool with real F4 drivers and engineers at an upcoming race event to measure if the AI insights help the driver learn faster than traditional graph-reading.

## 4. Academic Contribution

This project contributes to the field of **Sports Data Literacy** by investigating whether automated natural-language feedback reduces the cognitive load on young athletes, allowing them to focus on performance rather than data interpretation.
