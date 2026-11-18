"""Hardcoded research content, benchmark tables, GIS reference tables, and Google SVG for RAKSHA AI."""

GOOGLE_G_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="20" height="20" style="vertical-align:middle;margin-right:8px;">
<path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/>
<path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/>
<path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/>
<path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/>
</svg>"""

HEADLINE_METRICS = [
    ("304,988", "Real Road-User Pairs", "4,988 Argoverse 2 scenarios • 1,956,954 anchors"),
    ("0.966", "XGBoost Real AUC", "+0.221 over TTC threshold rule (0.745)"),
    ("57x", "AP Lift vs Random", "0.067 AP vs 0.016 TTC rule (0.12% positives)"),
    ("0.938–0.986", "3 s Lead-Time AUC", "0.938 at 2–3 s before near-miss onset"),
    ("217,160", "Crash Records", "210,300 real • 296 blackspots • 100 emerging zones"),
]

STORY_STEPS = [
    (
        "1. Watch",
        "10 Hz Trajectories",
        "Ingests real Argoverse 2 motion-forecasting scenarios (4,988 scenarios across 6 US cities) alongside "
        "a 6,000-scenario mixed-traffic simulator and a YOLO + ByteTrack video tracking module.",
    ),
    (
        "2. Measure",
        "27 Causal Features",
        "At each 0.5 s anchor t, summarises strictly past frames [t - 1.5 s, t] into 6 surrogate safety measures "
        "(TTC, projected PET, DRAC), 11 kinematic features, and 10 context flags.",
    ),
    (
        "3. Predict",
        "3 s Look-Ahead Onset",
        "XGBoost predicts whether physical clearance will fall below 1.0 m within the next 3 s. Scenario-wise "
        "splits and onset filtering eliminate leakage; TreeSHAP explains every alert.",
    ),
    (
        "4. Act",
        "GIS & Authority Triage",
        "Aggregates 217,160 crash records onto a 150 m grid (score = 1 + 2·injured + 10·killed), flags 296 "
        "blackspots, clusters alerts into 100 emerging risk zones, and requires authority review.",
    ),
]

DATASET_PROVENANCE = [
    {"Data Stream": "Crash records for blackspots", "Source": "NYC Open Data, City of Chicago, UK DfT STATS19", "Volume": "210,300 crashes", "Status": "Real"},
    {"Data Stream": "India, eight cities", "Source": "OpenStreetMap junctions + synthetic counts", "Volume": "6,860 crashes", "Status": "Simulated counts on real coords"},
    {"Data Stream": "Trajectories for main ML result", "Source": "Argoverse 2 Motion Forecasting (val split, CC BY-NC-SA 4.0)", "Volume": "4,988 scenarios / 304,988 pairs", "Status": "Real"},
    {"Data Stream": "Trajectories for method development", "Source": "src/simulate.py (IDM car-following + crossing conflicts)", "Volume": "6,000 scenarios / 108,659 anchors", "Status": "Simulated"},
    {"Data Stream": "Map alerts, emerging zones, reports", "Source": "src/build_gis.py anchored to real cities", "Volume": "4,000 alerts / 500 reports", "Status": "Simulated demonstration"},
    {"Data Stream": "DoTA, IDD, highD", "Source": "Benchmark video/trajectory datasets", "Volume": "Not used (55 GB / reg / request)", "Status": "Not used"},
]

CRASH_SOURCES_TABLE = [
    {"Source": "New York City (NYC Open Data)", "Records": "48,782", "Period": "Apr–Nov 2026", "Injured": "30,995", "Killed": "131", "Status": "Real"},
    {"Source": "Chicago (Chicago Data Portal)", "Records": "59,995", "Period": "Jan–Jul 2023", "Injured": "12,181", "Killed": "84", "Status": "Real"},
    {"Source": "Great Britain (UK DfT STATS19)", "Records": "101,523", "Period": "Calendar 2025", "Injured": "125,503", "Killed": "2,378", "Status": "Real"},
    {"Source": "Eight Indian Cities (OSM Junctions)", "Records": "6,860", "Period": "2023–2025 (sim)", "Injured": "—", "Killed": "—", "Status": "Simulated counts"},
    {"Source": "Total", "Records": "217,160", "Period": "Multi-region", "Injured": "168,679+ (real)", "Killed": "2,593+ (real)", "Status": "210,300 Real"},
]

# Hardcoded benchmark tables (Argoverse 2 Real Traffic & Simulated Traffic)
AV2_LEADERBOARD = [
    {"Model": "TTC threshold rule", "AUC": 0.745, "AP": 0.016, "F1": 0.064, "Notes": "13x random AP"},
    {"Model": "LogReg (TTC only)", "AUC": 0.745, "AP": 0.016, "F1": 0.064, "Notes": "Linear TTC baseline"},
    {"Model": "RandomForest (TTC only)", "AUC": 0.704, "AP": 0.018, "F1": 0.057, "Notes": "Tree TTC baseline"},
    {"Model": "XGBoost (TTC only)", "AUC": 0.747, "AP": 0.022, "F1": 0.074, "Notes": "Boosted TTC baseline"},
    {"Model": "LogReg (all 27 features)", "AUC": 0.952, "AP": 0.026, "F1": 0.063, "Notes": "Full causal features"},
    {"Model": "RandomForest (all 27 features)", "AUC": 0.960, "AP": 0.059, "F1": 0.099, "Notes": "100 trees, balanced"},
    {"Model": "XGBoost (all 27 features)", "AUC": 0.966, "AP": 0.067, "F1": 0.122, "Notes": "57x random AP (Best)"},
]

SIM_LEADERBOARD = [
    {"Model": "TTC threshold rule", "AUC": 0.834, "AP": 0.625, "F1": 0.640},
    {"Model": "LogReg (TTC only)", "AUC": 0.834, "AP": 0.625, "F1": 0.640},
    {"Model": "RandomForest (TTC only)", "AUC": 0.851, "AP": 0.536, "F1": 0.585},
    {"Model": "XGBoost (TTC only)", "AUC": 0.872, "AP": 0.611, "F1": 0.637},
    {"Model": "LogReg (all 27 features)", "AUC": 0.981, "AP": 0.793, "F1": 0.718},
    {"Model": "RandomForest (all 27 features)", "AUC": 0.993, "AP": 0.914, "F1": 0.842},
    {"Model": "XGBoost (all 27 features)", "AUC": 0.993, "AP": 0.924, "F1": 0.843},
]

AV2_CONFUSION = [
    {
        "True Negatives": "570,902",
        "False Positives": "2,525",
        "False Negatives": "445",
        "True Positives": "207",
        "Precision": "7.6%",
        "Recall": "31.7%",
        "False-Alert Rate": "0.44%",
        "F1": "0.122",
    }
]

LEAD_TIME_ONSET = [
    {"Time Until Onset": "0–1 s", "Positive Anchors": 238, "XGBoost AUC": 0.986, "TTC Rule AUC": 0.814, "Gain": "+0.172"},
    {"Time Until Onset": "1–2 s", "Positive Anchors": 213, "XGBoost AUC": 0.969, "TTC Rule AUC": 0.715, "Gain": "+0.254"},
    {"Time Until Onset": "2–3 s", "Positive Anchors": 201, "XGBoost AUC": 0.938, "TTC Rule AUC": 0.693, "Gain": "+0.245"},
]

LOOKAHEAD_WINDOW = [
    {"Look-Ahead L": "1 s", "Real XGBoost AUC": 0.993, "Real TTC Rule AUC": 0.814, "Sim XGBoost AUC": 1.000, "Sim TTC Rule AUC": 0.846},
    {"Look-Ahead L": "2 s", "Real XGBoost AUC": 0.980, "Real TTC Rule AUC": 0.767, "Sim XGBoost AUC": 0.998, "Sim TTC Rule AUC": 0.844},
    {"Look-Ahead L": "3 s", "Real XGBoost AUC": 0.966, "Real TTC Rule AUC": 0.745, "Sim XGBoost AUC": 0.993, "Sim TTC Rule AUC": 0.834},
    {"Look-Ahead L": "4 s", "Real XGBoost AUC": 0.954, "Real TTC Rule AUC": 0.731, "Sim XGBoost AUC": 0.986, "Sim TTC Rule AUC": 0.822},
    {"Look-Ahead L": "5 s", "Real XGBoost AUC": 0.940, "Real TTC Rule AUC": 0.721, "Sim XGBoost AUC": 0.978, "Sim TTC Rule AUC": 0.811},
]

ABLATION_TABLE = [
    {"Feature Set": "TTC", "Real AUC": 0.804, "Real AP": 0.023, "Sim AUC": 0.887, "Sim AP": 0.628},
    {"Feature Set": "+ PET", "Real AUC": 0.840, "Real AP": 0.025, "Sim AUC": 0.936, "Sim AP": 0.750},
    {"Feature Set": "+ DRAC", "Real AUC": 0.901, "Real AP": 0.042, "Sim AUC": 0.977, "Sim AP": 0.826},
    {"Feature Set": "+ kinematics", "Real AUC": 0.964, "Real AP": 0.062, "Sim AUC": 0.991, "Sim AP": 0.901},
    {"Feature Set": "+ road-user context", "Real AUC": 0.966, "Real AP": 0.067, "Sim AUC": 0.993, "Sim AP": 0.924},
]

SHAP_TOP8 = [
    {"Rank": 1, "Argoverse 2 (Real)": "clr_last", "Real Mean |SHAP|": 1.83, "Simulated Traffic": "drac_last", "Sim Mean |SHAP|": 1.90},
    {"Rank": 2, "Argoverse 2 (Real)": "drac_last", "Real Mean |SHAP|": 0.67, "Simulated Traffic": "pet_min", "Sim Mean |SHAP|": 0.71},
    {"Rank": 3, "Argoverse 2 (Real)": "ttc_min", "Real Mean |SHAP|": 0.52, "Simulated Traffic": "clr_last", "Sim Mean |SHAP|": 0.66},
    {"Rank": 4, "Argoverse 2 (Real)": "dist_last", "Real Mean |SHAP|": 0.52, "Simulated Traffic": "speed_b", "Sim Mean |SHAP|": 0.55},
    {"Rank": 5, "Argoverse 2 (Real)": "closing_last", "Real Mean |SHAP|": 0.32, "Simulated Traffic": "ttc_last", "Sim Mean |SHAP|": 0.53},
    {"Rank": 6, "Argoverse 2 (Real)": "speed_a", "Real Mean |SHAP|": 0.18, "Simulated Traffic": "pet_last", "Sim Mean |SHAP|": 0.47},
    {"Rank": 7, "Argoverse 2 (Real)": "ttc_last", "Real Mean |SHAP|": 0.15, "Simulated Traffic": "dist_last", "Sim Mean |SHAP|": 0.37},
    {"Rank": 8, "Argoverse 2 (Real)": "speed_b", "Real Mean |SHAP|": 0.14, "Simulated Traffic": "tb_pedestrian", "Sim Mean |SHAP|": 0.35},
]

PAIR_RATES_AV2 = [
    {"Pair (A vs B)": "bus vs bus", "Near-Miss Rate (%)": 5.9, "Pairs": 118},
    {"Pair (A vs B)": "motorcycle vs pedestrian", "Near-Miss Rate (%)": 2.6, "Pairs": 1235},
    {"Pair (A vs B)": "bus vs motorcycle", "Near-Miss Rate (%)": 2.6, "Pairs": 78},
    {"Pair (A vs B)": "motorcycle vs motorcycle", "Near-Miss Rate (%)": 2.6, "Pairs": 156},
    {"Pair (A vs B)": "car vs motorcycle", "Near-Miss Rate (%)": 1.6, "Pairs": 4692},
    {"Pair (A vs B)": "bus vs car", "Near-Miss Rate (%)": 1.4, "Pairs": 4198},
    {"Pair (A vs B)": "car vs bus", "Near-Miss Rate (%)": 1.2, "Pairs": 2373},
    {"Pair (A vs B)": "motorcycle vs car", "Near-Miss Rate (%)": 1.0, "Pairs": 3423},
    {"Pair (A vs B)": "car vs car", "Near-Miss Rate (%)": 0.5, "Pairs": 248067},
    {"Pair (A vs B)": "bus vs pedestrian", "Near-Miss Rate (%)": 0.3, "Pairs": 873},
    {"Pair (A vs B)": "car vs pedestrian", "Near-Miss Rate (%)": 0.3, "Pairs": 40300},
]

CITY_GRID_SUMMARY = [
    {"City": "Ahmedabad", "Country": "India", "Cells >= 1 crash": 234, "Blackspots": 2, "Max crashes in cell": 62, "Crash data": "Simulated counts"},
    {"City": "Bengaluru", "Country": "India", "Cells >= 1 crash": 221, "Blackspots": 2, "Max crashes in cell": 24, "Crash data": "Simulated counts"},
    {"City": "Chennai", "Country": "India", "Cells >= 1 crash": 243, "Blackspots": 2, "Max crashes in cell": 33, "Crash data": "Simulated counts"},
    {"City": "Delhi", "Country": "India", "Cells >= 1 crash": 243, "Blackspots": 2, "Max crashes in cell": 111, "Crash data": "Simulated counts"},
    {"City": "Hyderabad", "Country": "India", "Cells >= 1 crash": 228, "Blackspots": 2, "Max crashes in cell": 42, "Crash data": "Simulated counts"},
    {"City": "Kolkata", "Country": "India", "Cells >= 1 crash": 244, "Blackspots": 2, "Max crashes in cell": 30, "Crash data": "Simulated counts"},
    {"City": "Mumbai", "Country": "India", "Cells >= 1 crash": 226, "Blackspots": 2, "Max crashes in cell": 100, "Crash data": "Simulated counts"},
    {"City": "Pune", "Country": "India", "Cells >= 1 crash": 232, "Blackspots": 2, "Max crashes in cell": 28, "Crash data": "Simulated counts"},
    {"City": "Chicago", "Country": "USA", "Cells >= 1 crash": 14418, "Blackspots": 77, "Max crashes in cell": 139, "Crash data": "Real"},
    {"City": "New York", "Country": "USA", "Cells >= 1 crash": 16335, "Blackspots": 98, "Max crashes in cell": 47, "Crash data": "Real"},
    {"City": "Great Britain", "Country": "United Kingdom", "Cells >= 1 crash": 81028, "Blackspots": 105, "Max crashes in cell": 18, "Crash data": "Real"},
]

TOP_REAL_BLACKSPOTS = [
    {"City": "Chicago", "Area": "Chicago", "Crashes": 139, "Injured": 17, "Killed": 0, "Severity Score": 173},
    {"City": "New York", "Area": "Bronx", "Crashes": 46, "Injured": 54, "Killed": 0, "Severity Score": 154},
    {"City": "Chicago", "Area": "Chicago", "Crashes": 84, "Injured": 31, "Killed": 0, "Severity Score": 146},
    {"City": "Chicago", "Area": "Chicago", "Crashes": 94, "Injured": 23, "Killed": 0, "Severity Score": 140},
    {"City": "New York", "Area": "Bronx", "Crashes": 40, "Injured": 36, "Killed": 0, "Severity Score": 112},
    {"City": "Chicago", "Area": "Chicago", "Crashes": 58, "Injured": 26, "Killed": 0, "Severity Score": 110},
    {"City": "Chicago", "Area": "Chicago", "Crashes": 46, "Injured": 30, "Killed": 0, "Severity Score": 106},
    {"City": "New York", "Area": "Queens", "Crashes": 40, "Injured": 32, "Killed": 0, "Severity Score": 104},
]

FEATURE_DICTIONARY = [
    {"Feature": "ttc_min", "Group": "SSM", "Definition": "Minimum TTC over the 1.5 s window", "Unit": "s"},
    {"Feature": "ttc_last", "Group": "SSM", "Definition": "TTC at the anchor (30 s cap)", "Unit": "s"},
    {"Feature": "pet_min", "Group": "SSM", "Definition": "Minimum projected PET over the window", "Unit": "s"},
    {"Feature": "pet_last", "Group": "SSM", "Definition": "Projected PET at the anchor", "Unit": "s"},
    {"Feature": "drac_max", "Group": "SSM", "Definition": "Maximum DRAC over the window", "Unit": "m/s²"},
    {"Feature": "drac_last", "Group": "SSM", "Definition": "DRAC at the anchor", "Unit": "m/s²"},
    {"Feature": "dist_last", "Group": "Kinematics", "Definition": "Centre-to-centre distance at the anchor", "Unit": "m"},
    {"Feature": "clr_last", "Group": "Kinematics", "Definition": "Clearance (boundary-to-boundary gap, floored at 0.01 m)", "Unit": "m"},
    {"Feature": "closing_last", "Group": "Kinematics", "Definition": "Closing velocity (positive when approaching)", "Unit": "m/s"},
    {"Feature": "speed_a, speed_b", "Group": "Kinematics", "Definition": "Speed of road user A and B", "Unit": "m/s"},
    {"Feature": "rel_speed", "Group": "Kinematics", "Definition": "Absolute speed difference", "Unit": "m/s"},
    {"Feature": "acc_a, acc_b", "Group": "Kinematics", "Definition": "Acceleration of A and B at the anchor (backward difference)", "Unit": "m/s²"},
    {"Feature": "acc_a_min, acc_b_min", "Group": "Kinematics", "Definition": "Hardest braking (minimum acceleration) in the window", "Unit": "m/s²"},
    {"Feature": "jerk_a", "Group": "Kinematics", "Definition": "Mean absolute jerk of A over the window", "Unit": "m/s³"},
    {"Feature": "crossing", "Group": "Context", "Definition": "1 if paths cross (heading diff > 30 deg on real data), 0 if following", "Unit": "binary"},
    {"Feature": "ta_bus, ta_car, ta_motorcycle, ta_truck", "Group": "Context", "Definition": "One-hot type of road user A (vehicle)", "Unit": "binary"},
    {"Feature": "tb_bus, tb_car, tb_motorcycle, tb_pedestrian, tb_truck", "Group": "Context", "Definition": "One-hot type of road user B", "Unit": "binary"},
]

AV2_CLASSES = [
    {"AV2 Class": "vehicle", "RAKSHA Type": "car (includes trucks)", "Length (m)": 4.6, "Width (m)": 1.9},
    {"AV2 Class": "bus", "RAKSHA Type": "bus", "Length (m)": 12.0, "Width (m)": 2.6},
    {"AV2 Class": "motorcyclist", "RAKSHA Type": "motorcycle", "Length (m)": 2.2, "Width (m)": 0.8},
    {"AV2 Class": "cyclist", "RAKSHA Type": "motorcycle (two-wheeler)", "Length (m)": 1.8, "Width (m)": 0.6},
    {"AV2 Class": "pedestrian", "RAKSHA Type": "pedestrian", "Length (m)": 0.5, "Width (m)": 0.5},
]

FUNCTIONAL_REQS = [
    {"ID": "FR1", "Requirement": "Ingest crash records from NYC, Chicago and UK STATS19 into one unified schema", "Module": "src/fetch_cities.py"},
    {"ID": "FR2", "Requirement": "Generate Indian-city demonstration data from OSM junctions flagged real=False", "Module": "src/fetch_cities.py"},
    {"ID": "FR3", "Requirement": "Aggregate crashes onto a 150 m grid with severity weighting and per-city blackspots", "Module": "src/build_gis.py"},
    {"ID": "FR4", "Requirement": "Convert Argoverse 2 scenarios into oriented-box pairwise tracks and near-miss labels", "Module": "src/load_av2.py"},
    {"ID": "FR5", "Requirement": "Mixed-traffic simulator with outcome-based near-miss ground truth", "Module": "src/simulate.py"},
    {"ID": "FR6", "Requirement": "Compute TTC, PET, DRAC and kinematic features from causal 1.5 s windows at 0.5 s anchors", "Module": "src/features.py"},
    {"ID": "FR7", "Requirement": "Train & evaluate TTC rule, LogReg, RF and XGBoost with scenario-wise splits & tuned thresholds", "Module": "src/experiments.py"},
    {"ID": "FR8", "Requirement": "Produce lead-time, look-ahead, ablation, pair-rate and TreeSHAP artefacts", "Module": "src/experiments.py"},
    {"ID": "FR9", "Requirement": "Interactive GIS risk map with toggleable layers, filters and ranked inspection table", "Module": "app/raksha.py"},
    {"ID": "FR10", "Requirement": "Plain-language model workbench, live conflict simulator and technical details", "Module": "app/raksha.py"},
    {"ID": "FR11", "Requirement": "Citizen hazard reporting and human-in-the-loop authority triage workflow", "Module": "app/raksha.py"},
    {"ID": "FR12", "Requirement": "Extract road-user tracks from roadside video using YOLO + ByteTrack (extension)", "Module": "src/detect_track.py"},
]

LEAKAGE_GUARANTEES = [
    "Causal feature windows: only past frames [t - 1.5 s, t] are read at each 0.5 s anchor t.",
    "Onset-only prediction: anchors at or after near-miss onset are dropped before training/testing.",
    "Scenario-wise GroupShuffleSplit: 30% test scenarios, 20% of remaining as validation (asserted zero overlap).",
    "Validation-tuned threshold: F1-optimal threshold selected solely on validation split and frozen for test.",
    "Outcome-based ground truth: defined from physical clearance < 1.0 m (not TTC), keeping TTC rule non-trivial.",
    "Real-traffic noise filtering: Argoverse 2 labels additionally gate on rel_speed > 1.5 m/s and swept-path offset.",
]

LIMITATIONS = [
    ("Geography & Traffic Mix", "Real trajectories come from 6 US cities (Argoverse 2); Indian mixed traffic is not yet represented."),
    ("Assumed Object Dimensions", "Argoverse 2 omits bounding-box dimensions; nominal per-class sizes introduce label noise near 1 m."),
    ("Rare-Event Precision", "At 0.12% positive rate (652 test positives), XGBoost achieves 7.6% precision (~12 false alerts per true one) — designed for screening, not automated enforcement."),
    ("Overlapping Anchors", "Consecutive 0.5 s anchors in the same pair are correlated, so anchor counts overstate independent event counts."),
    ("Simulated Demo Layers", "Indian crash counts, map alerts, emerging zones and initial community reports are synthetic demonstration data."),
    ("Heterogeneous Crash Periods", "NYC (Apr–Nov 2026), Chicago (Jan–Jul 2023) and GB STATS19 (2025 injury-only) cover different windows; raw city totals are not comparable."),
    ("Simple Blackspot Rule", "Fixed-grid severity-weighted percentiles are subject to regression-to-mean and MAUP; empirical-Bayes screening is future work."),
    ("Video Pipeline Pending", "src/detect_track.py (YOLO + ByteTrack) is implemented but awaits pixel-to-metre homography calibration on roadside video."),
    ("Non-Commercial Licence", "Argoverse 2 is licensed CC BY-NC-SA 4.0 for non-commercial research."),
]

FUTURE_SCOPE = [
    ("Indian Mixed-Traffic Collection", "Record roadside junction videos across Indian intersections to capture motorcycles, auto-rickshaws, buses and pedestrians."),
    ("Calibrated YOLO + ByteTrack", "Run src/detect_track.py with camera homography matrices to extract metric trajectories directly from CCTV/dashcam feeds."),
    ("Real-Time Streaming Scoring", "Score live 1.5 s sliding windows on edge/junction servers with temporal alert debouncing across overlapping anchors."),
    ("Empirical-Bayes GIS & Field Pilot", "Replace percentile blackspot thresholds with Empirical-Bayes safety performance functions and validate alerts against municipal inspections."),
]

CONFLICT_PRESETS = {
    "Truck vs Pedestrian (Imminent Crossing Conflict)": {
        "crossing": 1, "type_a": "truck", "type_b": "pedestrian",
        "clr_last": 1.6, "dist_last": 6.0, "ttc_last": 1.4, "ttc_min": 1.1,
        "pet_last": 0.4, "pet_min": 0.3, "drac_last": 6.8, "drac_max": 7.9,
        "closing_last": 4.7, "speed_a": 7.5, "speed_b": 1.4,
        "acc_a": -0.4, "acc_b": 0.0, "acc_a_min": -1.2, "acc_b_min": -0.2, "jerk_a": 1.4,
    },
    "Car vs Truck (Hard-Braking Rear-End Approach)": {
        "crossing": 0, "type_a": "car", "type_b": "truck",
        "clr_last": 2.4, "dist_last": 8.5, "ttc_last": 1.9, "ttc_min": 1.6,
        "pet_last": 0.5, "pet_min": 0.4, "drac_last": 5.2, "drac_max": 6.1,
        "closing_last": 5.0, "speed_a": 13.5, "speed_b": 8.5,
        "acc_a": -2.8, "acc_b": -3.5, "acc_a_min": -3.8, "acc_b_min": -4.2, "jerk_a": 2.6,
    },
    "Car vs Car (Safe Free-Flow Highway Pass)": {
        "crossing": 0, "type_a": "car", "type_b": "car",
        "clr_last": 18.0, "dist_last": 22.2, "ttc_last": 28.0, "ttc_min": 24.0,
        "pet_last": 2.1, "pet_min": 1.9, "drac_last": 0.0, "drac_max": 0.2,
        "closing_last": 0.4, "speed_a": 12.0, "speed_b": 11.6,
        "acc_a": 0.1, "acc_b": 0.0, "acc_a_min": -0.2, "acc_b_min": -0.1, "jerk_a": 0.2,
    },
}
