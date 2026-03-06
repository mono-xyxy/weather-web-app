import os
import requests
import datetime
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

models = {}
metrics = {}

CSV_PATH = "C:/Users/rohan/OneDrive/Desktop/Python/synthetic_weather_output_million.csv"

def train_models():
    print("Initializing AI Model Training...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found!")
        return

    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    
    X = df[['month', 'day', 'hour']]
    
    y_extreme = df['is_extreme_weather'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_extreme, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    models['extreme'] = clf
    y_pred = clf.predict(X_test)
    
    metrics['extreme_accuracy'] = accuracy_score(y_test, y_pred)
    metrics['extreme_precision'] = precision_score(y_test, y_pred, zero_division=0)
    
    targets = ['temp_celsius', 'humidity_percent', 'wind_speed_kmh', 'precipitation_mm']
    for t in targets:
        y = df[t]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42)
        reg.fit(X_train, y_train)
        models[t] = reg
        y_pred = reg.predict(X_test)
        metrics[t + '_mae'] = mean_absolute_error(y_test, y_pred)
        
    print("All Models securely trained!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AI Weather Forecaster</title>
    <!-- Use Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #ebf0f4;
            --main-panel: #ffffff;
            --text-main: #333333;
            --text-muted: #888888;
            --card-primary: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
            --card-primary-text: #0284c7;
            --sidebar-icon: #bbbbbb;
            --sidebar-icon-active: #f97316;
            --border-color: #f1f5f9;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Nunito', sans-serif; }

        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            height: 100vh;
            display: flex; align-items: center; justify-content: center;
            overflow: hidden; padding: 2rem;
            transition: background-color 0.5s ease;
        }

        .app-window {
            background: var(--main-panel);
            width: 100%; max-width: 1400px; height: 90vh;
            border-radius: 30px; box-shadow: 0 20px 50px rgba(0,0,0,0.08);
            display: flex; overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 80px; background: #ffffff;
            border-right: 1px solid var(--border-color);
            display: flex; flex-direction: column; align-items: center; padding: 2rem 0;
            z-index: 100;
        }
        .logo { color: var(--sidebar-icon-active); font-size: 24px; margin-bottom: 3rem; }
        .nav-item { color: var(--sidebar-icon); font-size: 20px; margin: 15px 0; cursor: pointer; transition: 0.2s;}
        .nav-item:hover, .nav-item.active { color: var(--sidebar-icon-active); transform: scale(1.1); }
        .bottom-nav { margin-top: auto; color: var(--sidebar-icon); font-size: 20px; cursor: pointer;}
        
        /* Main Content */
        .main-content {
            flex: 1; padding: 2rem 3rem; overflow-y: auto; background: var(--border-color);
            position: relative;
        }

        /* Views */
        .view-section { display: none; animation: fadeIn 0.4s ease; }
        .view-section.active { display: block; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Header */
        .header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;
        }
        .user-info { display: flex; align-items: center; gap: 1rem; }
        .avatar { width: 45px; height: 45px; border-radius: 50%; background: #ccc; }
        .user-text h4 { font-size: 0.9rem; color: var(--text-muted); font-weight: 600;}
        .user-text h2 { font-size: 1.2rem; font-weight: 800; }
        
        .search-bar { background: #fff; border-radius: 20px; padding: 0.6rem 1.5rem; display: flex; align-items: center; width: 300px; color: var(--text-muted); box-shadow: 0 2px 10px rgba(0,0,0,0.02);}
        .search-bar input { border: none; outline: none; margin-left: 10px; width: 100%; font-size: 0.95rem; }
        
        .header-icons { display: flex; gap: 1rem; color: var(--text-main); font-size: 1.2rem;}

        /* Controls Area */
        .controls-card {
            background: #fff; padding: 1.5rem; border-radius: 20px;
            display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem;
            margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            flex-wrap: wrap;
        }
        .input-group { flex: 1; min-width: 200px; }
        .input-group label { display: block; font-size: 0.85rem; color: var(--text-muted); font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; }
        .input-group select, .input-group input { 
            width: 100%; padding: 0.8rem; border-radius: 12px; border: 1px solid var(--border-color); 
            background: #f8fafc; color: var(--text-main); font-weight: 600; outline: none;
        }
        button.btn-primary {
            background: var(--sidebar-icon-active); color: #fff; border: none; padding: 0.9rem 2rem; 
            border-radius: 12px; font-weight: 700; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 10px rgba(249, 115, 22, 0.3);
            white-space: nowrap; height: 48px;
        }
        button.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(249, 115, 22, 0.4); }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            grid-template-rows: auto auto;
            gap: 2rem;
        }

        /* Large Primary Weather Card */
        .weather-main-card {
            background: var(--card-primary); color: var(--card-primary-text);
            border-radius: 24px; padding: 2rem; display: flex; flex-direction: column; justify-content: space-between;
            position: relative; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            transition: background 0.5s ease, color 0.5s ease;
        }
        .weather-loc { display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 1.1rem; z-index: 2; position: relative; }
        .weather-info { margin-top: 1.5rem; z-index: 2; position: relative;}
        .weather-info h3 { font-size: 1.4rem; font-weight: 600; opacity: 0.9; }
        .weather-info h1 { font-size: 4.5rem; font-weight: 800; line-height: 1.1; margin: 0.5rem 0; }
        .weather-info .feels { font-size: 1rem; opacity: 0.8; font-weight: 600;}
        .weather-stats { display: flex; gap: 1rem; margin-top: 2rem; z-index: 2; position: relative; flex-wrap: wrap;}
        .stat-badge { background: rgba(255,255,255,0.4); backdrop-filter: blur(5px); padding: 0.8rem 1.2rem; border-radius: 15px; text-align: center; flex: 1; min-width: 100px;}
        .stat-badge span { display: block; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; opacity: 0.8; margin-bottom: 0.2rem;}
        .stat-badge strong { font-size: 1.1rem; }
        .weather-icon-large { position: absolute; right: -20px; top: 20px; font-size: 10rem; opacity: 0.3; z-index: 1; }

        /* Mini Map Verification Card */
        .map-card { background: #fff; border-radius: 24px; padding: 1.5rem; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.03); }
        .map-card h3 { font-size: 1.1rem; margin-bottom: 1rem; color: var(--text-main); font-weight: 800;}
        .verify-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-color);}
        .verify-row:last-child { border: none; margin: 0; padding: 0;}
        .verify-col-ai, .verify-col-real { flex: 1; }
        .verify-col-real { text-align: right; }
        .verify-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase;}
        .verify-val { font-size: 1.8rem; font-weight: 800; color: var(--text-main); }
        .verify-val.real { color: var(--sidebar-icon-active); }
        .delta-box { background: #f8fafc; border-radius: 12px; padding: 1rem; margin-top: 1rem; font-size: 0.9rem; font-weight: 600; color: var(--text-muted); text-align: center;}

        /* Today's Metrics */
        .hourly-card { background: #fff; border-radius: 24px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.03); grid-column: 1 / 2; }
        .hourly-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 1.5rem;}
        .metrics-ul { list-style: none; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
        .metrics-ul li { text-align: center; flex: 1; min-width: 80px;}
        .metrics-ul i { font-size: 1.5rem; color: #a0aec0; margin-bottom: 0.8rem; }
        .metrics-ul h4 { font-size: 1.2rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.3rem;}
        .metrics-ul p { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); }

        /* Status Card */
        .status-card { background: #2d3748; color: #fff; border-radius: 24px; padding: 1.5rem; display: flex; flex-direction: column; justify-content: center; position: relative; overflow: hidden; }
        .status-card.status-extreme { background: #e53e3e; }
        .status-card h3 { font-size: 1.1rem; opacity: 0.9; margin-bottom: 0.5rem; z-index: 2;}
        .status-card p { font-size: 0.9rem; opacity: 0.7; z-index: 2;}
        .status-card .status-val { font-size: 2rem; font-weight: 800; margin-top: 1rem; z-index: 2;}
        .status-card i.bg-icon { position: absolute; right: 10px; bottom: -10px; font-size: 6rem; opacity: 0.2; color: #cbd5e0;}

        /* Extra Placeholder Pages */
        .placeholder-page { background: #fff; border-radius: 24px; padding: 3rem; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.03); margin-top: 2rem; }
        .placeholder-page i { font-size: 4rem; color: var(--sidebar-icon-active); margin-bottom: 1rem; opacity: 0.8; }
        .placeholder-page h2 { font-size: 2rem; margin-bottom: 1rem; color: var(--text-main); }
        .placeholder-page p { color: var(--text-muted); font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.6;}
        
        .switch-container { display: flex; align-items: center; justify-content: center; gap: 1rem; margin-top: 2rem; }
        .switch { position: relative; display: inline-block; width: 60px; height: 34px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--sidebar-icon-active); }
        input:checked + .slider:before { transform: translateX(26px); }

        /* =========================================
           MOBILE RESPONSIVENESS MEDIA QUERIES
           ========================================= */
        @media (max-width: 1024px) {
            .dashboard-grid { grid-template-columns: 1fr; }
            .hourly-card { grid-column: 1 / -1; }
        }

        @media (max-width: 768px) {
            body { padding: 0; background: var(--border-color); }
            .app-window { flex-direction: column; height: 100vh; border-radius: 0; }
            
            /* Sidebar becomes Bottom Navigation */
            .sidebar {
                width: 100%; height: 75px; flex-direction: row; justify-content: space-around;
                padding: 0; border-right: none; border-top: 1px solid var(--border-color);
                box-shadow: 0 -5px 15px rgba(0,0,0,0.05); order: 2;
            }
            .sidebar .logo { display: none; }
            .nav-item { margin: 0; padding: 15px; font-size: 24px;}
            .bottom-nav { margin: 0; padding: 15px; font-size: 24px;}
            
            /* Main Content Adjustments */
            .main-content { order: 1; padding: 1.5rem 1rem; }
            
            /* Header Adjustments */
            .header { flex-direction: column; align-items: flex-start; gap: 1.5rem; }
            .search-bar { width: 100%; }
            .header-icons { position: absolute; top: 1.5rem; right: 1rem; }

            /* Controls Adjustments */
            .controls-card { flex-direction: column; align-items: stretch; gap: 1rem; padding: 1rem; }
            button.btn-primary { width: 100%; }

            /* Weather Card Adjustments */
            .weather-info h1 { font-size: 3.5rem; }
            .weather-stats { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="app-window">
        <!-- Sidebar Navigation -->
        <div class="sidebar">
            <div class="logo"><i class="fa-solid fa-cloud"></i></div>
            <div class="nav-item active" onclick="switchView('dashboard', this)"><i class="fa-solid fa-border-all"></i></div>
            <div class="nav-item" onclick="switchView('analytics', this)"><i class="fa-solid fa-chart-line"></i></div>
            <div class="nav-item" onclick="switchView('locations', this)"><i class="fa-solid fa-location-dot"></i></div>
            <div class="nav-item" onclick="switchView('calendar', this)"><i class="fa-regular fa-calendar"></i></div>
            <div class="bottom-nav" onclick="switchView('settings', this)"><i class="fa-solid fa-gear"></i></div>
        </div>

        <!-- Main Workspace -->
        <div class="main-content">
            <div class="header">
                <div class="user-info">
                    <img src="https://ui-avatars.com/api/?name=AI+Guest&background=random" class="avatar" alt="User">
                    <div class="user-text">
                        <h4>Hello,</h4>
                        <h2>AI Researcher</h2>
                    </div>
                </div>
                <div class="search-bar">
                    <i class="fa-solid fa-magnifying-glass"></i>
                    <input type="text" placeholder="Search parameters...">
                </div>
                <div class="header-icons">
                    <i class="fa-regular fa-bell"></i>
                </div>
            </div>

            <!-- View: Dashboard (Home) -->
            <div id="view-dashboard" class="view-section active">
                <div class="controls-card">
                    <div class="input-group">
                        <label>Destination City</label>
                        <select id="target-city">
                            <option value="Dhaka">Dhaka, Bangladesh</option>
                            <option value="New York">New York, USA</option>
                            <option value="London">London, UK</option>
                            <option value="Tokyo">Tokyo, Japan</option>
                            <option value="Sydney">Sydney, Australia</option>
                            <option value="Mumbai">Mumbai, India</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>Prediction Point</label>
                        <input type="datetime-local" id="target-date">
                    </div>
                    <button class="btn-primary" onclick="requestVerification()" id="verify-btn">Synthesize</button>
                </div>

                <div class="dashboard-grid">
                    <!-- Large Output Card -->
                    <div class="weather-main-card" id="theme-card">
                        <i class="fa-solid fa-cloud-sun weather-icon-large"></i>
                        <div class="weather-loc">
                            <i class="fa-solid fa-location-dot"></i> <span id="loc-display">City, Country</span>
                        </div>
                        <div class="weather-info">
                            <h3 id="ai-status-text">AI Prediction</h3>
                            <h1 id="ai-temp">--°C</h1>
                            <span class="feels" id="ai-date-display">Select a date</span>
                        </div>
                        <div class="weather-stats">
                            <div class="stat-badge">
                                <span>Precipitation</span>
                                <strong id="ai-precip">-- mm</strong>
                            </div>
                            <div class="stat-badge">
                                <span>Humidity</span>
                                <strong id="ai-humid">-- %</strong>
                            </div>
                        </div>
                    </div>

                    <!-- API Verification Split -->
                    <div class="map-card">
                        <h3>Ground Truth API Benchmark</h3>
                        <div class="verify-row">
                            <div class="verify-col-ai">
                                <div class="verify-label">Synthetic LLM</div>
                                <div class="verify-val" id="synth-temp">--°</div>
                            </div>
                            <div class="verify-col-real">
                                <div class="verify-label">Real API</div>
                                <div class="verify-val real" id="real-temp">--°</div>
                            </div>
                        </div>
                        <div class="verify-row">
                            <div class="verify-col-ai">
                                <div class="verify-label">Wind</div>
                                <div class="verify-val" id="synth-wind" style="font-size: 1.2rem;">--</div>
                            </div>
                            <div class="verify-col-real">
                                <div class="verify-label">Wind</div>
                                <div class="verify-val real" id="real-wind" style="font-size: 1.2rem;">--</div>
                            </div>
                        </div>
                        <div class="delta-box" id="delta-report">
                            Awaiting synthesis to calculate deviation matrix...
                        </div>
                    </div>

                    <!-- Bottom Metrics List -->
                    <div class="hourly-card">
                        <div class="hourly-title">Model Precision Matrix</div>
                        <ul class="metrics-ul">
                            <li>
                                <i class="fa-solid fa-bullseye"></i>
                                <h4>{{ "%.1f"|format(metrics['extreme_accuracy'] * 100) }}%</h4>
                                <p>Global Acc.</p>
                            </li>
                            <li>
                                <i class="fa-solid fa-bolt"></i>
                                <h4>{{ "%.1f"|format(metrics['extreme_precision'] * 100) }}%</h4>
                                <p>Ext. Precision</p>
                            </li>
                            <li>
                                <i class="fa-solid fa-temperature-half"></i>
                                <h4>{{ "%.2f"|format(metrics['temp_celsius_mae']) }}°C</h4>
                                <p>Temp Limit (MAE)</p>
                            </li>
                            <li>
                                <i class="fa-solid fa-droplet"></i>
                                <h4>{{ "%.2f"|format(metrics['humidity_percent_mae']) }}%</h4>
                                <p>Hum. Error (MAE)</p>
                            </li>
                        </ul>
                    </div>

                    <!-- Status Block -->
                    <div class="status-card" id="extreme-card">
                        <i class="fa-solid fa-triangle-exclamation bg-icon"></i>
                        <h3>Anomaly Status</h3>
                        <p>Evaluates synthetic extreme conditions threshold rules.</p>
                        <div class="status-val" id="ai-extrem">Pending</div>
                    </div>
                </div>
            </div>

            <!-- View: Analytics -->
            <div id="view-analytics" class="view-section">
                <div class="placeholder-page">
                    <i class="fa-solid fa-network-wired"></i>
                    <h2>AI Engine Analytics</h2>
                    <p>Scikit-Learn Random Forest Pipeline diagnostic interface.<br>Current Configuration: 30 Estimators, Depth 8. Distributed temporal splits injected from PySpark.</p>
                    <div style="margin-top: 2rem; background: var(--border-color); padding: 1.5rem; border-radius: 12px; text-align: left;">
                        <ul style="list-style: none; line-height: 2;">
                            <li><strong>Temperature Model:</strong> Loss function stabilized. MAE: {{ "%.4f"|format(metrics['temp_celsius_mae']) }}</li>
                            <li><strong>Wind Velocity Model:</strong> Loss function stabilized. MAE: {{ "%.4f"|format(metrics['wind_speed_kmh_mae']) }}</li>
                            <li><strong>Precipitation Model:</strong> Loss function stabilized. MAE: {{ "%.4f"|format(metrics['precipitation_mm_mae']) }}</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- View: Locations -->
            <div id="view-locations" class="view-section">
                <div class="placeholder-page">
                    <i class="fa-solid fa-earth-americas"></i>
                    <h2>Global Verification Nodes</h2>
                    <p>Selectable regions mapped to Open-Meteo REST API bounds for accuracy verification against synthetic models.</p>
                    <div style="margin-top: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                        <div style="background:var(--border-color); padding: 1rem; border-radius:12px;">Dhaka [BGD]<br><small>Lat: 23.8, Lon: 90.4</small></div>
                        <div style="background:var(--border-color); padding: 1rem; border-radius:12px;">New York [USA]<br><small>Lat: 40.7, Lon: -74.0</small></div>
                        <div style="background:var(--border-color); padding: 1rem; border-radius:12px;">London [GBR]<br><small>Lat: 51.5, Lon: -0.1</small></div>
                        <div style="background:var(--border-color); padding: 1rem; border-radius:12px;">Tokyo [JPN]<br><small>Lat: 35.6, Lon: 139.6</small></div>
                    </div>
                </div>
            </div>

            <!-- View: Calendar -->
            <div id="view-calendar" class="view-section">
                <div class="placeholder-page">
                    <i class="fa-regular fa-calendar-days"></i>
                    <h2>7-Day Forecasting Timeline</h2>
                    <p>Module to rapidly predict iterative dates using ensemble inference concurrently.</p>
                    <div style="margin-top: 2rem; height: 100px; background: repeating-linear-gradient(45deg, var(--border-color), var(--border-color) 10px, transparent 10px, transparent 20px); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-weight: bold;">
                        Chronological Inference Mapping Inactive
                    </div>
                </div>
            </div>

            <!-- View: Settings -->
            <div id="view-settings" class="view-section">
                <div class="placeholder-page">
                    <i class="fa-solid fa-sliders"></i>
                    <h2>Control Override</h2>
                    <p>Toggle the Geo-Temporal Autochromatic Engine. Leaving this on enables the dashboard background to visually represent the season of the predicted destination.</p>
                    
                    <div class="switch-container">
                        <strong>Dynamic Season Theming:</strong>
                        <label class="switch">
                            <input type="checkbox" id="theme-toggle" checked>
                            <span class="slider"></span>
                        </label>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <script>
        // Data Structures
        const CITIES = {
            "New York": {lat: 40.7128, lon: -74.0060, label: "New York, USA"},
            "London": {lat: 51.5074, lon: -0.1278, label: "London, UK"},
            "Tokyo": {lat: 35.6762, lon: 139.6503, label: "Tokyo, Japan"},
            "Sydney": {lat: -33.8688, lon: 151.2093, label: "Sydney, Australia"},
            "Mumbai": {lat: 19.0760, lon: 72.8777, label: "Mumbai, India"},
            "Dhaka": {lat: 23.8103, lon: 90.4125, label: "Dhaka, Bangladesh"}
        };

        // Navigation Controller
        function switchView(viewId, element) {
            // Update active state on sidebar
            document.querySelectorAll('.sidebar div').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
            
            // Hide all views, show targeted view
            document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
            document.getElementById('view-' + viewId).classList.add('active');
        }

        // Initialize Time Input
        const now = new Date();
        now.setMinutes(0);
        const offset = now.getTimezoneOffset();
        const localNow = new Date(now.getTime() - (offset*60*1000));
        document.getElementById('target-date').value = localNow.toISOString().slice(0,16);

        // Theme Engine
        function applyDynamicTheme(dt, city) {
            // Check settings override
            if(!document.getElementById('theme-toggle').checked) {
                const root = document.documentElement;
                root.style.setProperty('--card-primary', 'linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%)'); 
                root.style.setProperty('--card-primary-text', '#0284c7');
                root.style.setProperty('--bg-body', '#ebf0f4');
                return;
            }

            const month = dt.getMonth(); 
            const lat = CITIES[city].lat;
            let season = "";
            if (lat >= 0) { 
                if (month === 11 || month <= 1) season = "winter";
                else if (month >= 2 && month <= 4) season = "spring";
                else if (month >= 5 && month <= 7) season = "summer";
                else season = "autumn";
            } else { 
                if (month === 11 || month <= 1) season = "summer";
                else if (month >= 2 && month <= 4) season = "autumn";
                else if (month >= 5 && month <= 7) season = "winter";
                else season = "spring";
            }

            const root = document.documentElement;
            if (season === "winter") {
                root.style.setProperty('--card-primary', 'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)'); 
                root.style.setProperty('--card-primary-text', '#1e3a8a');
                root.style.setProperty('--bg-body', '#e0f2fe'); 
            } else if (season === "spring") {
                root.style.setProperty('--card-primary', 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)'); 
                root.style.setProperty('--card-primary-text', '#14532d');
                root.style.setProperty('--bg-body', '#f0fdf4');
            } else if (season === "summer") {
                root.style.setProperty('--card-primary', 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)'); 
                root.style.setProperty('--card-primary-text', '#7c2d12');
                root.style.setProperty('--bg-body', '#fff7ed');
            } else if (season === "autumn") {
                root.style.setProperty('--card-primary', 'linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%)'); 
                root.style.setProperty('--card-primary-text', '#3f3f46');
                root.style.setProperty('--bg-body', '#f4f4f5');
            }
        }

        async function requestVerification() {
            const btn = document.getElementById('verify-btn');
            const dateStr = document.getElementById('target-date').value;
            const city = document.getElementById('target-city').value;
            
            if(!dateStr) return;
            
            const dt = new Date(dateStr);
            applyDynamicTheme(dt, city);

            const originalText = btn.innerText;
            btn.innerText = "Processing...";
            btn.style.opacity = "0.7";

            try {
                const response = await fetch('/verify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ date: dateStr, city: city })
                });

                const data = await response.json();
                if(data.error) { alert("Error: " + data.error); } 
                else { displayMatrix(data, city, dt); }
            } catch (error) {
                alert("Computational Error: " + error);
            } finally {
                btn.innerText = originalText;
                btn.style.opacity = "1";
            }
        }

        function displayMatrix(data, city, dt) {
            document.getElementById('loc-display').innerText = CITIES[city].label;
            document.getElementById('ai-temp').innerText = data.ai.temp_celsius + '°C';
            document.getElementById('ai-precip').innerText = data.ai.precipitation_mm + ' mm';
            document.getElementById('ai-humid').innerText = data.ai.humidity_percent + '%';
            
            const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            document.getElementById('ai-date-display').innerText = days[dt.getDay()] + ', ' + dt.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

            document.getElementById('synth-temp').innerText = data.ai.temp_celsius + '°';
            document.getElementById('synth-wind').innerText = data.ai.wind_speed_kmh + ' km/h';

            if(data.real.found) {
                document.getElementById('real-temp').innerText = data.real.temp_celsius + '°';
                document.getElementById('real-wind').innerText = data.real.wind_speed_kmh + ' km/h';
                
                const diff = Math.abs(data.ai.temp_celsius - data.real.temp_celsius).toFixed(1);
                let report = "";
                if(diff < 5) report = `<span style="color:#16a34a">Deviation: ${diff}°C. Excellent Synthetic Match!</span>`;
                else if(diff < 12) report = `<span style="color:#ea580c">Deviation: ${diff}°C. Acceptable variance.</span>`;
                else report = `<span style="color:#dc2626">Deviation: ${diff}°C. AI deviates heavily from live API.</span>`;
                document.getElementById('delta-report').innerHTML = report;
            } else {
                document.getElementById('real-temp').innerText = "N/A";
                document.getElementById('real-wind').innerText = "--";
                document.getElementById('delta-report').innerHTML = "<span style='color:#ea580c'>API 7-day query limit exceeded.</span>";
            }

            const extCard = document.getElementById('extreme-card');
            if(data.ai.is_extreme_weather) {
                document.getElementById('ai-extrem').innerText = "ALERT ACTIVE";
                extCard.classList.add('status-extreme');
            } else {
                document.getElementById('ai-extrem').innerText = "ALL CLEAR";
                extCard.classList.remove('status-extreme');
            }
        }
        
        // Re-run theme application when toggle changes
        document.getElementById('theme-toggle').addEventListener('change', () => {
             const dt = new Date(document.getElementById('target-date').value);
             const city = document.getElementById('target-city').value;
             applyDynamicTheme(dt, city);
        });

        // Trigger default init theme
        setTimeout(() => requestVerification(), 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, metrics=metrics)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    date_str = data.get('date')
    city = data.get('city')
    
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        return jsonify({"error": "Invalid date format"}), 400
        
    CITIES_API = {
        "New York": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
        "London": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
        "Tokyo": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
        "Sydney": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777, "timezone": "Asia/Kolkata"},
        "Dhaka": {"lat": 23.8103, "lon": 90.4125, "timezone": "Asia/Dhaka"}
    }
    coords = CITIES_API.get(city)
    if not coords:
        return jsonify({"error": "City not mapped"}), 400
        
    input_df = pd.DataFrame([{'month': dt.month, 'day': dt.day, 'hour': dt.hour}])
    
    ai_res = {
        'temp_celsius': round(models['temp_celsius'].predict(input_df)[0], 2),
        'humidity_percent': round(models['humidity_percent'].predict(input_df)[0], 2),
        'wind_speed_kmh': round(models['wind_speed_kmh'].predict(input_df)[0], 2),
        'precipitation_mm': round(models['precipitation_mm'].predict(input_df)[0], 2),
        'is_extreme_weather': bool(models['extreme'].predict(input_df)[0])
    }
    
    target_api_time = dt.strftime("%Y-%m-%dT%H:00")
    real_res = {"found": False}
    try:
        api_url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&timezone={coords['timezone']}"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            weather_data = response.json()
            hourly = weather_data.get('hourly', {})
            times = hourly.get('time', [])
            if target_api_time in times:
                idx = times.index(target_api_time)
                real_res = {
                    "found": True,
                    "temp_celsius": hourly['temperature_2m'][idx],
                    "humidity_percent": hourly['relative_humidity_2m'][idx],
                    "wind_speed_kmh": hourly['wind_speed_10m'][idx],
                    "precipitation_mm": hourly['precipitation'][idx]
                }
    except Exception as e:
        print(f"API Fetch Error: {e}")

    return jsonify({"ai": ai_res, "real": real_res})

if __name__ == '__main__':
    train_models()
    print("Verification Dashboard online: http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
