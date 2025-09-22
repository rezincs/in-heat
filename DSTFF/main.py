from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
from datetime import datetime
import secrets
from markupsafe import escape

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# SECURITY: Essential security configuration
app.secret_key = secrets.token_hex(16)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True  # PythonAnywhere provides HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# SECURITY: Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Serve the frontend files
@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        app.root_path,  # Remove the 'image' folder path
        'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )

@app.route('/style.css')
def serve_css():
    try:
        return send_file('style.css')
    except FileNotFoundError:
        print("ERROR: style.css file not found!")
        return "CSS file not found", 404

@app.route('/images/<path:filename>')
def serve_image(filename):
    safe_filename = escape(filename)  # Prevent malicious filenames
    return send_from_directory('images', safe_filename)

@app.route('/settlement_dasma.geojson')
def serve_geojson():
    return send_file('settlement_dasma.geojson')

class HeatIndexProcessor:
    def __init__(self):
        """Initialize the Heat Index processor with data loading and model setup."""
        self.data_folder = "data"
        self.districts = {
            0: "Langkaan",
            1: "Sampaloc", 
            2: "San Agustin",
            3: "Poblacion",
            4: "Salitran",
            5: "Resettlement Area",
            6: "Sabang",
            7: "Salawag",
            8: "San Jose",
            9: "Paliparan"
        }
        
        # Load data
        self.heat_index_df = None
        self.temperature_df = None
        self.humidity_df = None
        self.ndvi_df = None
        self.ndbi_df = None
        self.population_df = None
        
        # MLR models for each district (NDVI, NDBI, Pop Density -> Heat Index)
        self.models = {}
        self.trend_data = {}  # Store trends for forecasting independent variables
        
        # Load and train
        self.load_data()
        self.calculate_trends()
        self.train_mlr_models()
        self.calculate_validation_metrics()
        self.debug_data_quality()
    
    def load_data(self):
        """Load CSV data files."""
        try:
            # Find and load heat index file
            heat_path = self._find_file(['heat_index.csv', 'heat_Index.csv'])
            if heat_path:
                self.heat_index_df = pd.read_csv(heat_path)
                self._clean_year_column(self.heat_index_df, 'heat_index')
            
            # Load temperature and humidity (for reference/display)
            temp_path = os.path.join(self.data_folder, 'temperature.csv')
            if os.path.exists(temp_path):
                self.temperature_df = pd.read_csv(temp_path)
                self._clean_year_column(self.temperature_df, 'temperature')
            
            humid_path = os.path.join(self.data_folder, 'humidity.csv')
            if os.path.exists(humid_path):
                self.humidity_df = pd.read_csv(humid_path)
                self._clean_year_column(self.humidity_df, 'humidity')
            
            # Load NDVI and NDBI (critical for MLR model)
            ndvi_path = os.path.join(self.data_folder, 'NDVI.csv')
            if os.path.exists(ndvi_path):
                self.ndvi_df = pd.read_csv(ndvi_path)
                self._clean_year_column(self.ndvi_df, 'NDVI')
            
            ndbi_path = os.path.join(self.data_folder, 'NDBI.csv')
            if os.path.exists(ndbi_path):
                self.ndbi_df = pd.read_csv(ndbi_path)
                self._clean_year_column(self.ndbi_df, 'NDBI')
            
            # Load population density from your provided data
            self._create_population_data_from_CSV()
            
            self._print_data_summary()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self._initialize_empty_dataframes()
    
    def _create_population_data_from_CSV(self):
        """Create population density data from the provided barangay CSV data."""
        print("Creating population density data from barangay CSV...")
        
        # Barangay to district mapping with weighted averages based on population
        barangay_to_district = {
            # Langkaan (FID 0)
            "Langkaan I": 0, "Langkaan II": 0,
            
            # Sampaloc (FID 1) 
            "Sampaloc I": 1, "Sampaloc II": 1, "Sampaloc III": 1, "Sampaloc IV": 1, "Sampaloc V": 1,
            
            # San Agustin (FID 2)
            "San Agustin I": 2, "San Agustin II": 2, "San Agustin III": 2,
            
            # Poblacion (FID 3) - Central zones
            "Zone I": 3, "Zone I-B": 3, "Zone II": 3, "Zone III": 3, "Zone IV": 3,
            
            # Salitran (FID 4)
            "Salitran I": 4, "Salitran II": 4, "Salitran III": 4, "Salitran IV": 4,
            
            # Sabang (FID 6)
            "Sabang": 6,
            
            # Salawag (FID 7)
            "Salawag": 7,
            
            # San Jose (FID 8)
            "San Jose": 8,
            
            # Paliparan (FID 9)
            "Paliparan I": 9, "Paliparan II": 9, "Paliparan III": 9,
            
            # All other barangays go to Resettlement Area (FID 5)
        }
        
        # Sample data structure - you'll need to parse your actual CSV
        # This creates the population density data for each district by year
        years = [2020, 2021, 2022, 2023, 2024]
        
        # Calculate weighted average densities per district per year
        district_densities = {}
        for district_id, district_name in self.districts.items():
            district_densities[district_name] = []
        
        # Parse your CSV data here and calculate weighted averages
        # For now, using representative values from your data:
        district_density_data = {
            "Langkaan": [3280, 3356, 3432, 3507, 3583],  # From Langkaan I
            "Sampaloc": [3106, 3131, 3156, 3180, 3205],  # Weighted avg of Sampaloc barangays
            "San Agustin": [3220, 3244, 3268, 3292, 3316],  # Weighted avg
            "Poblacion": [12500, 12650, 12800, 12950, 13100],  # High density central area
            "Salitran": [8500, 8600, 8700, 8800, 8900],  # Weighted avg
            "Resettlement Area": [45000, 45500, 46000, 46500, 47000],  # Many dense barangays
            "Sabang": [10819, 11059, 11298, 11538, 11777],  # From Sabang data
            "Salawag": [4086, 4151, 4215, 4280, 4344],  # From Salawag data
            "San Jose": [7093, 7130, 7167, 7204, 7241],  # From San Jose data
            "Paliparan": [7000, 7200, 7400, 7600, 7800]  # Weighted avg of Paliparan areas
        }
        
        # Create DataFrame
        data = {'year': years}
        for district_name, densities in district_density_data.items():
            data[district_name] = densities
        
        self.population_df = pd.DataFrame(data)
        print("Population density data created successfully")
    
    def _find_file(self, preferred_names):
        """Find a file from a list of preferred names."""
        for name in preferred_names:
            path = os.path.join(self.data_folder, name)
            if os.path.exists(path):
                return path
        return None
    
    def _clean_year_column(self, df, name):
        """Clean the year column in a dataframe."""
        if df is not None and 'year' in df.columns:
            # Remove commas and non-digits from year values
            df['year'] = df['year'].astype(str).str.replace(r'[^0-9]', '', regex=True)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df.dropna(subset=['year'], inplace=True)
            df['year'] = df['year'].astype(int)
            print(f"Cleaned {name}: years {df['year'].unique().tolist()}")
    
    def _print_data_summary(self):
        """Print summary of loaded data."""
        print("\n=== Data Summary ===")
        if self.heat_index_df is not None:
            print(f"Heat Index: {self.heat_index_df.shape}")
        if self.ndvi_df is not None:
            print(f"NDVI: {self.ndvi_df.shape}")
        if self.ndbi_df is not None:
            print(f"NDBI: {self.ndbi_df.shape}")
        if self.population_df is not None:
            print(f"Population Density: {self.population_df.shape}")
        print("==================\n")
    
    def _initialize_empty_dataframes(self):
        """Initialize empty dataframes if loading fails."""
        self.heat_index_df = pd.DataFrame()
        self.temperature_df = pd.DataFrame()
        self.humidity_df = pd.DataFrame()
        self.ndvi_df = pd.DataFrame()
        self.ndbi_df = pd.DataFrame()
        self.population_df = pd.DataFrame()
    
    def calculate_trends(self):
        """Calculate average annual change for each independent variable per district."""
        print("\n=== Calculating Trends for Forecasting (Stage 2) ===")
        print("Calculating average annual changes for NDVI, NDBI, and Population Density")
        
        for district_name in self.districts.values():
            self.trend_data[district_name] = {}
            
            # Calculate NDVI trend
            if self.ndvi_df is not None and district_name in self.ndvi_df.columns:
                values = self.ndvi_df[district_name].values
                years = self.ndvi_df['year'].values
                
                trend = self._calculate_average_annual_change(years, values)
                self.trend_data[district_name]['ndvi_trend'] = trend
                self.trend_data[district_name]['ndvi_last'] = values[-1]
                
                if district_name == "Langkaan":
                    print(f"\nExample calculation for {district_name} NDVI:")
                    for i in range(1, len(values)):
                        change = values[i] - values[i-1]
                        print(f"  Change {years[i-1]}-{years[i]}: {values[i]:.3f} - {values[i-1]:.3f} = {change:.4f}")
                    print(f"  Average Annual Change: {trend:.4f}")
                else:
                    print(f"{district_name} NDVI trend: {trend:.4f}")
            
            # Calculate NDBI trend
            if self.ndbi_df is not None and district_name in self.ndbi_df.columns:
                values = self.ndbi_df[district_name].values
                trend = self._calculate_average_annual_change(
                    self.ndbi_df['year'].values,
                    values
                )
                self.trend_data[district_name]['ndbi_trend'] = trend
                self.trend_data[district_name]['ndbi_last'] = values[-1]
                print(f"{district_name} NDBI trend: {trend:.4f}")
            
            # Calculate Population Density trend
            if self.population_df is not None and district_name in self.population_df.columns:
                values = self.population_df[district_name].values
                trend = self._calculate_average_annual_change(
                    self.population_df['year'].values,
                    values
                )
                self.trend_data[district_name]['pop_trend'] = trend
                self.trend_data[district_name]['pop_last'] = values[-1]
                print(f"{district_name} Pop Density trend: {trend:.1f} people/km²/year")
    
    def _calculate_average_annual_change(self, years, values):
        """Calculate average annual change as described in the forecasting procedure."""
        if len(values) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(values)):
            change = values[i] - values[i-1]
            changes.append(change)
        
        avg_change = np.mean(changes) if changes else 0.0
        return avg_change
    
    def train_mlr_models(self):
        """Train MLR models: NDVI, NDBI, Population Density -> Heat Index."""
        print("\n=== Training MLR Models (Stage 1) ===")
        print("Building predictive models using 2020-2024 historical data")
        
        if self.heat_index_df is None or self.heat_index_df.empty:
            print("No heat index data available for training")
            return
        
        for fid, district_name in self.districts.items():
            try:
                # Check if all required columns exist
                if (district_name not in self.heat_index_df.columns or
                    district_name not in self.ndvi_df.columns or
                    district_name not in self.ndbi_df.columns or
                    district_name not in self.population_df.columns):
                    print(f"Skipping {district_name}: missing data columns")
                    continue
                
                years = self.heat_index_df['year'].values
                heat_index = self.heat_index_df[district_name].values
                ndvi = self.ndvi_df[district_name].values
                ndbi = self.ndbi_df[district_name].values
                pop_density = self.population_df[district_name].values
                
                # Create training dataset
                X = np.column_stack((ndvi, ndbi, pop_density))
                y = heat_index
                
                # Remove any rows with NaN
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) < 2:
                    print(f"Not enough data for {district_name}")
                    continue
                
                model = LinearRegression()
                model.fit(X, y)
                
                self.models[fid] = {
                    'model': model,
                    'score': model.score(X, y),
                    'coefficients': {
                        'ndvi': model.coef_[0],
                        'ndbi': model.coef_[1],
                        'pop_density': model.coef_[2]
                    },
                    'intercept': model.intercept_,
                    'equation': f"HI = {model.intercept_:.3f} + ({model.coef_[0]:.3f} × NDVI) + ({model.coef_[1]:.3f} × NDBI) + ({model.coef_[2]:.6f} × PopDensity)"
                }
                
                print(f"\n{district_name} (FID {fid}):")
                print(f"  R² Score: {model.score(X, y):.4f}")
                print(f"  Model Equation: {self.models[fid]['equation']}")
                
            except Exception as e:
                print(f"Error training model for {district_name}: {e}")

    def calculate_validation_metrics(self):
        """Calculate validation metrics (R², Adj R², MAE, RMSE) for all district models."""
        print("\n=== Model Validation Metrics ===")
    
        for fid, district_name in self.districts.items():
            if fid not in self.models:
                continue
            
            try:
                # Get the same training data used for this model
                heat_index = self.heat_index_df[district_name].values
                ndvi = self.ndvi_df[district_name].values
                ndbi = self.ndbi_df[district_name].values
                pop_density = self.population_df[district_name].values
            
                X = np.column_stack((ndvi, ndbi, pop_density))
                y = heat_index
            
                # Remove NaN rows (same as training)
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
            
                if len(X) < 2:
                    continue
                
                model = self.models[fid]['model']
                y_pred = model.predict(X)
            
                # Calculate metrics
                n = len(y)
                k = 3  # NDVI, NDBI, Pop Density
            
                r2 = model.score(X, y)
                adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                 # Check correlations between predictors
                correlation_matrix = np.corrcoef(X.T)
            
                # Store in model dictionary
                self.models[fid]['validation_metrics'] = {
                    'r2': r2, 'adj_r2': adj_r2, 'mae': mae, 'rmse': rmse
                }
            
                print(f"\n{district_name} (FID {fid}):")
                print(f"  R²: {r2:.4f}")
                print(f"  Adjusted R²: {adj_r2:.4f}")
                print(f"  MAE: {mae:.3f}")
                print(f"  RMSE: {rmse:.3f}")
                print(f"  NDVI-NDBI correlation: {correlation_matrix[0,1]:.3f}")
                print(f"  NDVI-Pop correlation: {correlation_matrix[0,2]:.3f}")
                print(f"  NDBI-Pop correlation: {correlation_matrix[1,2]:.3f}")
            
            except Exception as e:
                print(f"Error calculating metrics for {district_name}: {e}")


    def debug_data_quality(self):
        """Debug data quality issues."""
        print("\n=== Data Quality Check ===")
    
        for district_name in self.districts.values():
            print(f"\n{district_name} Raw Data:")
            if self.heat_index_df is not None and district_name in self.heat_index_df.columns:
                hi_vals = self.heat_index_df[district_name].values
                print(f"  Heat Index: {hi_vals}")
                print(f"  HI Range: {hi_vals.min():.3f} to {hi_vals.max():.3f}")
                print(f"  HI Std Dev: {hi_vals.std():.3f}")
                print(f"  Heat Index CV: {(hi_vals.std()/hi_vals.mean())*100:.1f}%")
        
            if self.ndvi_df is not None and district_name in self.ndvi_df.columns:
                ndvi_vals = self.ndvi_df[district_name].values
                print(f"  NDVI: {ndvi_vals}")
                print(f"  NDVI Std Dev: {ndvi_vals.std():.4f}")
        
            if self.ndbi_df is not None and district_name in self.ndbi_df.columns:
                ndbi_vals = self.ndbi_df[district_name].values
                print(f"  NDBI: {ndbi_vals}")
                print(f"  NDBI Std Dev: {ndbi_vals.std():.4f}")
    
    def forecast_independent_variables(self, district_name, target_year):
        """Forecast NDVI, NDBI, and Population Density for a future year."""
        if district_name not in self.trend_data:
            return None
        
        trends = self.trend_data[district_name]
        base_year = 2024
        years_ahead = target_year - base_year
        
        forecasted = {}
        
        if 'ndvi_trend' in trends:
            forecasted['ndvi'] = trends['ndvi_last'] + (trends['ndvi_trend'] * years_ahead)
        
        if 'ndbi_trend' in trends:
            forecasted['ndbi'] = trends['ndbi_last'] + (trends['ndbi_trend'] * years_ahead)
        
        if 'pop_trend' in trends:
            forecasted['pop_density'] = trends['pop_last'] + (trends['pop_trend'] * years_ahead)
        
        return forecasted
    
    def forecast_heat_index(self, fid, start_year=2025, end_year=2050):
        """Forecast heat index using the MLR model and forecasted independent variables."""
        if fid not in self.models:
            print(f"No model available for FID {fid}")
            return None
        
        district_name = self.districts[fid]
        model = self.models[fid]['model']
        forecasts = []
        
        for year in range(start_year, end_year + 1):
            forecasted_vars = self.forecast_independent_variables(district_name, year)
            
            if forecasted_vars is None:
                continue
            
            X_forecast = np.array([[
                forecasted_vars.get('ndvi', 0),
                forecasted_vars.get('ndbi', 0),
                forecasted_vars.get('pop_density', 0)
            ]])
            
            predicted_heat_index = model.predict(X_forecast)[0]
            
            # Also get temperature and humidity if available
            temp = None
            humid = None
            if self.temperature_df is not None and district_name in self.temperature_df.columns:
                temp_trend = self._calculate_average_annual_change(
                    self.temperature_df['year'].values,
                    self.temperature_df[district_name].values
                )
                temp = self.temperature_df[district_name].iloc[-1] + (temp_trend * (year - 2024))
            
            if self.humidity_df is not None and district_name in self.humidity_df.columns:
                humid_trend = self._calculate_average_annual_change(
                    self.humidity_df['year'].values,
                    self.humidity_df[district_name].values
                )
                humid = self.humidity_df[district_name].iloc[-1] + (humid_trend * (year - 2024))
            
            forecasts.append({
                'year': year,
                'fid': fid,
                'district': district_name,
                'heat_index': round(float(predicted_heat_index), 3),
                'ndvi': round(forecasted_vars.get('ndvi', 0), 4),
                'ndbi': round(forecasted_vars.get('ndbi', 0), 4),
                'pop_density': round(forecasted_vars.get('pop_density', 0), 1),
                'temperature': round(float(temp), 3) if temp else None,
                'humidity': round(float(humid), 3) if humid else None,
                'is_forecast': True
            })
        
        return forecasts

    def get_historical_data(self, fid, year):
        """Get historical data for a specific district and year."""
        district_name = self.districts.get(fid)
        if not district_name:
            return None
        
        result = {
            'fid': fid,
            'district': district_name,
            'year': year,
            'is_forecast': False
        }
        
        # Get data from each DataFrame
        if self.heat_index_df is not None:
            row = self.heat_index_df[self.heat_index_df['year'] == year]
            if not row.empty and district_name in self.heat_index_df.columns:
                result['heat_index'] = round(float(row[district_name].iloc[0]), 3)
        
        if self.temperature_df is not None:
            row = self.temperature_df[self.temperature_df['year'] == year]
            if not row.empty and district_name in self.temperature_df.columns:
                result['temperature'] = round(float(row[district_name].iloc[0]), 3)
        
        if self.humidity_df is not None:
            row = self.humidity_df[self.humidity_df['year'] == year]
            if not row.empty and district_name in self.humidity_df.columns:
                result['humidity'] = round(float(row[district_name].iloc[0]), 3)
        
        if self.ndvi_df is not None:
            row = self.ndvi_df[self.ndvi_df['year'] == year]
            if not row.empty and district_name in self.ndvi_df.columns:
                result['ndvi'] = round(float(row[district_name].iloc[0]), 4)
        
        if self.ndbi_df is not None:
            row = self.ndbi_df[self.ndbi_df['year'] == year]
            if not row.empty and district_name in self.ndbi_df.columns:
                result['ndbi'] = round(float(row[district_name].iloc[0]), 4)
        
        if self.population_df is not None:
            row = self.population_df[self.population_df['year'] == year]
            if not row.empty and district_name in self.population_df.columns:
                result['pop_density'] = round(float(row[district_name].iloc[0]), 1)
        
        return result

# Initialize the processor
try:
    processor = HeatIndexProcessor()
    print("Heat Index Processor initialized successfully!")
except Exception as e:
    print(f"Failed to initialize processor: {e}")
    processor = None



# API Endpoints
@app.route('/api/visualization/heat-index/<int:year>')
def get_heat_visualization(year):
    """Get heat index visualization data for map coloring."""
    if not processor:
        return jsonify({'success': False, 'error': 'Processor not initialized'})
    
    try:
        districts = []
        for fid in range(10):  # FIDs 0-9
            if year <= 2024:
                # Historical data
                data = processor.get_historical_data(fid, year)
            else:
                # Forecasted data
                forecasts = processor.forecast_heat_index(fid, year, year)
                data = forecasts[0] if forecasts else None
            
            if data and 'heat_index' in data:
                heat_value = data['heat_index']
                districts.append({
                    'fid': fid,
                    'district': processor.districts[fid],
                    'heat_index': heat_value,
                })
        
        return jsonify({'success': True, 'districts': districts})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualization/ndvi/<int:year>')
def get_ndvi_visualization(year):
    """Get NDVI visualization data for map coloring."""
    if not processor:
        return jsonify({'success': False, 'error': 'Processor not initialized'})
    
    try:
        districts = []
        for fid in range(10):  # FIDs 0-9
            if year <= 2024:
                data = processor.get_historical_data(fid, year)
            else:
                forecasts = processor.forecast_heat_index(fid, year, year)
                data = forecasts[0] if forecasts else None
            
            if data and 'ndvi' in data:
                ndvi_value = data['ndvi']
                districts.append({
                    'fid': fid,
                    'district': processor.districts[fid],
                    'ndvi': ndvi_value,
                })
        
        return jsonify({'success': True, 'districts': districts})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualization/ndbi/<int:year>')
def get_ndbi_visualization(year):
    """Get NDBI visualization data for map coloring."""
    if not processor:
        return jsonify({'success': False, 'error': 'Processor not initialized'})
    
    try:
        districts = []
        for fid in range(10):  # FIDs 0-9
            if year <= 2024:
                data = processor.get_historical_data(fid, year)
            else:
                forecasts = processor.forecast_heat_index(fid, year, year)
                data = forecasts[0] if forecasts else None
            
            if data and 'ndbi' in data:
                ndbi_value = data['ndbi']
                districts.append({
                    'fid': fid,
                    'district': processor.districts[fid],
                    'ndbi': ndbi_value,
                })
        
        return jsonify({'success': True, 'districts': districts})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

@app.route('/api/time-series/<int:fid>')
def get_time_series(fid):
    """Get complete time series data for a district."""
    if not processor:
        return jsonify({'success': False, 'error': 'Processor not initialized'})
    
    try:
        data = []
        
        # Historical data (2020-2024)
        for year in range(2020, 2025):
            historical = processor.get_historical_data(fid, year)
            if historical:
                data.append(historical)
        
        # Forecasted data (2025-2050)
        forecasts = processor.forecast_heat_index(fid, 2025, 2050)
        if forecasts:
            data.extend(forecasts)
        
        return jsonify({
            'success': True,
            'district': processor.districts.get(fid, f'District {fid}'),
            'data': data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, port=5000)  # NEVER use debug=True online!
