"""
Predictive Policing ML Desktop Application
A comprehensive offline app for crime prediction and analysis

Requirements:
pip install PyQt6 pandas numpy scikit-learn matplotlib seaborn folium xgboost
pip install PyQt6-WebEngine  # Optional for maps, will work without it
"""

import sys
import os
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.cluster import DBSCAN, KMeans
import xgboost as xgb

# Visualization
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend for PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster

# PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget,
                             QTableWidget, QTableWidgetItem, QFileDialog,
                             QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                             QProgressBar, QTextEdit, QGroupBox, QGridLayout,
                             QLineEdit, QCheckBox, QSplitter, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QFont, QPalette, QColor, QDesktopServices

# Try to import WebEngine, but make it optional
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False
    print("Note: PyQt6-WebEngine not available. Maps will open in browser instead.")
    print("To enable in-app maps, run: pip install PyQt6-WebEngine")


class DataProcessor:
    """Handles all data processing and feature engineering"""
    
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load data from CSV or Excel"""
        try:
            if filepath.endswith('.csv'):
                self.data = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(filepath)
            return True, f"Loaded {len(self.data)} records"
        except Exception as e:
            return False, str(e)
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample crime data for testing"""
        np.random.seed(42)
        
        # Generate timestamps
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=np.random.randint(0, 365),
                                       hours=np.random.randint(0, 24))
                for _ in range(n_samples)]
        
        # Crime types
        crime_types = ['Theft', 'Burglary', 'Assault', 'Vandalism', 'Robbery', 
                      'Vehicle Theft', 'Drug Offense']
        
        # Generate spatial coordinates (simulating a city)
        lat_center, lon_center = 40.7128, -74.0060  # NYC-like coordinates
        
        data = {
            'date': dates,
            'crime_type': np.random.choice(crime_types, n_samples, 
                                          p=[0.25, 0.15, 0.20, 0.10, 0.10, 0.12, 0.08]),
            'latitude': np.random.normal(lat_center, 0.05, n_samples),
            'longitude': np.random.normal(lon_center, 0.05, n_samples),
            'district': np.random.randint(1, 11, n_samples),
            'resolved': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        self.data = pd.DataFrame(data)
        
        # Add time features
        self.data['hour'] = pd.to_datetime(self.data['date']).dt.hour
        self.data['day_of_week'] = pd.to_datetime(self.data['date']).dt.dayofweek
        self.data['month'] = pd.to_datetime(self.data['date']).dt.month
        
        return True, f"Generated {n_samples} sample records"
    
    def engineer_features(self):
        """Create additional features for ML models"""
        if self.data is None:
            return False, "No data loaded"
        
        df = self.data.copy()
        
        # Temporal features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 6)).astype(int)
        
        # Spatial features (if lat/lon available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from city center
            lat_center = df['latitude'].mean()
            lon_center = df['longitude'].mean()
            df['dist_from_center'] = np.sqrt(
                (df['latitude'] - lat_center)**2 + 
                (df['longitude'] - lon_center)**2
            )
        
        self.data = df
        return True, f"Engineered features. Total columns: {len(df.columns)}"
    
    def prepare_for_training(self, target_column, feature_columns=None):
        """Prepare data for ML training"""
        if self.data is None:
            return None, None, None, None, "No data loaded"
        
        df = self.data.copy()
        
        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns 
                             if col != target_column and col != 'date']
        
        X = df[feature_columns]
        
        # Handle target
        if df[target_column].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df[target_column])
            self.label_encoders[target_column] = le
        else:
            y = df[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, "Data prepared successfully"


class MLModelManager:
    """Manages machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize available ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a specific model"""
        if model_name not in self.models:
            return False, "Model not found"
        
        try:
            model = self.models[model_name]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            results = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'y_test': y_test,
                'y_pred': y_pred_test
            }
            
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_
            
            self.trained_models[model_name] = model
            self.results[model_name] = results
            
            return True, f"{model_name} trained successfully"
        except Exception as e:
            return False, str(e)
    
    def predict(self, model_name, X):
        """Make predictions with trained model"""
        if model_name not in self.trained_models:
            return None, "Model not trained"
        
        try:
            model = self.trained_models[model_name]
            predictions = model.predict(X)
            probabilities = None
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            
            return (predictions, probabilities), "Predictions made successfully"
        except Exception as e:
            return None, str(e)
    
    def save_model(self, model_name, filepath):
        """Save trained model to disk"""
        if model_name not in self.trained_models:
            return False, "Model not trained"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.trained_models[model_name], f)
            return True, "Model saved successfully"
        except Exception as e:
            return False, str(e)
    
    def load_model(self, model_name, filepath):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.trained_models[model_name] = model
            return True, "Model loaded successfully"
        except Exception as e:
            return False, str(e)


class BiasAnalyzer:
    """Analyze model fairness and bias"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_disparate_impact(self, y_true, y_pred, sensitive_feature):
        """Calculate disparate impact ratio"""
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        # Calculate selection rates per group
        groups = df.groupby('sensitive')['y_pred'].mean()
        
        if len(groups) < 2:
            return None, "Need at least 2 groups"
        
        # Disparate impact: min_rate / max_rate
        disparate_impact = groups.min() / groups.max() if groups.max() > 0 else 0
        
        return disparate_impact, groups.to_dict()
    
    def calculate_demographic_parity(self, y_pred, sensitive_feature):
        """Calculate demographic parity difference"""
        df = pd.DataFrame({
            'y_pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        rates = df.groupby('sensitive')['y_pred'].mean()
        parity_diff = rates.max() - rates.min()
        
        return parity_diff, rates.to_dict()
    
    def calculate_equal_opportunity(self, y_true, y_pred, sensitive_feature):
        """Calculate equal opportunity difference (TPR difference)"""
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'sensitive': sensitive_feature
        })
        
        # True positive rates per group
        tpr_by_group = {}
        for group in df['sensitive'].unique():
            group_data = df[df['sensitive'] == group]
            tp = ((group_data['y_true'] == 1) & (group_data['y_pred'] == 1)).sum()
            p = (group_data['y_true'] == 1).sum()
            tpr_by_group[group] = tp / p if p > 0 else 0
        
        tpr_values = list(tpr_by_group.values())
        eo_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        
        return eo_diff, tpr_by_group


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.data_processor = DataProcessor()
        self.model_manager = MLModelManager()
        self.model_manager.initialize_models()
        self.bias_analyzer = BiasAnalyzer()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Predictive Policing ML Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("🚔 Predictive Policing ML Platform")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_tab()
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_visualization_tab()
        self.create_map_tab()
        self.create_analysis_tab()
        self.create_fairness_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_data_tab(self):
        """Create data management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls group
        controls_group = QGroupBox("Data Management")
        controls_layout = QHBoxLayout()
        
        load_btn = QPushButton("📁 Load Data (CSV/Excel)")
        load_btn.clicked.connect(self.load_data)
        controls_layout.addWidget(load_btn)
        
        generate_btn = QPushButton("🎲 Generate Sample Data")
        generate_btn.clicked.connect(self.generate_sample_data)
        controls_layout.addWidget(generate_btn)
        
        engineer_btn = QPushButton("⚙️ Engineer Features")
        engineer_btn.clicked.connect(self.engineer_features)
        controls_layout.addWidget(engineer_btn)
        
        export_btn = QPushButton("💾 Export Data")
        export_btn.clicked.connect(self.export_data)
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.data_info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Data preview table
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        preview_layout.addWidget(self.data_table)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        self.tabs.addTab(tab, "📊 Data")
    
    def create_training_tab(self):
        """Create model training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Training controls
        controls_group = QGroupBox("Training Configuration")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Target Column:"), 0, 0)
        self.target_combo = QComboBox()
        controls_layout.addWidget(self.target_combo, 0, 1)
        
        controls_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_manager.models.keys())
        controls_layout.addWidget(self.model_combo, 1, 1)
        
        train_btn = QPushButton("🚀 Train Model")
        train_btn.clicked.connect(self.train_model)
        controls_layout.addWidget(train_btn, 2, 0, 1, 2)
        
        train_all_btn = QPushButton("🔥 Train All Models")
        train_all_btn.clicked.connect(self.train_all_models)
        controls_layout.addWidget(train_all_btn, 3, 0, 1, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Progress bar
        self.training_progress = QProgressBar()
        layout.addWidget(self.training_progress)
        
        # Results
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()
        self.training_results = QTextEdit()
        self.training_results.setReadOnly(True)
        results_layout.addWidget(self.training_results)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Model management
        management_group = QGroupBox("Model Management")
        management_layout = QHBoxLayout()
        
        save_model_btn = QPushButton("💾 Save Model")
        save_model_btn.clicked.connect(self.save_model)
        management_layout.addWidget(save_model_btn)
        
        load_model_btn = QPushButton("📁 Load Model")
        load_model_btn.clicked.connect(self.load_model)
        management_layout.addWidget(load_model_btn)
        
        management_layout.addStretch()
        management_group.setLayout(management_layout)
        layout.addWidget(management_group)
        
        self.tabs.addTab(tab, "🤖 Training")
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Prediction controls
        controls_group = QGroupBox("Make Predictions")
        controls_layout = QVBoxLayout()
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.pred_model_combo = QComboBox()
        model_layout.addWidget(self.pred_model_combo)
        predict_btn = QPushButton("🎯 Predict on Test Data")
        predict_btn.clicked.connect(self.make_predictions)
        model_layout.addWidget(predict_btn)
        controls_layout.addLayout(model_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Prediction results
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout()
        self.prediction_results = QTextEdit()
        self.prediction_results.setReadOnly(True)
        results_layout.addWidget(self.prediction_results)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Prediction table
        table_group = QGroupBox("Predictions Preview")
        table_layout = QVBoxLayout()
        self.prediction_table = QTableWidget()
        table_layout.addWidget(self.prediction_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        self.tabs.addTab(tab, "🎯 Predictions")
    
    def create_visualization_tab(self):
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout()
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Confusion Matrix",
            "ROC Curve",
            "Feature Importance",
            "Model Comparison",
            "Crime Time Series",
            "Crime by Type"
        ])
        controls_layout.addWidget(QLabel("Chart Type:"))
        controls_layout.addWidget(self.viz_type_combo)
        
        generate_viz_btn = QPushButton("📈 Generate Chart")
        generate_viz_btn.clicked.connect(self.generate_visualization)
        controls_layout.addWidget(generate_viz_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Chart area
        self.chart_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        layout.addWidget(self.chart_canvas)
        
        self.tabs.addTab(tab, "📈 Visualizations")
    
    def create_map_tab(self):
        """Create interactive map tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Map Controls")
        controls_layout = QHBoxLayout()
        
        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems([
            "Crime Heat Map",
            "Crime Clusters",
            "Prediction Heat Map",
            "District Analysis"
        ])
        controls_layout.addWidget(QLabel("Map Type:"))
        controls_layout.addWidget(self.map_type_combo)
        
        generate_map_btn = QPushButton("🗺️ Generate Map")
        generate_map_btn.clicked.connect(self.generate_map)
        controls_layout.addWidget(generate_map_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Map view - use WebEngine if available, otherwise show instructions
        if WEBENGINE_AVAILABLE:
            self.map_view = QWebEngineView()
            layout.addWidget(self.map_view)
        else:
            self.map_view = QTextEdit()
            self.map_view.setReadOnly(True)
            self.map_view.setHtml("""
            <div style='padding: 20px; font-family: Arial;'>
                <h2>📍 Map Viewer Not Available</h2>
                <p>To view maps in the application, install PyQt6-WebEngine:</p>
                <pre style='background: #f0f0f0; padding: 10px; border-radius: 5px;'>
pip install PyQt6-WebEngine
                </pre>
                <p><b>Don't worry!</b> Maps will still be generated and opened in your default web browser.</p>
            </div>
            """)
            layout.addWidget(self.map_view)
        
        self.tabs.addTab(tab, "🗺️ Maps")
    
    def create_analysis_tab(self):
        """Create statistical analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Analysis Tools")
        controls_layout = QHBoxLayout()
        
        correlation_btn = QPushButton("📊 Correlation Analysis")
        correlation_btn.clicked.connect(self.correlation_analysis)
        controls_layout.addWidget(correlation_btn)
        
        temporal_btn = QPushButton("⏰ Temporal Analysis")
        temporal_btn.clicked.connect(self.temporal_analysis)
        controls_layout.addWidget(temporal_btn)
        
        spatial_btn = QPushButton("📍 Spatial Analysis")
        spatial_btn.clicked.connect(self.spatial_analysis)
        controls_layout.addWidget(spatial_btn)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results
        self.analysis_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        layout.addWidget(self.analysis_canvas)
        
        self.tabs.addTab(tab, "🔍 Analysis")
    
    def create_fairness_tab(self):
        """Create fairness and bias analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Fairness Analysis")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Model:"), 0, 0)
        self.fairness_model_combo = QComboBox()
        controls_layout.addWidget(self.fairness_model_combo, 0, 1)
        
        controls_layout.addWidget(QLabel("Sensitive Attribute:"), 1, 0)
        self.sensitive_attr_combo = QComboBox()
        controls_layout.addWidget(self.sensitive_attr_combo, 1, 1)
        
        analyze_btn = QPushButton("⚖️ Analyze Fairness")
        analyze_btn.clicked.connect(self.analyze_fairness)
        controls_layout.addWidget(analyze_btn, 2, 0, 1, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results
        results_group = QGroupBox("Fairness Metrics")
        results_layout = QVBoxLayout()
        self.fairness_results = QTextEdit()
        self.fairness_results.setReadOnly(True)
        results_layout.addWidget(self.fairness_results)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Visualization
        self.fairness_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        layout.addWidget(self.fairness_canvas)
        
        self.tabs.addTab(tab, "⚖️ Fairness")
    
    # Data management methods
    def load_data(self):
        """Load data from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if filepath:
            success, message = self.data_processor.load_data(filepath)
            if success:
                self.update_data_display()
                self.statusBar().showMessage(message)
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)
    
    def generate_sample_data(self):
        """Generate sample crime data"""
        success, message = self.data_processor.generate_sample_data(1000)
        if success:
            self.update_data_display()
            self.statusBar().showMessage(message)
            QMessageBox.information(self, "Success", message)
    
    def engineer_features(self):
        """Engineer features from data"""
        success, message = self.data_processor.engineer_features()
        if success:
            self.update_data_display()
            self.statusBar().showMessage(message)
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Warning", message)
    
    def export_data(self):
        """Export data to CSV"""
        if self.data_processor.data is None:
            QMessageBox.warning(self, "Warning", "No data to export")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv)"
        )
        if filepath:
            try:
                self.data_processor.data.to_csv(filepath, index=False)
                QMessageBox.information(self, "Success", "Data exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
    
    def update_data_display(self):
        """Update data display in UI"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data
        
        # Update info text
        info = f"Rows: {len(df)}\n"
        info += f"Columns: {len(df.columns)}\n"
        info += f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
        info += "Columns:\n" + "\n".join(f"  - {col} ({df[col].dtype})" for col in df.columns)
        self.data_info_text.setText(info)
        
        # Update table
        self.data_table.setRowCount(min(100, len(df)))
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i in range(min(100, len(df))):
            for j, col in enumerate(df.columns):
                self.data_table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        
        # Update combo boxes
        self.target_combo.clear()
        self.target_combo.addItems(df.columns.tolist())
        self.sensitive_attr_combo.clear()
        self.sensitive_attr_combo.addItems(df.columns.tolist())
    
    # Training methods
    def train_model(self):
        """Train selected model"""
        if self.data_processor.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        target = self.target_combo.currentText()
        model_name = self.model_combo.currentText()
        
        if not target:
            QMessageBox.warning(self, "Warning", "Please select target column")
            return
        
        self.statusBar().showMessage(f"Training {model_name}...")
        self.training_progress.setValue(30)
        QApplication.processEvents()  # Update UI
        
        # Prepare data
        X_train, X_test, y_train, y_test, message = self.data_processor.prepare_for_training(target)
        
        if X_train is None:
            QMessageBox.critical(self, "Error", message)
            self.training_progress.setValue(0)
            return
        
        self.training_progress.setValue(50)
        QApplication.processEvents()
        
        # Train model
        success, message = self.model_manager.train_model(
            model_name, X_train, y_train, X_test, y_test
        )
        
        self.training_progress.setValue(100)
        
        if success:
            results = self.model_manager.results[model_name]
            result_text = f"=== {model_name} Results ===\n\n"
            result_text += f"Train Accuracy: {results['train_accuracy']:.4f}\n"
            result_text += f"Test Accuracy: {results['test_accuracy']:.4f}\n"
            result_text += f"Precision: {results['precision']:.4f}\n"
            result_text += f"Recall: {results['recall']:.4f}\n"
            result_text += f"F1-Score: {results['f1']:.4f}\n"
            
            self.training_results.append(result_text)
            self.statusBar().showMessage(message)
            
            # Update prediction combo
            self.pred_model_combo.clear()
            self.pred_model_combo.addItems(self.model_manager.trained_models.keys())
            self.fairness_model_combo.clear()
            self.fairness_model_combo.addItems(self.model_manager.trained_models.keys())
            
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
        
        self.training_progress.setValue(0)
    
    def train_all_models(self):
        """Train all available models"""
        if self.data_processor.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        target = self.target_combo.currentText()
        if not target:
            QMessageBox.warning(self, "Warning", "Please select target column")
            return
        
        # Prepare data once
        X_train, X_test, y_train, y_test, message = self.data_processor.prepare_for_training(target)
        
        if X_train is None:
            QMessageBox.critical(self, "Error", message)
            return
        
        self.training_results.clear()
        
        # Train each model
        for i, model_name in enumerate(self.model_manager.models.keys()):
            progress = int((i + 1) / len(self.model_manager.models) * 100)
            self.training_progress.setValue(progress)
            self.statusBar().showMessage(f"Training {model_name}...")
            QApplication.processEvents()
            
            success, message = self.model_manager.train_model(
                model_name, X_train, y_train, X_test, y_test
            )
            
            if success:
                results = self.model_manager.results[model_name]
                result_text = f"\n=== {model_name} ===\n"
                result_text += f"Accuracy: {results['test_accuracy']:.4f} | "
                result_text += f"F1: {results['f1']:.4f} | "
                result_text += f"Precision: {results['precision']:.4f} | "
                result_text += f"Recall: {results['recall']:.4f}\n"
                self.training_results.append(result_text)
        
        self.training_progress.setValue(0)
        self.statusBar().showMessage("All models trained successfully")
        
        # Update combos
        self.pred_model_combo.clear()
        self.pred_model_combo.addItems(self.model_manager.trained_models.keys())
        self.fairness_model_combo.clear()
        self.fairness_model_combo.addItems(self.model_manager.trained_models.keys())
        
        QMessageBox.information(self, "Success", "All models trained successfully")
    
    def save_model(self):
        """Save trained model"""
        model_name = self.model_combo.currentText()
        if model_name not in self.model_manager.trained_models:
            QMessageBox.warning(self, "Warning", "Model not trained yet")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Model", f"{model_name}.pkl", "Pickle Files (*.pkl)"
        )
        if filepath:
            success, message = self.model_manager.save_model(model_name, filepath)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)
    
    def load_model(self):
        """Load trained model"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Pickle Files (*.pkl)"
        )
        if filepath:
            model_name = self.model_combo.currentText()
            success, message = self.model_manager.load_model(model_name, filepath)
            if success:
                QMessageBox.information(self, "Success", message)
                self.pred_model_combo.clear()
                self.pred_model_combo.addItems(self.model_manager.trained_models.keys())
            else:
                QMessageBox.critical(self, "Error", message)
    
    # Prediction methods
    def make_predictions(self):
        """Make predictions with trained model"""
        model_name = self.pred_model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "Warning", "No trained model selected")
            return
        
        target = self.target_combo.currentText()
        X_train, X_test, y_train, y_test, _ = self.data_processor.prepare_for_training(target)
        
        if X_test is None:
            QMessageBox.warning(self, "Warning", "Please prepare data first")
            return
        
        result, message = self.model_manager.predict(model_name, X_test)
        
        if result is not None:
            predictions, probabilities = result
            
            # Display results
            result_text = f"=== Predictions using {model_name} ===\n\n"
            result_text += f"Total predictions: {len(predictions)}\n"
            
            if probabilities is not None:
                result_text += f"Average confidence: {probabilities.max(axis=1).mean():.4f}\n"
            
            # Show distribution
            unique, counts = np.unique(predictions, return_counts=True)
            result_text += "\nPrediction Distribution:\n"
            for val, count in zip(unique, counts):
                result_text += f"  Class {val}: {count} ({count/len(predictions)*100:.1f}%)\n"
            
            self.prediction_results.setText(result_text)
            
            # Update table
            self.prediction_table.setRowCount(min(100, len(predictions)))
            self.prediction_table.setColumnCount(3 if probabilities is not None else 2)
            headers = ["Index", "Prediction"]
            if probabilities is not None:
                headers.append("Confidence")
            self.prediction_table.setHorizontalHeaderLabels(headers)
            
            for i in range(min(100, len(predictions))):
                self.prediction_table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.prediction_table.setItem(i, 1, QTableWidgetItem(str(predictions[i])))
                if probabilities is not None:
                    conf = probabilities[i].max()
                    self.prediction_table.setItem(i, 2, QTableWidgetItem(f"{conf:.4f}"))
            
            self.statusBar().showMessage("Predictions completed")
        else:
            QMessageBox.critical(self, "Error", message)
    
    # Visualization methods
    def generate_visualization(self):
        """Generate selected visualization"""
        viz_type = self.viz_type_combo.currentText()
        
        self.chart_canvas.figure.clear()
        
        if viz_type == "Confusion Matrix":
            self.plot_confusion_matrix()
        elif viz_type == "ROC Curve":
            self.plot_roc_curve()
        elif viz_type == "Feature Importance":
            self.plot_feature_importance()
        elif viz_type == "Model Comparison":
            self.plot_model_comparison()
        elif viz_type == "Crime Time Series":
            self.plot_time_series()
        elif viz_type == "Crime by Type":
            self.plot_crime_types()
        
        self.chart_canvas.draw()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for trained model"""
        if not self.model_manager.results:
            return
        
        model_name = list(self.model_manager.results.keys())[0]
        cm = self.model_manager.results[model_name]['confusion_matrix']
        
        ax = self.chart_canvas.figure.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def plot_roc_curve(self):
        """Plot ROC curves for all models"""
        if not self.model_manager.results:
            return
        
        ax = self.chart_canvas.figure.add_subplot(111)
        
        for model_name, results in self.model_manager.results.items():
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            # Binary classification ROC
            if len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if not self.model_manager.results:
            return
        
        # Find a model with feature importance
        for model_name, results in self.model_manager.results.items():
            if 'feature_importance' in results:
                importance = results['feature_importance']
                
                # Get feature names
                if self.data_processor.data is not None:
                    target = self.target_combo.currentText()
                    features = [col for col in self.data_processor.data.columns 
                              if col != target and col != 'date']
                    features = features[:len(importance)]
                else:
                    features = [f'Feature {i}' for i in range(len(importance))]
                
                # Sort by importance
                indices = np.argsort(importance)[::-1][:15]
                
                ax = self.chart_canvas.figure.add_subplot(111)
                ax.barh(range(len(indices)), importance[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([features[i] for i in indices])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top 15 Feature Importance - {model_name}')
                ax.invert_yaxis()
                break
    
    def plot_model_comparison(self):
        """Compare all trained models"""
        if not self.model_manager.results:
            return
        
        models = list(self.model_manager.results.keys())
        metrics = ['test_accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        data = []
        for metric in metrics:
            data.append([self.model_manager.results[m][metric] for m in models])
        
        x = np.arange(len(models))
        width = 0.2
        
        ax = self.chart_canvas.figure.add_subplot(111)
        for i, (metric_data, metric_name) in enumerate(zip(data, metric_names)):
            ax.bar(x + i * width, metric_data, width, label=metric_name)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_time_series(self):
        """Plot crime time series"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data.copy()
        if 'date' not in df.columns:
            return
        
        df['date'] = pd.to_datetime(df['date'])
        daily_counts = df.groupby(df['date'].dt.date).size()
        
        ax = self.chart_canvas.figure.add_subplot(111)
        ax.plot(daily_counts.index, daily_counts.values)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Crimes')
        ax.set_title('Crime Time Series')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def plot_crime_types(self):
        """Plot crime distribution by type"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data
        if 'crime_type' not in df.columns:
            return
        
        crime_counts = df['crime_type'].value_counts()
        
        ax = self.chart_canvas.figure.add_subplot(111)
        ax.bar(range(len(crime_counts)), crime_counts.values)
        ax.set_xticks(range(len(crime_counts)))
        ax.set_xticklabels(crime_counts.index, rotation=45, ha='right')
        ax.set_xlabel('Crime Type')
        ax.set_ylabel('Count')
        ax.set_title('Crime Distribution by Type')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Map methods
    def generate_map(self):
        """Generate interactive map"""
        if self.data_processor.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        df = self.data_processor.data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            QMessageBox.warning(self, "Warning", "Data must contain latitude and longitude columns")
            return
        
        map_type = self.map_type_combo.currentText()
        
        # Center map on data
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        if map_type == "Crime Heat Map":
            heat_data = [[row['latitude'], row['longitude']] 
                        for _, row in df.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)
        
        elif map_type == "Crime Clusters":
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in df.head(500).iterrows():  # Limit for performance
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=f"Crime: {row.get('crime_type', 'Unknown')}"
                ).add_to(marker_cluster)
        
        elif map_type == "District Analysis":
            if 'district' in df.columns:
                districts = df.groupby('district').agg({
                    'latitude': 'mean',
                    'longitude': 'mean'
                }).reset_index()
                
                for _, row in districts.iterrows():
                    count = len(df[df['district'] == row['district']])
                    folium.CircleMarker(
                        [row['latitude'], row['longitude']],
                        radius=count/10,
                        popup=f"District {row['district']}: {count} crimes",
                        color='red',
                        fill=True
                    ).add_to(m)
        
        # Save map
        map_file = 'crime_map.html'
        m.save(map_file)
        
        # Display or open in browser
        if WEBENGINE_AVAILABLE:
            with open(map_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            self.map_view.setHtml(html_content)
        else:
            # Open in default browser
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(map_file)}')
            QMessageBox.information(
                self, 
                "Map Generated", 
                f"Map opened in your default browser.\nFile saved as: {os.path.abspath(map_file)}"
            )
        
        self.statusBar().showMessage("Map generated successfully")
    
    # Analysis methods
    def correlation_analysis(self):
        """Perform correlation analysis"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data.select_dtypes(include=[np.number])
        
        if df.empty:
            QMessageBox.warning(self, "Warning", "No numeric columns for correlation")
            return
        
        self.analysis_canvas.figure.clear()
        ax = self.analysis_canvas.figure.add_subplot(111)
        
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix')
        
        self.analysis_canvas.draw()
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data
        if 'hour' not in df.columns:
            QMessageBox.warning(self, "Warning", "No temporal features available")
            return
        
        self.analysis_canvas.figure.clear()
        
        # Hour distribution
        ax1 = self.analysis_canvas.figure.add_subplot(221)
        df['hour'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_title('Crimes by Hour')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Count')
        
        # Day of week
        if 'day_of_week' in df.columns:
            ax2 = self.analysis_canvas.figure.add_subplot(222)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_counts = df['day_of_week'].value_counts().sort_index()
            ax2.bar(range(len(dow_counts)), dow_counts.values)
            ax2.set_xticks(range(len(dow_counts)))
            ax2.set_xticklabels([day_names[i] for i in dow_counts.index])
            ax2.set_title('Crimes by Day of Week')
            ax2.set_ylabel('Count')
        
        # Month
        if 'month' in df.columns:
            ax3 = self.analysis_canvas.figure.add_subplot(223)
            df['month'].value_counts().sort_index().plot(kind='bar', ax=ax3)
            ax3.set_title('Crimes by Month')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Count')
        
        # Weekend vs Weekday
        if 'is_weekend' in df.columns:
            ax4 = self.analysis_canvas.figure.add_subplot(224)
            weekend_counts = df['is_weekend'].value_counts()
            ax4.bar(['Weekday', 'Weekend'], 
                   [weekend_counts.get(0, 0), weekend_counts.get(1, 0)])
            ax4.set_title('Weekday vs Weekend')
            ax4.set_ylabel('Count')
        
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw()
    
    def spatial_analysis(self):
        """Analyze spatial patterns"""
        if self.data_processor.data is None:
            return
        
        df = self.data_processor.data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            QMessageBox.warning(self, "Warning", "No spatial features available")
            return
        
        self.analysis_canvas.figure.clear()
        
        # Spatial distribution
        ax1 = self.analysis_canvas.figure.add_subplot(121)
        ax1.scatter(df['longitude'], df['latitude'], alpha=0.5, s=10)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Crime Spatial Distribution')
        
        # District analysis
        if 'district' in df.columns:
            ax2 = self.analysis_canvas.figure.add_subplot(122)
            district_counts = df['district'].value_counts().sort_index()
            ax2.bar(district_counts.index, district_counts.values)
            ax2.set_xlabel('District')
            ax2.set_ylabel('Crime Count')
            ax2.set_title('Crimes by District')
        
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw()
    
    # Fairness methods
    def analyze_fairness(self):
        """Analyze model fairness"""
        model_name = self.fairness_model_combo.currentText()
        if not model_name or model_name not in self.model_manager.results:
            QMessageBox.warning(self, "Warning", "Please train a model first")
            return
        
        sensitive_attr = self.sensitive_attr_combo.currentText()
        if not sensitive_attr or self.data_processor.data is None:
            QMessageBox.warning(self, "Warning", "Please select sensitive attribute")
            return
        
        results = self.model_manager.results[model_name]
        y_true = results['y_test']
        y_pred = results['y_pred']
        
        # Get sensitive attribute values (use district as proxy)
        df = self.data_processor.data
        if sensitive_attr not in df.columns:
            QMessageBox.warning(self, "Warning", f"Column {sensitive_attr} not found")
            return
        
        # Sample sensitive feature to match test set size
        sensitive_feature = df[sensitive_attr].values[:len(y_true)]
        
        # Calculate fairness metrics
        di, di_groups = self.bias_analyzer.calculate_disparate_impact(
            y_true, y_pred, sensitive_feature
        )
        
        dp, dp_groups = self.bias_analyzer.calculate_demographic_parity(
            y_pred, sensitive_feature
        )
        
        eo, eo_groups = self.bias_analyzer.calculate_equal_opportunity(
            y_true, y_pred, sensitive_feature
        )
        
        # Display results
        result_text = f"=== Fairness Analysis for {model_name} ===\n\n"
        result_text += f"Sensitive Attribute: {sensitive_attr}\n\n"
        
        result_text += f"Disparate Impact Ratio: {di:.4f}\n"
        result_text += "  (Ideal = 1.0, Legal threshold often >= 0.8)\n\n"
        
        result_text += f"Demographic Parity Difference: {dp:.4f}\n"
        result_text += "  (Ideal = 0.0, Lower is better)\n\n"
        
        result_text += f"Equal Opportunity Difference: {eo:.4f}\n"
        result_text += "  (Ideal = 0.0, Lower is better)\n\n"
        
        result_text += "Selection Rates by Group:\n"
        for group, rate in dp_groups.items():
            result_text += f"  Group {group}: {rate:.4f}\n"
        
        self.fairness_results.setText(result_text)
        
        # Visualize
        self.fairness_canvas.figure.clear()
        
        ax1 = self.fairness_canvas.figure.add_subplot(121)
        groups = list(dp_groups.keys())
        rates = list(dp_groups.values())
        ax1.bar(groups, rates)
        ax1.set_xlabel('Group')
        ax1.set_ylabel('Selection Rate')
        ax1.set_title('Selection Rates by Group')
        ax1.axhline(y=np.mean(rates), color='r', linestyle='--', label='Average')
        ax1.legend()
        
        ax2 = self.fairness_canvas.figure.add_subplot(122)
        metrics_names = ['Disparate\nImpact', 'Demographic\nParity', 'Equal\nOpportunity']
        metrics_values = [di, dp, eo]
        colors = ['green' if di >= 0.8 else 'red', 
                 'green' if dp <= 0.1 else 'red',
                 'green' if eo <= 0.1 else 'red']
        ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.6)
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Fairness Metrics')
        ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='DI Threshold')
        ax2.axhline(y=0.1, color='blue', linestyle='--', alpha=0.5, label='DP/EO Threshold')
        ax2.legend()
        
        self.fairness_canvas.figure.tight_layout()
        self.fairness_canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide font
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()