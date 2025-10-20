"""
RiskGrid Flask Dashboard
Complete crime prediction dashboard with Flask
Run: python app.py
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ============================================
# DATA LOADING
# ============================================

def load_predictions():
    """Load prediction data"""
    possible_paths = [
        ("outputs/ultra/ultra_predictions.csv", "ultra_predicted", "ultra_error", "Ultra Model"),
        ("outputs/test_predictions.csv", "predicted_incidents", "prediction_error", "Advanced Model")
    ]
    
    for path, pred_col_name, error_col_name, source in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Limit to recent data for performance
                if len(df) > 200000:
                    df = df.tail(200000)
                
                df['time_window'] = pd.to_datetime(df['time_window'])
                
                # Detect columns
                if pred_col_name in df.columns:
                    pred_col = pred_col_name
                    error_col = error_col_name
                elif 'ultra_predicted' in df.columns:
                    pred_col = 'ultra_predicted'
                    error_col = 'ultra_error'
                elif 'predicted_incidents' in df.columns:
                    pred_col = 'predicted_incidents'
                    error_col = 'prediction_error'
                else:
                    continue
                
                # Calculate error if missing
                if error_col not in df.columns:
                    df[error_col] = abs(df['target'] - df[pred_col])
                
                # Add risk levels
                df['risk_level'] = pd.cut(
                    df[pred_col],
                    bins=[-np.inf, 1, 3, np.inf],
                    labels=['Low', 'Medium', 'High']
                )
                
                return df, pred_col, error_col, source
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    return None, None, None, None

def load_feature_importance():
    """Load feature importance"""
    paths = [
        "outputs/ultra/ultra_feature_importance.csv",
        "outputs/feature_importance.csv"
    ]
    
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                pass
    return None

# Load data once at startup
print("Loading RiskGrid data...")
predictions, pred_col, error_col, data_source = load_predictions()
feature_importance = load_feature_importance()

if predictions is None:
    print("ERROR: No prediction data found!")
else:
    print(f"✓ Loaded {len(predictions):,} predictions")
    print(f"✓ Prediction column: {pred_col}")
    print(f"✓ Error column: {error_col}")
    print(f"✓ Data source: {data_source}")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Main dashboard page"""
    if predictions is None:
        return render_template('error.html', 
                             message="No prediction data found. Run generate_predictions.py first.")
    
    # Get available times
    times = sorted(predictions['time_window'].unique())
    
    # Basic stats
    stats = {
        'total_records': len(predictions),
        'data_source': data_source,
        'pred_col': pred_col,
        'error_col': error_col,
        'date_range': f"{predictions['time_window'].min().date()} to {predictions['time_window'].max().date()}",
        'times': [str(t) for t in times]
    }
    
    return render_template('dashboard.html', stats=stats)

@app.route('/api/filter_data')
def filter_data():
    """Filter data based on parameters"""
    try:
        # Get parameters
        selected_time = request.args.get('time', predictions['time_window'].max())
        selected_time = pd.to_datetime(selected_time)
        
        risk_levels = request.args.getlist('risk_levels[]')
        if not risk_levels:
            risk_levels = ['Low', 'Medium', 'High']
        
        threshold = float(request.args.get('threshold', 5.0))
        min_pred = float(request.args.get('min_pred', 0.0))
        
        # Filter data
        filtered = predictions[
            (predictions['time_window'] == selected_time) &
            (predictions['risk_level'].isin(risk_levels)) &
            (predictions[pred_col] >= min_pred)
        ].copy()
        
        if len(filtered) == 0:
            return jsonify({'error': 'No data matches filters'})
        
        # Calculate metrics
        total_pred = float(filtered[pred_col].sum())
        total_actual = float(filtered['target'].sum())
        high_risk = int((filtered[pred_col] >= threshold).sum())
        avg_error = float(filtered[error_col].mean())
        accuracy = float((filtered[error_col] < 0.5).mean() * 100)
        
        # Risk distribution
        risk_counts = filtered['risk_level'].value_counts().to_dict()
        
        metrics = {
            'total_predicted': total_pred,
            'total_actual': total_actual,
            'high_risk_count': high_risk,
            'avg_error': avg_error,
            'accuracy': accuracy,
            'risk_distribution': risk_counts,
            'n_cells': len(filtered)
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/map_data')
def map_data():
    """Get data for map visualization"""
    try:
        selected_time = request.args.get('time', predictions['time_window'].max())
        selected_time = pd.to_datetime(selected_time)
        
        risk_levels = request.args.getlist('risk_levels[]')
        if not risk_levels:
            risk_levels = ['Low', 'Medium', 'High']
        
        threshold = float(request.args.get('threshold', 5.0))
        min_pred = float(request.args.get('min_pred', 0.0))
        
        # Filter data
        filtered = predictions[
            (predictions['time_window'] == selected_time) &
            (predictions['risk_level'].isin(risk_levels)) &
            (predictions[pred_col] >= min_pred)
        ].copy()
        
        # Sample for performance if too many
        if len(filtered) > 5000:
            filtered = filtered.sample(5000)
        
        # Create map
        fig = go.Figure()
        
        # Main scatter
        fig.add_trace(go.Scattermapbox(
            lat=filtered['cell_center_lat'],
            lon=filtered['cell_center_lon'],
            mode='markers',
            marker=dict(
                size=np.sqrt(filtered[pred_col]) * 5 + 3,
                color=filtered[pred_col],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score"),
                opacity=0.7
            ),
            text=[f"Cell: {row['cell_id']}<br>Predicted: {row[pred_col]:.2f}<br>Actual: {row['target']:.0f}<br>Risk: {row['risk_level']}" 
                  for _, row in filtered.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name='Predictions'
        ))
        
        # High risk alerts
        high_risk_data = filtered[filtered[pred_col] >= threshold]
        if len(high_risk_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=high_risk_data['cell_center_lat'],
                lon=high_risk_data['cell_center_lon'],
                mode='markers',
                marker=dict(size=15, color='red', opacity=0.3),
                text=[f"⚠️ HIGH RISK: {row[pred_col]:.2f}" for _, row in high_risk_data.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='High Risk'
            ))
        
        # Layout
        center_lat = filtered['cell_center_lat'].mean()
        center_lon = filtered['cell_center_lon'].mean()
        
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10
            ),
            height=700,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance_charts')
def performance_charts():
    """Get performance analysis charts"""
    try:
        selected_time = request.args.get('time', predictions['time_window'].max())
        selected_time = pd.to_datetime(selected_time)
        
        filtered = predictions[predictions['time_window'] == selected_time].copy()
        
        # Risk pie chart
        risk_counts = filtered['risk_level'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=['#28a745', '#ffc107', '#dc3545']),
            hole=0.4
        )])
        fig_pie.update_layout(title="Risk Distribution", height=400)
        
        # Actual vs Predicted scatter
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=filtered['target'],
            y=filtered[pred_col],
            mode='markers',
            marker=dict(size=6, color=filtered[error_col], colorscale='Reds', showscale=True)
        ))
        max_val = max(filtered['target'].max(), filtered[pred_col].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(dash='dash', color='blue'),
            name='Perfect'
        ))
        fig_scatter.update_layout(
            title="Actual vs Predicted",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            height=400
        )
        
        # Error histogram
        fig_hist = go.Figure(data=[go.Histogram(x=filtered[error_col], nbinsx=50)])
        fig_hist.update_layout(title="Error Distribution", height=350)
        
        return jsonify({
            'pie': json.loads(fig_pie.to_json()),
            'scatter': json.loads(fig_scatter.to_json()),
            'histogram': json.loads(fig_hist.to_json())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/temporal_trends')
def temporal_trends():
    """Get temporal trend charts"""
    try:
        # Aggregate by time
        time_agg = predictions.groupby('time_window').agg({
            pred_col: 'sum',
            'target': 'sum',
            error_col: 'mean'
        }).reset_index()
        
        # Time series
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Incidents Over Time", "Error Trend"))
        
        fig.add_trace(
            go.Scatter(x=time_agg['time_window'], y=time_agg[pred_col],
                      mode='lines', name='Predicted', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_agg['time_window'], y=time_agg['target'],
                      mode='lines', name='Actual', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_agg['time_window'], y=time_agg[error_col],
                      mode='lines+markers', name='Error', line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.update_layout(height=600)
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/hotspots')
def hotspots():
    """Get top hotspots"""
    try:
        selected_time = request.args.get('time', predictions['time_window'].max())
        selected_time = pd.to_datetime(selected_time)
        n = int(request.args.get('n', 10))
        
        filtered = predictions[predictions['time_window'] == selected_time].copy()
        
        top = filtered.nlargest(n, pred_col)[
            ['cell_id', pred_col, 'target', error_col, 'risk_level']
        ].copy()
        
        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(top)+1)), y=top[pred_col], name='Predicted'))
        fig.add_trace(go.Bar(x=list(range(1, len(top)+1)), y=top['target'], name='Actual'))
        fig.update_layout(title=f"Top {n} Hotspots", barmode='group', height=400)
        
        return jsonify({
            'data': top.to_dict('records'),
            'chart': json.loads(fig.to_json())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/feature_importance')
def get_feature_importance():
    """Get feature importance chart"""
    try:
        if feature_importance is None:
            return jsonify({'error': 'Feature importance not available'})
        
        n = int(request.args.get('n', 20))
        top = feature_importance.head(n)
        
        fig = go.Figure(data=[go.Bar(
            x=top['importance'],
            y=top['feature'],
            orientation='h',
            marker=dict(color=top['importance'], colorscale='Viridis')
        )])
        
        fig.update_layout(
            title=f"Top {n} Features",
            xaxis_title="Importance",
            height=max(400, n * 20),
            yaxis=dict(autorange="reversed")
        )
        
        return jsonify(json.loads(fig.to_json()))
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  RiskGrid Flask Dashboard")
    print("="*60)
    if predictions is not None:
        print(f"✓ Data loaded: {len(predictions):,} predictions")
        print(f"✓ Opening: http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ ERROR: No data found!")
        print("Run: python generate_predictions.py")
        print("="*60 + "\n")