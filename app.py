from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lung-cancer-prediction-2025'

# Load trained models and preprocessors
try:
    best_model = joblib.load('lung_cancer_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models - {e}")
    best_model = None

# Load model comparison results if available
try:
    results_df = pd.read_csv('model_comparison_results.csv')
except:
    results_df = None

@app.route('/')
def index():
    """Home page with hero section"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/prediction')
def prediction():
    """Prediction form page"""
    return render_template('prediction.html')

@app.route('/visualizations')
def visualizations():
    """Visualization page showing model performance"""
    if results_df is not None:
        models_data = results_df.to_dict('records')
    else:
        models_data = []

    return render_template('visualizations.html', models_data=models_data)

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if best_model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500

        # Get form data
        data = request.get_json()

        # Create patient data dictionary
        patient_data = {
            'age': float(data['age']),
            'gender': data['gender'],
            'pack_years': float(data['pack_years']),
            'radon_exposure': data['radon_exposure'],
            'asbestos_exposure': data['asbestos_exposure'],
            'secondhand_smoke_exposure': data['secondhand_smoke_exposure'],
            'copd_diagnosis': data['copd_diagnosis'],
            'alcohol_consumption': data['alcohol_consumption'],
            'family_history': data['family_history']
        }

        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in patient_df.columns:
                patient_df[col] = encoder.transform(patient_df[col])

        # Scale features
        patient_scaled = scaler.transform(patient_df)

        # Make prediction
        prediction = best_model.predict(patient_scaled)[0]
        probability = best_model.predict_proba(patient_scaled)[0]

        # Decode prediction
        result = target_encoder.inverse_transform([prediction])[0]

        # Determine risk level
        cancer_prob = probability[1] * 100
        if cancer_prob < 30:
            risk_level = "Low Risk"
            risk_color = "#2ecc71"
        elif cancer_prob < 60:
            risk_level = "Medium Risk"
            risk_color = "#f39c12"
        else:
            risk_level = "High Risk"
            risk_color = "#e74c3c"

        # Generate prediction chart
        chart_data = generate_prediction_chart(probability, result)

        return jsonify({
            'success': True,
            'prediction': result,
            'probability': {
                'no_cancer': round(probability[0] * 100, 2),
                'has_cancer': round(probability[1] * 100, 2)
            },
            'risk_level': risk_level,
            'risk_color': risk_color,
            'chart_data': chart_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

def generate_prediction_chart(probability, result):
    """Generate base64 encoded chart for prediction result"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Probability Bar Chart
        categories = ['No Cancer', 'Has Cancer']
        probs = [probability[0] * 100, probability[1] * 100]
        colors = ['#2ecc71', '#e74c3c']

        bars = ax1.bar(categories, probs, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Gauge Chart (Pie chart style)
        cancer_prob = probability[1]
        remaining = 1 - cancer_prob

        if cancer_prob < 0.3:
            gauge_colors = ['#2ecc71', '#ecf0f1']
        elif cancer_prob < 0.6:
            gauge_colors = ['#f39c12', '#ecf0f1']
        else:
            gauge_colors = ['#e74c3c', '#ecf0f1']

        ax2.pie([cancer_prob, remaining], colors=gauge_colors, startangle=90,
               counterclock=False, wedgeprops=dict(width=0.3, edgecolor='black'))

        # Add center circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor='black', linewidth=2)
        ax2.add_artist(centre_circle)

        # Add percentage text in center
        ax2.text(0, 0, f'{cancer_prob*100:.1f}%', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax2.text(0, -0.15, 'Cancer Risk', 
                ha='center', va='center', fontsize=10)

        ax2.set_title('Risk Assessment Gauge', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

@app.route('/api/model-performance')
def model_performance():
    """API endpoint for model performance data"""
    if results_df is not None:
        return jsonify(results_df.to_dict('records'))
    return jsonify([])

@app.route('/static/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization images"""
    return send_from_directory('static/visualizations', filename)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 LUNG CANCER PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://127.0.0.1:5001")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
