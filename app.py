from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash
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
import genai_service
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lung-cancer-prediction-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lungcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    # Relationship to PredictionHistory
    predictions = db.relationship('PredictionHistory', backref='author', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    risk_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    personalized_report = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

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

@app.route('/privacy')
def privacy():
    """Privacy Policy page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms of Service page"""
    return render_template('terms.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User Registration Route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists
        user_exists = User.query.filter_by(email=email).first() or User.query.filter_by(username=username).first()
        if user_exists:
            flash('Email or Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('register'))
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Login Route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            flash('Logged in successfully.', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User Logout Route"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User Dashboard - Prediction History"""
    history = PredictionHistory.query.filter_by(author=current_user).order_by(PredictionHistory.date_posted.desc()).all()
    return render_template('dashboard.html', history=history)

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

        # Generate personalized report
        personalized_report = genai_service.generate_personalized_report(patient_data, round(cancer_prob, 2), result)

        # Save to database if user is logged in
        if current_user.is_authenticated:
            new_pred = PredictionHistory(
                risk_probability=round(cancer_prob, 2),
                risk_level=risk_level,
                prediction_result=result,
                personalized_report=personalized_report,
                author=current_user
            )
            db.session.add(new_pred)
            db.session.commit()
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
            'personalized_report': personalized_report,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/preventive-guide', methods=['POST'])
def preventive_guide():
    try:
        data = request.get_json()
        patient_data = {
            'age': float(data.get('age', 0)),
            'gender': data.get('gender', 'Unknown'),
            'pack_years': float(data.get('pack_years', 0)),
            'radon_exposure': data.get('radon_exposure', 'Unknown'),
            'asbestos_exposure': data.get('asbestos_exposure', 'Unknown'),
            'secondhand_smoke_exposure': data.get('secondhand_smoke_exposure', 'Unknown'),
            'copd_diagnosis': data.get('copd_diagnosis', 'Unknown'),
            'alcohol_consumption': data.get('alcohol_consumption', 'Unknown'),
            'family_history': data.get('family_history', 'Unknown')
        }
        risk_level = data.get('risk_level', 'Unknown')
        
        guide_html = genai_service.generate_preventive_guide(patient_data, risk_level)
        return jsonify({'success': True, 'guide_html': guide_html})
    except Exception as e:
        print(f"Preventive Guide Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages for the GenAI conversational assistant"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        # Expecting string or format it if needed
        conversation_history = data.get('history', '')
        
        response = genai_service.conversational_chat(user_message, conversation_history)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/parse-voice', methods=['POST'])
def parse_voice():
    """Endpoint to take raw voice STT transcripts and parse them into structured ML data"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
            
        parsed_data = genai_service.extract_patient_data(transcript)
        return jsonify(parsed_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-scan', methods=['POST'])
def analyze_scan():
    """Endpoint for Vision API analysis of uploaded Chest X-Rays or CT Scans"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img_bytes = file.read()
        analysis = genai_service.analyze_medical_scan(img_bytes)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 LUNG CANCER PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("Access the application at: http://127.0.0.1:5001")
    print("="*60 + "\n")
    # Start the application with multi-threading enabled to prevent GenAI API blocking!
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
