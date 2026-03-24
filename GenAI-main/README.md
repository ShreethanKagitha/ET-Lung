# 🫁 Lung Cancer Prediction System

An AI-powered web application for lung cancer risk assessment using machine learning algorithms. This system analyzes patient health and lifestyle factors to provide instant risk predictions with visual analytics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 🤖 AI-Powered Analysis
- **Multiple ML Algorithms**: Comparison of 7 different machine learning models
- **Best Model Selection**: Automatic selection of the highest-performing algorithm
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Visual Analytics**: Interactive charts and risk assessment gauges

### 🏥 Medical-Grade Accuracy
- **Comprehensive Dataset**: Trained on 50,000 patient records
- **Clinical Parameters**: Age, gender, smoking history, environmental exposures
- **Risk Factors**: COPD, family history, alcohol consumption, environmental toxins
- **Validated Results**: Cross-validation and performance metrics

### 🌐 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Forms**: User-friendly prediction interface
- **Real-time Visualization**: Dynamic charts and risk gauges
- **Professional UI**: Modern, medical-grade interface design

## 📊 Dataset Information

The system uses a comprehensive lung cancer dataset with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Patient age (18-100 years) |
| `gender` | Categorical | Male/Female |
| `pack_years` | Numeric | Smoking intensity (pack-years) |
| `radon_exposure` | Categorical | Low/Medium/High |
| `asbestos_exposure` | Categorical | Yes/No |
| `secondhand_smoke_exposure` | Categorical | Yes/No |
| `copd_diagnosis` | Categorical | Yes/No |
| `alcohol_consumption` | Categorical | None/Moderate/Heavy |
| `family_history` | Categorical | Yes/No |
| `lung_cancer` | Target | Yes/No (prediction target) |

**Dataset Statistics:**
- **Total Records**: 50,000 patients
- **Features**: 9 input features + 1 target variable
- **Missing Values**: None (complete dataset)
- **Class Distribution**: 68.7% positive cases, 31.3% negative cases

## 🧠 Machine Learning Models

The system compares multiple algorithms to select the best performer:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble method with feature importance
3. **Support Vector Machine (SVM)** - Non-linear classification
4. **Decision Tree** - Interpretable tree-based model
5. **XGBoost** - Gradient boosting algorithm
6. **K-Nearest Neighbors (KNN)** - Instance-based learning
7. **Naive Bayes** - Probabilistic classifier

### Model Performance Metrics
- **Accuracy Score**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix**
- **Cross-Validation Results**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd lung-cancer-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the application**
Open your browser and navigate to: `http://localhost:5000`

## 📋 Requirements

```
Flask==3.0.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.2
Werkzeug==3.0.0
```

## 🏗️ Project Structure

```
lung-cancer-prediction/
├── app.py                              # Main Flask application
├── lung_cancer_prediction_complete.ipynb  # ML development notebook
├── lung_cancer_dataset.csv             # Training dataset
├── requirements.txt                     # Python dependencies
├── README.md                           # Project documentation
│
├── models/                             # Trained models
│   ├── lung_cancer_best_model.pkl     # Best performing model
│   ├── scaler.pkl                      # Feature scaler
│   ├── label_encoders.pkl              # Categorical encoders
│   └── target_encoder.pkl              # Target encoder
│
├── static/                             # Static assets
│   ├── css/
│   │   └── style.css                   # Application styles
│   └── js/
│       └── main.js                     # JavaScript functionality
│
├── templates/                          # HTML templates
│   ├── base.html                       # Base template
│   ├── index.html                      # Home page
│   ├── prediction.html                 # Prediction form
│   ├── visualizations.html             # Model performance
│   ├── about.html                      # About page
│   └── contact.html                    # Contact page
│
└── visualizations/                     # Generated charts
    ├── confusion_matrices.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── model_comparison_visualization.png
    ├── roc_curves.png
    └── target_distribution.png
```

## 🎯 Usage Guide

### Making Predictions

1. **Navigate to Prediction Page**
   - Click "Start Prediction" on the home page
   - Or go directly to `/prediction`

2. **Fill Patient Information**
   - **Age**: Enter patient age (18-100)
   - **Gender**: Select Male/Female
   - **Pack Years**: Smoking intensity (0-100)
   - **Environmental Exposures**: Radon, Asbestos, Secondhand smoke
   - **Medical History**: COPD diagnosis, Family history
   - **Lifestyle**: Alcohol consumption level

3. **Get Results**
   - Click "Predict Risk" button
   - View probability scores and risk level
   - Analyze visual charts and gauges

### Understanding Results

- **Risk Level**: Low (<30%), Medium (30-60%), High (>60%)
- **Probability Scores**: Percentage chance for each outcome
- **Visual Charts**: Bar charts and risk assessment gauges
- **Color Coding**: Green (Low), Orange (Medium), Red (High)

## 📈 Model Development

The machine learning pipeline includes:

### 1. Data Preprocessing
- **Missing Value Handling**: Complete dataset validation
- **Categorical Encoding**: Label encoding for categorical features
- **Feature Scaling**: StandardScaler for numerical features
- **Train-Test Split**: 80-20 split with stratification

### 2. Model Training
- **Algorithm Comparison**: 7 different ML algorithms
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: GridSearchCV for optimization
- **Performance Metrics**: Comprehensive evaluation

### 3. Model Selection
- **Best Model**: Automatically selected based on performance
- **Model Persistence**: Saved using joblib for deployment
- **Preprocessing Pipeline**: Scalers and encoders saved separately

### 4. Visualization
- **Feature Importance**: Understanding key predictors
- **Confusion Matrices**: Classification performance
- **ROC Curves**: Model discrimination ability
- **Correlation Analysis**: Feature relationships

## 🔧 API Endpoints

### Web Routes
- `GET /` - Home page
- `GET /prediction` - Prediction form
- `GET /visualizations` - Model performance charts
- `GET /about` - About page
- `GET /contact` - Contact information

### API Routes
- `POST /predict` - Make prediction (JSON)
- `GET /api/model-performance` - Get model metrics

### Prediction API Example
```python
import requests

data = {
    "age": 65,
    "gender": "Male",
    "pack_years": 45.5,
    "radon_exposure": "High",
    "asbestos_exposure": "No",
    "secondhand_smoke_exposure": "Yes",
    "copd_diagnosis": "Yes",
    "alcohol_consumption": "Moderate",
    "family_history": "No"
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
```

## 🎨 Customization

### Styling
- Modify `static/css/style.css` for custom styling
- Update templates in `templates/` directory
- Add custom JavaScript in `static/js/main.js`

### Model Updates
- Retrain models using the Jupyter notebook
- Replace model files in the root directory
- Update feature preprocessing if needed

## 🔒 Security Features

- **Input Validation**: Server-side validation for all inputs
- **Error Handling**: Graceful error management
- **Secure Headers**: Flask security configurations
- **Data Privacy**: No persistent storage of patient data

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Warning: Could not load models
   ```
   - Ensure all .pkl files are present
   - Check file permissions
   - Verify joblib version compatibility

2. **Port Already in Use**
   ```
   Address already in use
   ```
   - Change port in app.py: `app.run(port=5001)`
   - Or kill existing process

3. **Missing Dependencies**
   ```
   ModuleNotFoundError
   ```
   - Run: `pip install -r requirements.txt`
   - Check Python version compatibility

## 📊 Performance Metrics

The system achieves high performance across multiple metrics:

- **Accuracy**: >90% on test dataset
- **Precision**: High positive prediction accuracy
- **Recall**: Effective detection of positive cases
- **F1-Score**: Balanced precision-recall performance
- **ROC-AUC**: Strong discrimination ability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important Medical Disclaimer:**

This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding medical conditions.

The predictions provided by this system are based on statistical models and should be interpreted by qualified medical professionals. This tool is not intended for clinical diagnosis or treatment decisions.

## 📞 Support

For support, questions, or suggestions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

## 🙏 Acknowledgments

- Dataset providers and medical research community
- Open-source machine learning libraries
- Flask web framework developers
- Healthcare professionals for domain expertise

---

**Made with ❤️ for advancing healthcare through AI**