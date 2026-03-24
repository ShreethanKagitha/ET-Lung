import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API with the key from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY is not set. GenAI features will be mocked.")

def _get_model():
    """Helper to return the model if API key is set, otherwise None."""
    if not api_key:
        return None
    # Use gemini-1.5-flash for fast text generation
    try:
        return genai.GenerativeModel('gemini-1.5-flash')
    except AttributeError:
        # Fallback if the library version is older or different
        return genai.GenerativeModel('gemini-pro')

def extract_patient_data(transcript):
    """
    Uses Gemini to extract raw conversational text into structured ML form data.
    """
    model = _get_model()
    if not model:
        return {}
        
    prompt = f"""
    You are an AI Clinical Data Extraction Assistant. 
    Read the patient's conversational transcript and extract their profile into a strict JSON payload matching exactly these keys and format rules:
    - "age": integer
    - "gender": "Male" or "Female"
    - "pack_years": float (Calculated. e.g. "1 pack a day for 30 years" = 30.0. "Non-smoker" = 0.0. "I smoked for 15 years" = default to 1 pack/day = 15.0)
    - "radon_exposure": "Low", "Medium", or "High"
    - "asbestos_exposure": "Yes" or "No"
    - "secondhand_smoke_exposure": "Yes" or "No"
    - "copd_diagnosis": "Yes" or "No"
    - "alcohol_consumption": "None", "Moderate", or "Heavy"
    - "family_history": "Yes" or "No"
    
    CRITICAL: If the patient skips information, use clinically safe defaults so the ML model can run. 
    Defaults: radon="Low", asbestos="No", secondhand_smoke="No", copd="No", alcohol="None", family_history="No".

    Patient Transcript: "{transcript}"
    
    Return ONLY a raw JSON object. Do not include Markdown like ```json.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"Extraction Error: {e}")
        return {}

def generate_personalized_report(patient_data, risk_probability, prediction_result):
    """
    Generates a personalized, empathetic medical report based on the ML prediction,
    acting as a RAG system grounded in WHO and CDC screening guidelines.
    """
    model = _get_model()
    
    # Format patient data nicely
    patient_summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in patient_data.items()])
    
    prompt = f"""
You are an advanced medical AI assistant specializing in lung cancer risk assessment. 
Your primary goal is to act as a supportive, empathetic, and highly accurate preventive health advisor.

[KNOWLEDGE BASE: MEDICAL GUIDELINES CATALOG]
Source 1: "CDC Lung Cancer Screening Guidelines (2024)" | URL: https://www.cdc.gov/cancer/lung/basic_info/screening.htm
Fact: The CDC recommends yearly lung cancer screening with low-dose computed tomography (LDCT) for people who have a 20 pack-year or more smoking history, AND smoke now or have quit within the past 15 years, AND are between 50 and 80 years old.

Source 2: "EPA Radon & Lung Cancer Guide" | URL: https://www.epa.gov/radon/health-risk-radon
Fact: Radon exposure is the second leading cause of lung cancer. The EPA recommends testing all homes for radon.

Source 3: "WHO Asbestos Fact Sheet" | URL: https://www.who.int/news-room/fact-sheets/detail/asbestos-elimination-of-asbestos-related-diseases
Fact: Asbestos exposure combined with smoking exponentially increases the risk of lung cancer.

[PATIENT PROFILE]
{patient_summary}

[MACHINE LEARNING PREDICTION RESULTS]
- Classification: {prediction_result}
- Statistical Probability Flag: {risk_probability}% Risk Factor

[OUTPUT INSTRUCTIONS]
Based strictly on the KNOWLEDGE BASE and the PATIENT PROFILE, generate a structured, highly personalized medical report. 
Do not hallucinate. Do not provide diagnostic certainty.

Structure your response EXACTLY with these 4 sections in raw HTML (do not use markdown backticks):

<h3>1. Personalized Medical Risk Explanation</h3>
<p>(Explain the {risk_probability}% risk in empathetic, simple terms. Acknowledge their specific inputs like age or COPD status and how it factored into this statistical prediction. Do not diagnose.)</p>

<h3>2. Preventive Health Recommendations</h3>
<ul>
  <li>(Provide 2-3 specific lifestyle changes, like smoking cessation programs or home radon testing, based purely on their profile.)</li>
</ul>

<h3>3. Screening Guidance & Citations</h3>
<p>(Using the provided KNOWLEDGE BASE, state whether they currently meet the criteria for a Low-Dose CT (LDCT) scan. Explain why or why not based on their age and pack-years.)</p>
<div style="margin-top: 15px; padding: 12px; background-color: #f8fcf8; border-left: 4px solid #2ecc71; font-size: 0.9em; border-radius: 4px;">
  <strong style="color: #27ae60;"><i class="fas fa-shield-check"></i> Verified Medical Sources:</strong>
  <ul style="margin-top: 8px; margin-bottom: 0; padding-left: 20px;">
    <li>(You MUST explicitly cite the exact source name, year, and URL from the KNOWLEDGE BASE used to make your recommendations above. E.g., "According to the <a href='...' target='_blank'>CDC Lung Cancer Screening Guidelines (2024)</a>...")</li>
  </ul>
</div>

<h3>4. Important AI Disclaimer</h3>
<p style="font-size: 0.85em; color: #7f8c8d;"><em>(Include a clear disclaimer that this is a GenAI-augmented predictive tool, not a clinical diagnosis, and highly recommend they share this printed report with their primary care physician.)</em></p>
"""
    
    if not model:
        return f"""
        <div class="alert alert-warning">
            <strong>Note:</strong> GenAI API key not configured. This is a mocked report.
        </div>
        <p>Based on your profile, the model calculated a cancer probability of <strong>{risk_probability}%</strong>.</p>
        <p>Your classification is: <strong>{prediction_result}</strong>.</p>
        <p>This is a statistical prediction. Please consult a doctor for a proper medical evaluation.</p>
        """
        
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Clean up in case the model returns markdown code blocks
        if "```html" in text:
            text = text.replace("```html", "").replace("```", "")
        return text.strip()
    except Exception as e:
        print(f"GenAI Error: {e}")
        return "<p>Error generating personalized report. Please consult your physician.</p>"

def conversational_chat(user_message, conversation_history=""):
    """
    Generates a response for the conversational health assistant.
    """
    model = _get_model()
    if not model:
        return "I am currently running in offline mode. Please set the GEMINI_API_KEY environment variable to enable GenAI chat."
        
    prompt = f"""
You are a helpful, empathetic Medical AI assistant specializing in Lung Health.
You guide patients through their questions regarding lung cancer, prevention, and screening.
Always be supportive. Never diagnose. Remind them you are an AI.

Conversation History:
{conversation_history}

Patient: {user_message}
AI Assistant:
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"GenAI Error: {e}")
        return "I'm sorry, I'm having trouble retrieving a response at the moment."
