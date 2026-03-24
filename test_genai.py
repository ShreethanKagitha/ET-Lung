import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_actual_api_key":
    print("❌ API Key not found or still set to default.")
    exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

try:
    response = model.generate_content("Say 'Gemini is connected successfully!' in short.")
    print("✅ Connection successful!")
    print(f"🤖 Gemini says: {response.text.strip()}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
