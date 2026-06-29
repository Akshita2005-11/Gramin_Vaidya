# Gramin Vaidya — AI Backend

## How to Setup:

1. Clone the repository:
git clone https://github.com/Akshita2005-11/Gramin_Vaidya.git

2. Create virtual environment:
python -m venv venv
venv\Scripts\activate

3. Install packages:
pip install -r requirements.txt

4. Create .env file and add your API key:
GROQ_API_KEY=your-groq-api-key-here
(Get free key from: https://console.groq.com)

5. Start the server:
python main.py

6. Test it:
http://localhost:8000/health

## API Endpoints:
- POST /ask — Send health query
- GET /health — Check server status

## Supported Languages:
Hindi, English, Bengali, Tamil, Telugu, Marathi, Gujarati,
Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese,
Bhojpuri, Rajasthani, Haryanvi, Hinglish and more!

## Emergency:
If serious symptoms detected, AI will say "Call 108 immediately!"