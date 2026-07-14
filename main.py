from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from groq import Groq
import google.generativeai as genai
from PIL import Image
import io
import base64
import re
import os
import uvicorn

app = FastAPI(title="Gramin Vaidya API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"])

# ── Groq client for TEXT ──
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Gemini client for IMAGES ──
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Supported app languages -> used to validate default_language sent from the app
SUPPORTED_LANGUAGES = {
    "hindi": "Hindi", "english": "English", "marwari": "Marwari/Rajasthani",
    "bhojpuri": "Bhojpuri", "haryanvi": "Haryanvi", "chhattisgarhi": "Chhattisgarhi",
    "maithili": "Maithili", "tamil": "Tamil", "telugu": "Telugu", "bengali": "Bengali",
    "marathi": "Marathi", "gujarati": "Gujarati", "kannada": "Kannada",
    "malayalam": "Malayalam", "punjabi": "Punjabi", "urdu": "Urdu",
    "odia": "Odia", "assamese": "Assamese", "hinglish": "Hinglish"
}

# Hardcoded, simple disclaimers — not AI-generated, so wording stays simple and consistent every time
DISCLAIMERS = {
    "hindi": "*यह जवाब AI ने दिया है, सिर्फ शुरुआती जानकारी के लिए। सही जांच के लिए पास के डॉक्टर या क्लिनिक ज़रूर जाएं।*",
    "english": "*This answer is given by AI, only for early guidance. Please visit a nearby doctor or clinic for a proper check-up.*",
    "kannada": "*ಈ ಉತ್ತರವನ್ನು AI ನೀಡಿದೆ, ಇದು ಕೇವಲ ಆರಂಭಿಕ ಮಾಹಿತಿಗಾಗಿ. ಸರಿಯಾದ ಪರೀಕ್ಷೆಗಾಗಿ ಹತ್ತಿರದ ವೈದ್ಯರು ಅಥವಾ ಕ್ಲಿನಿಕ್‌ಗೆ ಭೇಟಿ ನೀಡಿ.*",
    "marathi": "*हे उत्तर AI ने दिले आहे, फक्त सुरुवातीच्या माहितीसाठी. योग्य तपासणीसाठी जवळच्या डॉक्टरकडे किंवा दवाखान्यात जरूर जा.*",
    "gujarati": "*આ જવાબ AI એ આપ્યો છે, ફક્ત શરૂઆતની જાણકારી માટે. યોગ્ય તપાસ માટે નજીકના ડોક્ટર અથવા ક્લિનિકની મુલાકાત જરૂર લો.*",
    "punjabi": "*ਇਹ ਜਵਾਬ AI ਨੇ ਦਿੱਤਾ ਹੈ, ਸਿਰਫ਼ ਸ਼ੁਰੂਆਤੀ ਜਾਣਕਾਰੀ ਲਈ। ਸਹੀ ਜਾਂਚ ਲਈ ਨੇੜੇ ਦੇ ਡਾਕਟਰ ਜਾਂ ਕਲੀਨਿਕ ਜ਼ਰੂਰ ਜਾਓ।*",
    "bengali": "*এই উত্তরটি AI দিয়েছে, শুধু প্রাথমিক তথ্যের জন্য। সঠিক পরীক্ষার জন্য কাছের ডাক্তার বা ক্লিনিকে যান।*",
    "tamil": "*இந்த பதில் AI ஆல் வழங்கப்பட்டது, ஆரம்ப தகவலுக்காக மட்டுமே. சரியான பரிசோதனைக்கு அருகிலுள்ள மருத்துவரை அல்லது கிளினிக்கை அணுகவும்.*",
    "telugu": "*ఈ సమాధానం AI ఇచ్చింది, ఇది కేవలం ప్రాథమిక సమాచారం కోసమే. సరైన పరీక్ష కోసం సమీపంలోని డాక్టర్ లేదా క్లినిక్‌ను సందర్శించండి.*",
    "hinglish": "*Yeh jawab AI ne diya hai, sirf shuruaati jaankari ke liye. Sahi jaanch ke liye pass ke doctor ya clinic zaroor jaayein.*",
}

def get_disclaimer(default_language: Optional[str]) -> str:
    key = (default_language or "hindi").strip().lower()
    return DISCLAIMERS.get(key, DISCLAIMERS["hindi"])

BASE_SYSTEM_PROMPT = """
You are Gramin Vaidya, a professional and experienced village health assistant in India.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONCISENESS RULE — MOST STRICTLY FOLLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keep the ENTIRE response short and to the point. Guidelines:
- Each section should be 1-3 short lines maximum, not long paragraphs.
- Limit to 2-3 items (causes, remedies, warning signs) instead of 3-4 — pick only the most relevant ones.
- Avoid repeating information across sections.
- Avoid flowery or overly warm language — be brief, clear, and respectful, not chatty.
- The overall response should be readable in under 30 seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULE — MOST STRICTLY FOLLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{language_instruction}
ALL section headers must also be translated into the reply language. Never mix two languages in one response. ALL section headers/headings must always be formatted in BOLD (using double asterisks like **heading**) — never italics, never plain unformatted text.

SIMPLE LANGUAGE RULE: When replying in Hindi or any Indian regional language, use simple, everyday spoken words that a rural villager with basic schooling would easily understand — the way people actually talk at home or with a local doctor, not formal/bookish/Sanskritized language. For example, prefer "जांच" over "आकलन", "इस्तेमाल" over "प्रयोग", "वजह" over "उद्देश्य", "लगाना" over "स्थानीय उपयोग करना". Avoid English-origin technical words where a common Hindi/regional word already exists and is widely understood (e.g. "खुजली" not "इचिंग"). Keep sentences short and direct.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPIC RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only answer health related questions. If the user asks anything non-health related, politely say (in the reply language) that you can only help with health-related concerns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEVERITY TIER RULE — MOST STRICTLY FOLLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every query (text or image), first internally classify the condition into ONE of these three tiers, then respond using ONLY the matching format below. Do not mix formats.

**TIER 1 — MINOR, HOME-CURABLE** (e.g. mild cold, minor cuts, mild indigestion, minor skin irritation, mild headache, common cold, mild cough)
Respond with:
1. UNDERSTANDING YOUR CONCERN
2. POSSIBLE CAUSES (2-3 reasons)
3. HOME REMEDIES (3-4 remedies using: Turmeric, Neem, Tulsi, Ginger, Ajwain, Garlic, Honey, Lemon, Mustard oil, Coconut oil, Fenugreek, Cinnamon, Amla, Giloy, Mint — with exact quantities and frequency)
4. WARNING SIGNS (signs that mean it's actually worse than it looks)
5. CLOSING LINE: End with one short generic line (translated to reply language) meaning: "If the pain or discomfort increases or does not improve, do not delay — visit a doctor." Do NOT describe specific conditions for this — keep it a short, universal closing sentence.
Do NOT mention medicines in this tier — home remedies only.

**TIER 2 — MODERATE, NEEDS MEDICINE BUT NOT URGENT** (e.g. moderate infections, persistent fever, moderate allergic reactions, moderate skin conditions, recurring issues)
Respond with:
1. UNDERSTANDING YOUR CONCERN
2. POSSIBLE CAUSES (2-3 reasons)
3. HOME REMEDIES (for symptom relief alongside medicine)
4. ALLERGY CHECK FIRST: Explain simply how to do a basic patch/allergy test before trying any new remedy or medicine (e.g. apply a small amount on inner wrist/elbow, wait 15-20 minutes, check for redness/itching/swelling before full use).
5. MEDICINE CATEGORY: Mention only the GENERAL CATEGORY of medicine that is typically used for this condition (e.g. "an antihistamine", "a paracetamol-based fever reducer", "an antifungal cream") — NEVER give an exact brand name, drug name, or dosage. Always add: "Please confirm the exact medicine and correct dosage for your age, weight, and health condition with a nearby pharmacist or doctor before taking anything."
6. WARNING SIGNS
7. CLOSING LINE: End with one short generic line (translated to reply language) meaning: "If the pain or discomfort increases or does not improve, do not delay — visit a doctor." Keep it short and universal, not a detailed description.

**TIER 3 — SERIOUS / CANNOT BE SELF-TREATED** (e.g. severe chest pain, difficulty breathing, high fever with confusion, heavy bleeding, suspected fractures, severe abdominal pain, stroke symptoms, seizures, poisoning, deep wounds, symptoms suggesting diabetes/BP crisis/kidney/liver issues, anything unclear or potentially serious)
Respond with:
1. UNDERSTANDING YOUR CONCERN (brief, calm, non-alarming tone)
2. WHAT THIS MIGHT BE (in simple words, without causing panic)
3. WHY YOU NEED A DOCTOR NOW: Clearly explain why home treatment is not safe here and what a doctor will need to check/do.
4. If life-threatening: "Call 108 immediately or go to the nearest hospital now!"
Do NOT give home remedies or medicine suggestions in this tier — refer to a doctor only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEASES COVERED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eyes, Ears, Nose, Mouth & Tongue, Skin, Internal (fever, cough, breathing, stomach, headache, joint pain, chronic conditions), Children's health, Women's health, Mental health.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROFESSIONAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Be professional, warm and respectful like a caring doctor
- NEVER give exact medicine brand names or exact dosages, under any tier
- Never shame anyone for their condition
- Emergency number India: 108

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY OUTPUT TAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
At the very end of your response, on its own new line, you MUST add exactly one of these two tags, written exactly like this in English (do not translate this tag, do not skip it):
[EMERGENCY:YES]  — only if this is a Tier 3 case requiring urgent/immediate medical attention right now
[EMERGENCY:NO]   — for Tier 1 and Tier 2 cases, or any case that is not urgent
"""

def build_language_instruction(default_language: Optional[str]) -> str:
    if default_language and default_language.strip().lower() in SUPPORTED_LANGUAGES:
        lang_name = SUPPORTED_LANGUAGES[default_language.strip().lower()]
        return (
            f"The app's selected default language is {lang_name}. "
            f"ALWAYS reply in {lang_name}, regardless of what language the user typed their message in. "
            f"Do NOT switch to match the user's input language. "
            f"If for any reason you cannot produce a reply in {lang_name}, fall back to Hindi, and if that is not possible, fall back to English."
        )
    return (
        "No default app language was provided. Detect the language the user wrote in and reply in that same language. "
        "If the language cannot be confidently detected, reply in Hindi. If Hindi is not appropriate, reply in English."
    )

class HealthQuery(BaseModel):
    text: str = ""
    image_base64: Optional[str] = None
    default_language: Optional[str] = None  # e.g. "hindi", "kannada", "english" — sent by the app
    conversation_history: Optional[List[Dict[str, str]]] = None
    # Each item: {"role": "user" or "assistant", "text": "..."}
    # App must build and send this list itself — backend does not store any memory.

class HealthResponse(BaseModel):
    response: str
    is_emergency: bool

EMERGENCY_TAG_PATTERN = re.compile(r"\[EMERGENCY:(YES|NO)\]", re.IGNORECASE)

def extract_emergency_and_clean(raw_text: str):
    match = EMERGENCY_TAG_PATTERN.search(raw_text)
    is_emergency = bool(match and match.group(1).upper() == "YES")
    cleaned_text = EMERGENCY_TAG_PATTERN.sub("", raw_text).strip()
    return cleaned_text, is_emergency

@app.post("/ask", response_model=HealthResponse)
async def ask_gramin_vaidya(query: HealthQuery):
    if not query.text.strip() and not query.image_base64:
        raise HTTPException(status_code=400, detail="Text or image required")

    try:
        language_instruction = build_language_instruction(query.default_language)
        system_prompt = BASE_SYSTEM_PROMPT.format(language_instruction=language_instruction)

        # Build a simple text summary of prior turns, if the app sent any
        history_text = ""
        if query.conversation_history:
            lines = []
            for turn in query.conversation_history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                label = "User" if role == "user" else "Gramin Vaidya"
                lines.append(f"{label}: {text}")
            history_text = "\n".join(lines)

        if query.image_base64:
            image_bytes = base64.b64decode(query.image_base64)
            image = Image.open(io.BytesIO(image_bytes))

            gemini_prompt = system_prompt
            if history_text:
                gemini_prompt += f"\n\nHere is the earlier conversation with this user, for context:\n{history_text}\n\nUse this context if the new message refers back to it (e.g. follow-up questions like 'what medicine did you mean'). If the new message is about a completely different/new health concern, ignore the old context and treat it fresh."
            gemini_prompt += "\n\nThe user has sent a photo of a health concern (e.g. skin issue, wound, rash, eye problem)."
            if query.text.strip():
                gemini_prompt += f" They also wrote this message: {query.text}"
            gemini_prompt += "\n\nLook at the image carefully, classify the severity tier, and respond following the exact tier format above. Remember the mandatory [EMERGENCY:YES] or [EMERGENCY:NO] tag at the end."

            gemini_response = gemini_model.generate_content([gemini_prompt, image])
            raw_text = gemini_response.text

        else:
            groq_messages = [{"role": "system", "content": system_prompt}]

            if history_text:
                groq_messages.append({
                    "role": "system",
                    "content": f"Here is the earlier conversation with this user, for context:\n{history_text}\n\nUse this context if the new message refers back to it. If the new message is about a completely different/new health concern, ignore the old context and treat it fresh."
                })

            groq_messages.append({"role": "user", "content": query.text})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                temperature=0.3,
                max_tokens=1500
            )
            raw_text = response.choices[0].message.content

        response_text, is_emergency = extract_emergency_and_clean(raw_text)
        disclaimer = get_disclaimer(query.default_language)
        response_text = f"{disclaimer}\n\n{response_text}"

        return HealthResponse(response=response_text, is_emergency=is_emergency)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Gramin Vaidya API v4.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)