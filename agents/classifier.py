# agents/classifier.py

import os, json
from dotenv import load_dotenv
# import spacy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
# nlp = spacy.load("en_core_web_sm")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def clean_json(text: str) -> str:
    """Strip markdown code fences if Gemini wraps response in them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # remove first line (```json)
        text = text.rsplit("```", 1)[0]  # remove closing ```
    return text.strip()

def classify_message(message: str) -> dict:
    system_prompt = """
    You are a real estate support triage agent.
    For every incoming message, respond with ONLY a JSON object like:
    {"urgency": "HIGH/MEDIUM/LOW", "intent": "maintenance/inquiry/complaint/payment/viewing"}
    
    Urgency rules:
    - HIGH = safety, flooding, fire, no water/heat, lockout
    - MEDIUM = broken appliance, noise, billing issue
    - LOW = general question, scheduling, feedback
    
    Respond with raw JSON only. No markdown, no explanation.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ])
    
    return json.loads(clean_json(str(response.content)))