# agents/responder.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def generate_response(message: str, classification: dict, entities: dict) -> str:
    system_prompt = f"""
    You are a professional real estate support agent.
    Write a warm, helpful draft reply to this tenant message.
    
    Context:
    - Urgency: {classification['urgency']}
    - Intent: {classification['intent']}
    - Tenant ID: {entities.get('tenant_id', 'unknown')}
    - Unit: {entities.get('unit_number', 'unknown')}
    
    Rules:
    - HIGH urgency: promise action within 2 hours
    - MEDIUM urgency: promise action within 24 hours
    - LOW urgency: promise action within 3 business days
    - Always be empathetic and professional
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Tenant message: {message}")
    ])
    
    return str(response.content)