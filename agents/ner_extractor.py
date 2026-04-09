# agents/ner_extractor.py

import os, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()

def extract_entities(message: str) -> dict:
    system_prompt = """
    Extract real estate entities from this message.
    Respond ONLY with raw JSON, no markdown:
    {
      "tenant_id": "...",
      "unit_number": "...",
      "lease_date": "...",
      "property_address": "...",
      "person_name": "...",
      "dates": "...",
      "issue_type": "..."
    }
    Use null for any field not found.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=message)
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)
    return json.loads(clean_json(content))