# main.py

import time
from agents.classifier import classify_message
from agents.ner_extractor import extract_entities
from agents.responder import generate_response

def triage_pipeline(incoming_message: str) -> dict:
    print("\n" + "="*50)
    print(f"Incoming message: {incoming_message}")
    print("="*50)

    print("\nStep 1: Classifying message...")
    classification = classify_message(incoming_message)
    print(f"   Urgency : {classification['urgency']}")
    print(f"   Intent  : {classification['intent']}")

    print("\nStep 2: Extracting entities...")
    entities = extract_entities(incoming_message)
    print(f"   Entities: {entities}")

    print("\nStep 3: Generating draft response...")
    draft_reply = generate_response(incoming_message, classification, entities)
    print(f"\n--- DRAFT REPLY ---\n{draft_reply}\n-------------------\n")

    return {
        "original_message": incoming_message,
        "urgency": classification["urgency"],
        "intent": classification["intent"],
        "entities": entities,
        "draft_reply": draft_reply
    }

if __name__ == "__main__":
    test_messages = [
        "Hi, I'm tenant #T-4421 in Unit 3B. My heater stopped working!",
        "Water is flooding my apartment RIGHT NOW, please help!",
        "What are your office hours on weekends?"
    ]

    for i, msg in enumerate(test_messages):
        result = triage_pipeline(msg)

        # Wait 30 seconds between messages to respect free tier rate limit
        if i < len(test_messages) - 1:
            print("Waiting 30 seconds before next message (free tier rate limit)...")
            time.sleep(30)