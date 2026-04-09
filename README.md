# Real Estate Support Triage System

An AI-powered support ticket triage system for real estate queries. 
It classifies incoming tickets, extracts key entities, and auto-responds 
using intelligent agents.

## Features
- 🏠 Classifies real estate support tickets automatically
- 🔍 Extracts named entities (location, price, property type, etc.)
- 💬 Auto-generates responses based on ticket type
- 🗄️ Stores and manages tickets in a local database

## Project Structure
├── agents/
│   ├── classifier.py       # Ticket classification logic
│   ├── ner_extractor.py    # Named entity recognition
│   └── responder.py        # Auto-response generation
├── api/
│   └── server.py           # API server
├── database/
│   ├── db.py               # Database connection
│   └── models.py           # Database models
├── tests/
│   └── test_agents.py      # Unit tests
├── main.py                 # Entry point
├── index.html              # Frontend
└── requirements.txt        # Dependencies

## Setup & Installation

1. **Clone the repository**
```bash
   git clone https://github.com/surbhijadam/real-estate-support-triage.git
   cd real-estate-support-triage
```

2. **Create and activate virtual environment**
```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
   cp .env.example .env
   # Edit .env with your actual values
```

5. **Run the app**
```bash
   python main.py
```

## Running Tests
```bash
pytest tests/
```