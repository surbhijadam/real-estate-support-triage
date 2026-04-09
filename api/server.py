# api/server.py

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import sys, os, json
import warnings

# Suppress langchain_core Pydantic V1 deprecation warning for Python 3.14+
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import triage_pipeline
from database.db import init_db, get_db
from database.models import Ticket as TicketModel

Ticket = TicketModel

# Auto-create tables on startup
init_db()

app = FastAPI(
    title="Real Estate Support Triage Agent",
    description="AI-powered triage system for real estate communications",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ──

class IncomingMessage(BaseModel):
    message: str
    source: str = "web"  # email, whatsapp, web, etc.

class TriageResponse(BaseModel):
    id: int
    original_message: str
    urgency: str
    intent: str
    entities: dict
    draft_reply: str
    source: str
    created_at: str

class TicketSummary(BaseModel):
    id: int
    urgency: str
    intent: str
    original_message: str
    created_at: str

# ── Routes ──

@app.get("/")
def root():
    return {"status": "running", "agent": "Real Estate Support Triage Agent"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/chat")
def chat_ui():
    html_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "index.html"
    )
    return FileResponse(html_path, media_type="text/html")

@app.post("/triage", response_model=TriageResponse)
def triage(data: IncomingMessage, db: Session = Depends(get_db)):
    # Run the 3-agent pipeline
    result = triage_pipeline(data.message)

    # Persist ticket to DB
    ticket: Ticket = Ticket(
        original_message = result["original_message"],
        urgency          = result["urgency"],
        intent           = result["intent"],
        entities         = json.dumps(result["entities"]),
        draft_reply      = result["draft_reply"],
        source           = data.source,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return TriageResponse(
        id               = int(ticket.id), # type: ignore
        original_message = ticket.original_message, # type: ignore
        urgency          = ticket.urgency, # type: ignore
        intent           = ticket.intent, # type: ignore
        entities         = json.loads(ticket.entities), # type: ignore
        draft_reply      = ticket.draft_reply, # type: ignore
        source           = ticket.source, # type: ignore
        created_at       = ticket.created_at.isoformat(), # type: ignore
    )

@app.get("/tickets", response_model=List[TicketSummary])
def list_tickets(
    urgency: Optional[str] = None,
    intent:  Optional[str] = None,
    limit:   int = 50,
    db: Session = Depends(get_db)
):
    q = db.query(Ticket)
    if urgency: q = q.filter(Ticket.urgency == urgency.upper())
    if intent:  q = q.filter(Ticket.intent  == intent.lower())
    tickets = q.order_by(Ticket.created_at.desc()).limit(limit).all()
    return [
        TicketSummary(
            id               = t.id,
            urgency          = t.urgency,
            intent           = t.intent,
            original_message = t.original_message,
            created_at       = t.created_at.isoformat(),
        ) for t in tickets
    ]

@app.get("/tickets/{ticket_id}", response_model=TriageResponse)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TriageResponse(
        id               = t.id,
        original_message = t.original_message,
        urgency          = t.urgency,
        intent           = t.intent,
        entities         = json.loads(t.entities),
        draft_reply      = t.draft_reply,
        source           = t.source,
        created_at       = t.created_at.isoformat(),
    )

@app.delete("/tickets/{ticket_id}")
def delete_ticket(ticket_id: int, db: Session = Depends(get_db)):
    t = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Ticket not found")
    db.delete(t)
    db.commit()
    return {"deleted": ticket_id}