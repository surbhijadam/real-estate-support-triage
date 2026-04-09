from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"

    id               = Column(Integer, primary_key=True, index=True)
    original_message = Column(Text, nullable=False)
    urgency          = Column(String(10))
    intent           = Column(String(20))
    entities         = Column(Text, nullable=False)
    draft_reply      = Column(Text)
    source           = Column(String(50), default="web")
    created_at       = Column(DateTime, default=datetime.utcnow)