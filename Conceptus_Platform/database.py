"""
Conceptus Pro Database Models
User management and session recording
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./conceptus.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(200))
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(16), unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    question = Column(Text)
    mode = Column(String(20))
    status = Column(String(20), default="completed")
    formula_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Formula(Base):
    __tablename__ = "formulas"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(16), ForeignKey("analysis_sessions.session_id"))
    formula = Column(String(500))
    name = Column(String(200))
    agent_name = Column(String(50))
    mean_auc = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    Base.metadata.create_all(bind=engine)
    print("[Database] Tables created successfully")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_default_admin():
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin = User(
                username="admin",
                email="admin@conceptus.pro",
                hashed_password=pwd_context.hash("admin123"),
                role="admin"
            )
            db.add(admin)
            db.commit()
            print("[Database] Default admin created: admin / admin123")
    finally:
        db.close()
