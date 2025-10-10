import os
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_time = Column(DateTime, default=datetime.datetime.utcnow)
    prediction_value = Column(Float)

    # Inputs del modelo
    crim = Column(Float, name="CRIM")
    zn = Column(Float, name="ZN", nullable=True)
    indus = Column(Float, name="INDUS")
    chas = Column(Integer, name="CHAS", nullable=True)
    nox = Column(Float, name="NOX")
    rm = Column(Float, name="RM")
    age = Column(Float, name="AGE")
    dis = Column(Float, name="DIS")
    rad = Column(Integer, name="RAD", nullable=True)
    tax = Column(Float, name="TAX")
    ptratio = Column(Float, name="PTRATIO")
    b = Column(Float, name="B")
    lstat = Column(Float, name="LSTAT")


def init_db():
    Base.metadata.create_all(bind=engine)