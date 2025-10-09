from typing import Optional
from pydantic import BaseModel

class HousingFeatures(BaseModel):
    # --- Features Relevantes (Obligatorias) ---
    RM: float
    LSTAT: float
    DIS: float
    AGE: float
    TAX: float
    B: float
    CRIM: float
    INDUS: float
    PTRATIO: float
    NOX: float
    # --- Features Menos Relevantes (Opcionales) ---
    RAD: Optional[int] = None
    CHAS: Optional[int] = None
    ZN: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "CRIM": 0.02731,
                "ZN": 0.0,
                "INDUS": 7.07,
                "CHAS": 0,
                "NOX": 0.469,
                "RM": 6.421,
                "AGE": 78.9,
                "DIS": 4.9671,
                "RAD": 2,
                "TAX": 242,
                "PTRATIO": 17.8,
                "B": 396.9,
                "LSTAT": 9.14
            }
        }