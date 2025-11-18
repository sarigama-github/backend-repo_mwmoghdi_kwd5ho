"""
Database Schemas for the app

We define:
- Car: raw training/prediction inputs
- ModelMeta: track model versioning and metrics
"""
from pydantic import BaseModel, Field
from typing import Optional

class Car(BaseModel):
    brand: str = Field(..., description="Car brand/make, e.g., Maruti, Hyundai")
    model: str = Field(..., description="Car model name, e.g., Swift, i20")
    year: int = Field(..., ge=1990, le=2100, description="Year of manufacture")
    km_driven: int = Field(..., ge=0, description="Total kilometres driven")
    fuel: str = Field(..., description="Fuel type: Petrol/Diesel/CNG/Electric/LPG")
    seller_type: str = Field(..., description="Individual/Dealer/Trustmark Dealer")
    transmission: str = Field(..., description="Manual/Automatic")
    owner: str = Field(..., description="First Owner/Second Owner/Third Owner/Fourth & Above Owner/Test Drive Car")
    mileage: float = Field(..., ge=0, description="Mileage in kmpl (approx)")
    engine: int = Field(..., ge=100, le=6000, description="Engine CC")
    max_power: float = Field(..., ge=10, le=1000, description="Max power in bhp")
    seats: int = Field(..., ge=2, le=12, description="Number of seats")
    
class Modelmeta(BaseModel):
    version: str = Field(..., description="Model version string")
    algorithm: str = Field(..., description="Algorithm used, e.g., RandomForestRegressor")
    r2: Optional[float] = Field(None, description="R^2 on validation set")
    mae: Optional[float] = Field(None, description="MAE on validation set")
    created_by: Optional[str] = Field("system", description="Who trained the model")
