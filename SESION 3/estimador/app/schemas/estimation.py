# schemas/estimation.py
from pydantic import BaseModel, Field

class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión con el cliente"
    )

class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str