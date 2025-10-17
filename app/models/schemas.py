from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class UserProfile(BaseModel):
    user_id: str
    skills: List[str]
    objectives: List[str]
    semester: int
    age: int
    location: Dict[str, float]  # {"lat": -12.0464, "lng": -77.0428}
    time_availability: str
    commitment_level: str

class RecommendationRequest(BaseModel):
    user_id: str
    exclude_users: Optional[List[str]] = []  # Usuarios ya swipeados
    limit: Optional[int] = 10

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]  # lista de recomendaciones
    total_filtered: int                    # <-- agregamos este campo
    model_version: str
    generated_at: str
    
class TrainingResult(BaseModel):
    status: str
    users_processed: int
    features_shape: List[int]
    k_neighbors: int

class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    timestamp: str
    users_loaded: int

class ModelStatsResponse(BaseModel):
    total_users: int
    feature_dimensions: int
    k_neighbors: int
    last_trained: str