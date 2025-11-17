from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class UserProfile(BaseModel):
    user_id: str
    skills: List[str]
    objectives: List[str]
    semester: int
    age: int
    location: Dict[str, float] 
    time_availability: str
    commitment_level: str

class PaginationMetadata(BaseModel):
    """Metadata de paginación"""
    page: int = Field(..., ge=1, description="Número de página actual (1-indexed)")
    limit: int = Field(..., ge=1, le=100, description="Resultados por página")
    total: int = Field(..., ge=0, description="Total de resultados disponibles")
    total_pages: int = Field(..., ge=0, description="Total de páginas")
    has_next: bool = Field(..., description="Existe página siguiente")
    has_prev: bool = Field(..., description="Existe página anterior")
    showing: int = Field(..., ge=0, description="Cantidad de resultados en esta página")

class RecommendationRequest(BaseModel):
    user_id: str
    exclude_users: Optional[List[str]] = Field(default_factory=list, description="Usuarios ya swipeados")
    limit: Optional[int] = Field(default=10, ge=1, le=50, description="Resultados por página")
    page: Optional[int] = Field(default=1, ge=1, description="Número de página")
    use_cache: Optional[bool] = Field(default=True, description="Usar cache de recomendaciones")

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    pagination: PaginationMetadata
    compatibility_metrics: Dict[str, Any]
    model_version: str
    generated_at: str
    cache_used: bool = Field(default=False, description="Si se usó cache")
    
class TrainingResult(BaseModel):
    status: str
    users_processed: int
    features_shape: List[int]
    k_neighbors: int
    cache_cleared: bool = Field(default=True, description="Cache limpiado tras entrenamiento")

class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    timestamp: str
    users_loaded: int
    cache_entries: int = Field(default=0, description="Entradas en cache")

class ModelStatsResponse(BaseModel):
    total_users: int
    feature_dimensions: int
    k_neighbors: int
    last_trained: str
    cache_size: int = Field(default=0, description="Tamaño del cache")

class CacheClearRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Usuario específico (None = limpiar todo)")

class CacheClearResponse(BaseModel):
    status: str
    message: str
    cleared_entries: int
