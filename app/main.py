from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .models.matcher import AcademicMatcher
from .models.schemas import (
    RecommendationRequest, RecommendationResponse, 
    HealthResponse, ModelStatsResponse
)
from .config.settings import settings

# Crear instancia de FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Servicio de Machine Learning para recomendaciones académicas"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del matcher
matcher = AcademicMatcher()

@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando inicia el servicio - ENTRENA AUTOMÁTICAMENTE"""
    print("🚀 Iniciando servicio de ML...")
    try:
        result = matcher.train_model()
        print(f"✅ Modelo entrenado en startup: {result}")
    except Exception as e:
        print(f"⚠️ Error al entrenar modelo en startup: {e}")
        print("El modelo se puede entrenar manualmente con POST /retrain")

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Endpoint principal para obtener recomendaciones"""
    recommendations = matcher.get_recommendations(
        user_id=request.user_id,
        exclude_users=request.exclude_users,
        limit=request.limit
    )
    
    return RecommendationResponse(
        recommendations=recommendations,
        model_version=settings.API_VERSION,
        generated_at=datetime.now().isoformat()
    )

@app.post("/retrain")
async def retrain_model():
    """Endpoint para re-entrenar el modelo manualmente"""
    result = matcher.train_model()
    return {"message": "Modelo re-entrenado exitosamente", "details": result}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    health_data = matcher.is_healthy()
    return HealthResponse(
        status="healthy",
        model_trained=health_data["model_trained"],
        timestamp=datetime.now().isoformat(),
        users_loaded=health_data["users_loaded"]
    )

@app.get("/model/stats", response_model=ModelStatsResponse)
async def model_stats():
    """Estadísticas del modelo actual"""
    stats = matcher.get_model_stats()
    return ModelStatsResponse(
        total_users=stats["total_users"],
        feature_dimensions=stats["feature_dimensions"],
        k_neighbors=stats["k_neighbors"],
        last_trained=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    """Endpoint raíz"""
    health_data = matcher.is_healthy()
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "model_trained": health_data["model_trained"],
        "documentation": "/docs"
    }