from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional
import asyncio
from threading import Lock

import pandas as pd

from .models.matcher import AcademicMatcher
from .models.schemas import (
    CacheClearRequest, CacheClearResponse, RecommendationRequest, RecommendationResponse, 
    HealthResponse, ModelStatsResponse, PaginationMetadata
)
from .config.settings import settings
from .utils.database import DatabaseManager  # 👈 IMPORTAR DatabaseManager
# Si no funciona, prueba una de estas alternativas:
# from app.config.database import DatabaseManager
# from config.database import DatabaseManager

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Servicio de Machine Learning para recomendaciones académicas"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = AcademicMatcher()
needs_retraining = False
is_retraining = False
retrain_lock = Lock()

@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando inicia el servicio"""
    print("🚀 Iniciando servicio de ML...")
    try:
        result = matcher.train_model()
        print(f"✅ Modelo entrenado en startup: {result}")
    except Exception as e:
        print(f"⚠️ Error al entrenar modelo en startup: {e}")

def retrain_in_background():
    """Re-entrena el modelo en segundo plano"""
    global needs_retraining, is_retraining
    
    with retrain_lock:
        if is_retraining:
            print("⚠️ Re-entrenamiento ya en proceso, saltando...")
            return
        is_retraining = True
    
    try:
        print("🔄 Re-entrenando modelo en segundo plano...")
        result = matcher.train_model()
        needs_retraining = False
        print(f"✅ Modelo re-entrenado: {result['users_processed']} usuarios")
    except Exception as e:
        print(f"❌ Error re-entrenando: {e}")
        import traceback
        traceback.print_exc()
        needs_retraining = True
    finally:
        is_retraining = False

@app.post("/webhook/user-updated")
async def user_updated_webhook(
    x_api_key: Optional[str] = Header(None)
):
    """
    🔔 WEBHOOK llamado desde NestJS cuando se actualiza un usuario
    RE-ENTRENA DE FORMA SÍNCRONA para garantizar datos actualizados
    
    Headers opcionales:
    - x-api-key: Token de autenticación
    """
    global needs_retraining
    
    if settings.WEBHOOK_API_KEY and x_api_key != settings.WEBHOOK_API_KEY:
        return {"error": "Unauthorized"}, 401
    
    print(f"📥 Webhook recibido - RE-ENTRENANDO SÍNCRONAMENTE")
    
    # 🔥 SIEMPRE re-entrenar de forma síncrona
    try:
        print("⏳ Iniciando re-entrenamiento síncrono...")
        result = matcher.train_model()
        needs_retraining = False
        print(f"✅ Modelo re-entrenado: {result['users_processed']} usuarios")
        
        return {
            "message": "Model retrained successfully",
            "status": "completed",
            "users_processed": result['users_processed'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Error re-entrenando: {e}")
        import traceback
        traceback.print_exc()
        needs_retraining = True
        
        return {
            "message": "Model retraining failed",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks
):
    """
    Endpoint principal con paginación
    
    - **user_id**: ID del usuario solicitante
    - **exclude_users**: Lista de usuarios ya swipeados (opcional)
    - **limit**: Resultados por página (1-50, default: 10)
    - **page**: Número de página (1-indexed, default: 1)
    - **use_cache**: Usar cache de recomendaciones (default: true)
    """
    global needs_retraining, is_retraining
    
    print(f"📥 Request de recomendaciones:")
    print(f"   Usuario: {request.user_id}")
    print(f"   Página: {request.page}, Límite: {request.limit}")
    print(f"   Excluidos: {len(request.exclude_users)}")
    print(f"   needs_retraining={needs_retraining}, is_retraining={is_retraining}")
    
    # Si hay cambios pendientes, ESPERAR a que termine el re-entrenamiento
    if needs_retraining or is_retraining:
        print("⏳ Esperando a que termine el re-entrenamiento actual...")
        
        max_wait = 30
        waited = 0
        while (needs_retraining or is_retraining) and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5
        
        if waited >= max_wait:
            print("⚠️ Timeout esperando re-entrenamiento, usando modelo actual")
        else:
            print(f"✅ Re-entrenamiento completado después de {waited}s")
    
    # Si aún hay flag de re-entrenamiento, forzar uno síncrono
    if needs_retraining:
        print("🔄 Forzando re-entrenamiento síncrono antes de recomendar...")
        try:
            matcher.train_model()
            needs_retraining = False
            print("✅ Modelo actualizado antes de recomendar")
        except Exception as e:
            print(f"❌ Error re-entrenando: {e}")
    
    result = matcher.get_recommendations(
        user_id=request.user_id,
        exclude_users=request.exclude_users,
        limit=request.limit,
        page=request.page,
        use_cache=request.use_cache
    )
    
    return RecommendationResponse(
        recommendations=result["recommendations"],
        pagination=PaginationMetadata(**result["pagination"]),
        compatibility_metrics=result["compatibility_metrics"],
        model_version=settings.API_VERSION,
        generated_at=datetime.now().isoformat(),
        cache_used=result.get("cache_used", False)
    )

@app.post("/cache/clear", response_model=CacheClearResponse)
async def clear_cache(request: CacheClearRequest):
    """
    Limpia el cache de recomendaciones
    
    - Sin user_id: limpia todo el cache
    - Con user_id: limpia solo cache de ese usuario
    """
    try:
        cache_size_before = len(matcher._recommendation_cache)
        
        matcher.clear_cache(request.user_id)
        
        cache_size_after = len(matcher._recommendation_cache)
        cleared = cache_size_before - cache_size_after
        
        message = f"Cache limpiado para {request.user_id}" if request.user_id else "Cache completo limpiado"
        
        return CacheClearResponse(
            status="success",
            message=message,
            cleared_entries=cleared
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error limpiando cache: {str(e)}")
    
@app.post("/retrain")
async def retrain_model():
    """Re-entrena el modelo manualmente"""
    global needs_retraining
    result = matcher.train_model()
    needs_retraining = False
    return {
        "message": "Modelo re-entrenado exitosamente",
        "details": result,
        "timestamp": datetime.now().isoformat()
    }

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
@app.post("/test-webhook")
async def test_webhook():
    """Endpoint de prueba"""
    print("🧪 TEST WEBHOOK EJECUTADO")
    return {"message": "Test successful", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Endpoint raíz"""
    health_data = matcher.is_healthy()
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "model_trained": health_data["model_trained"],
        "needs_retraining": needs_retraining,
        "is_retraining": is_retraining,
        "users_loaded": health_data["users_loaded"],
        "documentation": "/docs"
    }