from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

import pandas as pd

from .models.matcher import AcademicMatcher
from .models.schemas import (
    BulkSyncResponse, RecommendationRequest, RecommendationResponse, 
    HealthResponse, ModelStatsResponse, UserSyncRequest
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
    result = matcher.get_recommendations(
        user_id=request.user_id,
        exclude_users=request.exclude_users,
        limit=request.limit
    )
    
    return RecommendationResponse(
        recommendations=result["recommendations"],  # lista de recomendaciones
        total_filtered=result["total_filtered"],    # número total
        model_version=settings.API_VERSION,
        generated_at=datetime.now().isoformat()
    )
@app.post("/users/sync")
async def sync_user(request: UserSyncRequest):
    """
    Sincroniza un usuario específico desde MongoDB al sistema de ML
    """
    try:
        print(f"🔄 Sincronizando usuario {request.user_id}...")
        
        # Verificar si ya existe
        user_mask = matcher.user_data['user_id'] == request.user_id
        exists = user_mask.any()
        
        if exists and not request.force_reload:
            return {
                "success": True,
                "message": f"Usuario {request.user_id} ya existe en el sistema",
                "action": "skipped"
            }
        
        # Recargar usuario
        if exists:
            # Eliminar usuario existente primero
            matcher.user_data = matcher.user_data[~user_mask]
            print(f"   Usuario existente eliminado, recargando...")
        
        # Cargar desde MongoDB
        matcher._reload_single_user(request.user_id)
        
        return {
            "success": True,
            "message": f"Usuario {request.user_id} sincronizado exitosamente",
            "action": "reloaded" if exists else "added",
            "total_users": len(matcher.user_data)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error sincronizando: {str(e)}")


@app.post("/users/sync-all", response_model=BulkSyncResponse)
async def sync_all_users():
    """
    Recarga TODOS los usuarios desde MongoDB
    Útil después de migraciones o cambios masivos
    """
    try:
        print("🔄 Iniciando sincronización masiva...")
        
        # Obtener todos los usuarios de MongoDB
        from bson import ObjectId
        users_cursor = matcher.db['users'].find({
            "profile": {"$exists": True},  # Solo usuarios con perfil completo
            "activity.isActive": True  # Solo usuarios activos
        })
        
        synced = 0
        failed = 0
        
        # Limpiar DataFrame actual
        matcher.user_data = pd.DataFrame()
        matcher.features_list = []
        
        for user_doc in users_cursor:
            try:
                user_id = str(user_doc['_id'])
                user_row = matcher._convert_mongo_doc_to_dataframe_row(user_doc)
                
                # Agregar al DataFrame
                matcher.user_data = pd.concat(
                    [matcher.user_data, pd.DataFrame([user_row])], 
                    ignore_index=True
                )
                
                synced += 1
                
                if synced % 10 == 0:
                    print(f"   Procesados: {synced} usuarios...")
                
            except Exception as e:
                print(f"❌ Error procesando usuario {user_id}: {e}")
                failed += 1
        
        # Reconstruir features y modelo
        print("🔨 Reconstruyendo features y modelo KNN...")
        matcher._rebuild_features()
        matcher._rebuild_knn_model()
        
        print(f"✅ Sincronización completa:")
        print(f"   Sincronizados: {synced}")
        print(f"   Fallidos: {failed}")
        
        return BulkSyncResponse(
            success=True,
            users_synced=synced,
            users_failed=failed,
            message=f"Sincronización masiva completada. {synced} usuarios cargados."
        )
        
    except Exception as e:
        print(f"❌ Error en sincronización masiva: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en sincronización: {str(e)}"
        )


@app.get("/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    """
    Verifica si un usuario existe en el sistema de ML
    """
    user_mask = matcher.user_data['user_id'] == user_id
    exists = user_mask.any()
    
    if exists:
        user_idx = matcher.user_data[user_mask].index[0]
        user_info = matcher.features_list[user_idx]
        
        return {
            "exists": True,
            "user_id": user_id,
            "data": {
                "university": user_info.get('university'),
                "semester": user_info.get('semester'),
                "age": user_info.get('age'),
            }
        }
    
    return {
        "exists": False,
        "user_id": user_id,
        "message": "Usuario no encontrado en el sistema de ML"
    }

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
@app.get("/model/validation")
async def validate_model():
    """Endpoint para obtener métricas de validación (tu matriz)"""
    metrics = matcher.calculate_validation_metrics()
    return {
        "validation_metrics": metrics,
        "interpretation": {
            "accuracy_status": "PASS" if metrics['accuracy'] >= 0.80 else "FAIL",
            "precision_status": "PASS" if metrics['precision'] >= 0.75 else "FAIL",
            "recall_status": "PASS" if metrics['recall'] >= 0.70 else "FAIL"
        }
    }