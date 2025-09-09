from typing import List, Dict
from fastapi import HTTPException
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

from ..utils.database import DatabaseManager
from ..utils.preprocessing import FeaturePreprocessor
from ..config.settings import settings

class AcademicMatcher:
    """Clase principal del recomendador - EL CEREBRO DEL SISTEMA"""
    
    def __init__(self):
        self.knn_model = None
        self.preprocessor = FeaturePreprocessor()
        self.db_manager = DatabaseManager()
        self.user_data = None
        self.feature_matrix = None
        self.model_trained = False
    
    def train_model(self):
        """Entrena el modelo KNN - AQUÍ ES DONDE SE ENTRENA"""
        try:
            print("🚀 Iniciando entrenamiento del modelo...")
            
            # Paso 1: Cargar datos de usuarios
            users_data = self.db_manager.get_active_users()
            user_count = len(users_data)
            
            if user_count < settings.MIN_USERS_FOR_TRAINING:
                raise ValueError(f"Insuficientes usuarios para entrenar: {user_count}")
            
            # Paso 2: Preprocesar características
            features, self.user_data = self.preprocessor.extract_user_features(users_data)
            
            # Paso 3: Crear matriz de características
            self.feature_matrix = self.preprocessor.create_feature_matrix(features)
            
            # Paso 4: Configurar y entrenar KNN
            k = min(settings.DEFAULT_K_NEIGHBORS, max(2, len(features) - 1))
            self.knn_model = NearestNeighbors(
                n_neighbors=k,
                metric=settings.KNN_METRIC,
                algorithm=settings.KNN_ALGORITHM
            )
            
            print("🧠 Entrenando modelo KNN...")
            self.knn_model.fit(self.feature_matrix)
            self.model_trained = True
            
            result = {
                "status": "success",
                "users_processed": user_count,
                "features_shape": list(self.feature_matrix.shape),
                "k_neighbors": k
            }
            
            print(f"✅ Modelo entrenado exitosamente: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            raise HTTPException(status_code=500, detail=f"Error entrenando modelo: {str(e)}")
    
    def get_recommendations(self, user_id: str, exclude_users: List[str] = [], limit: int = None):
        """Obtiene recomendaciones para un usuario específico"""
        if not self.model_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado. Llama a /retrain primero")
        
        if limit is None:
            limit = settings.DEFAULT_RECOMMENDATION_LIMIT
        
        try:
            # Encontrar el usuario en los datos
            user_mask = self.user_data['user_id'] == user_id
            if not user_mask.any():
                raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
            
            user_idx = self.user_data[user_mask].index[0]
            user_features = self.feature_matrix[user_idx].reshape(1, -1)
            
            # Obtener vecinos más cercanos
            distances, indices = self.knn_model.kneighbors(user_features)
            
            recommendations = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if i == 0:  # Saltar el mismo usuario
                    continue
                    
                candidate_id = self.user_data.iloc[idx]['user_id']
                
                # Filtrar usuarios excluidos
                if candidate_id in exclude_users:
                    continue
                
                # Calcular score de similitud (1 - distancia coseno)
                similarity_score = max(0, 1 - distance)
                
                # Obtener datos del usuario candidato
                candidate_data = self.user_data.iloc[idx]
                
                recommendation = self._build_recommendation(
                    candidate_id, similarity_score, candidate_data, user_idx, idx
                )
                recommendations.append(recommendation)
                
                if len(recommendations) >= limit:
                    break
            
            return recommendations
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}")
    
    def _build_recommendation(self, candidate_id: str, similarity_score: float, 
                            candidate_data, user_idx: int, candidate_idx: int) -> Dict:
        """Construye un objeto de recomendación completo"""
        return {
            "user_id": candidate_id,
            "similarity_score": float(similarity_score),
            "match_reasons": self.preprocessor.get_match_reasons(
                self.user_data, user_idx, candidate_idx
            ),
            "profile_preview": {
                "age": candidate_data.get('profile', {}).get('age', 'No especificado'),
                "semester": candidate_data.get('profile', {}).get('semester', 'No especificado'),
                "top_skills": candidate_data.get('skills', {}).get('technical', [])[:3],
                "objectives": candidate_data.get('objectives', {}).get('primary', [])[:2]
            }
        }
    
    def get_model_stats(self):
        """Obtiene estadísticas del modelo actual"""
        if not self.model_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        return {
            "total_users": len(self.user_data),
            "feature_dimensions": self.feature_matrix.shape[1],
            "k_neighbors": self.knn_model.n_neighbors
        }
    
    def is_healthy(self):
        """Verifica el estado de salud del modelo"""
        return {
            "model_trained": self.model_trained,
            "users_loaded": len(self.user_data) if self.user_data is not None else 0
        }