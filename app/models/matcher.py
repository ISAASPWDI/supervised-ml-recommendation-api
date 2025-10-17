from typing import List, Dict
from fastapi import HTTPException
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from geopy.distance import geodesic

from ..utils.database import DatabaseManager
from ..utils.preprocessing import FeaturePreprocessor
from ..config.settings import settings

class AcademicMatcher:
    """Matcher optimizado - ENFOQUE PRINCIPAL EN SEMESTRE"""
    
    def __init__(self):
        self.knn_model = None
        self.preprocessor = FeaturePreprocessor()
        self.db_manager = DatabaseManager()
        self.user_data = None
        self.feature_matrix = None
        self.model_trained = False
        self.features_list = None
    
    def train_model(self):
        """Entrena con K=3 (óptimo según tu análisis)"""
        try:
            print("🚀 Iniciando entrenamiento del modelo KNN...")
            
            users_data = self.db_manager.get_active_users()
            user_count = len(users_data)
            
            if user_count < settings.MIN_USERS_FOR_TRAINING:
                raise ValueError(f"Insuficientes usuarios: {user_count} < {settings.MIN_USERS_FOR_TRAINING}")
            
            self.features_list, self.user_data = self.preprocessor.extract_user_features(users_data)
            self.feature_matrix = self.preprocessor.create_feature_matrix(self.features_list)
            
            optimal_k = min(
                settings.OPTIMAL_K_NEIGHBORS,
                max(3, len(self.features_list) - 1)
            )
            
            self.knn_model = NearestNeighbors(
                n_neighbors=optimal_k,
                metric=settings.KNN_METRIC,
                algorithm=settings.KNN_ALGORITHM
            )
            
            print(f"🧠 Entrenando KNN con k={optimal_k}...")
            self.knn_model.fit(self.feature_matrix)
            self.model_trained = True
            
            result = {
                "status": "success",
                "users_processed": user_count,
                "features_shape": list(self.feature_matrix.shape),
                "k_neighbors": optimal_k,
                "feature_weights": self.preprocessor.feature_weights
            }
            
            print(f"✅ Modelo entrenado: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error entrenando: {e}")
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    def _generate_smart_preferences(self, user_info: Dict) -> Dict:
        """Genera preferencias automáticas - DISTANCIA MUY AMPLIA"""
        user_age = user_info.get('age', 21)
        user_semester = user_info.get('semester', 5)
        
        return {
            'age_min': max(18, user_age - 3),
            'age_max': min(50, user_age + 3),
            'semester_min': max(1, user_semester - 2),
            'semester_max': min(12, user_semester + 2),
            'max_distance': 500  # 500 km - MUY AMPLIO, prácticamente sin restricción
        }
    
    def get_recommendations(self, user_id: str, exclude_users: List[str] = [], limit: int = None):
        """Recomendaciones con ENFOQUE EN SEMESTRE (sin filtrar por edad)"""
        if not self.model_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        if limit is None:
            limit = settings.DEFAULT_RECOMMENDATION_LIMIT
        
        try:
            # 1. Encontrar usuario
            user_mask = self.user_data['user_id'] == user_id
            if not user_mask.any():
                raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado")
            
            user_idx = self.user_data[user_mask].index[0]
            user_features = self.feature_matrix[user_idx].reshape(1, -1)
            user_info = self.features_list[user_idx]
            
            # Generar preferencias automáticas
            user_prefs = self._generate_smart_preferences(user_info)
            
            print(f"\n{'='*70}")
            print(f"👤 Usuario: {user_id}")
            print(f"   Edad: {user_info['age']} (ya no se filtra por edad)")
            print(f"   Semestre: {user_info['semester']} → Rango: {user_prefs['semester_min']}-{user_prefs['semester_max']}")
            print(f"   📍 Distancia máxima: {user_prefs['max_distance']} km (sin restricción práctica)")
            print(f"{'='*70}\n")
            
            # 2. KNN búsqueda
            search_k = min(
                len(self.user_data) - 1,
                limit * 3
            )
            
            if search_k > self.knn_model.n_neighbors:
                temp_knn = NearestNeighbors(
                    n_neighbors=search_k,
                    metric=settings.KNN_METRIC,
                    algorithm=settings.KNN_ALGORITHM
                )
                temp_knn.fit(self.feature_matrix)
                distances, indices = temp_knn.kneighbors(user_features)
            else:
                distances, indices = self.knn_model.kneighbors(user_features)
            
            print(f"🔍 KNN encontró {len(indices[0])} vecinos cercanos\n")
            
            # 3. Filtrar candidatos - SOLO SEMESTRE y EXCLUSIONES
            recommendations = []
            filtered_counts = {
                'excluded': 0,
                'semester': 0,
                'distance': 0,
                'accepted': 0
            }
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if i == 0:  # Saltar el mismo usuario
                    continue
                
                candidate_id = self.user_data.iloc[idx]['user_id']
                candidate_info = self.features_list[idx]
                candidate_data = self.user_data.iloc[idx]
                
                print(f"[{i}] Evaluando: {candidate_id}")
                
                # FILTRO 1: Excluidos
                if candidate_id in exclude_users:
                    print(f"    ❌ EXCLUIDO manualmente")
                    filtered_counts['excluded'] += 1
                    continue
                
                # FILTRO 2: SEMESTRE (FILTRO PRINCIPAL)
                semester_diff = abs(user_info['semester'] - candidate_info['semester'])
                if semester_diff > settings.MAX_SEMESTER_DIFFERENCE:
                    print(f"    ❌ SEMESTRE: diff={semester_diff} > {settings.MAX_SEMESTER_DIFFERENCE} (candidato: {candidate_info['semester']})")
                    filtered_counts['semester'] += 1
                    continue
                print(f"    ✓ Semestre OK (diff: {semester_diff}) ⭐ CRITERIO PRINCIPAL")
                
                # 🔹 EDAD - Solo informativo
                candidate_age = candidate_info['age']
                print(f"    ℹ️ Edad del candidato: {candidate_age} (no se usa como filtro)")
                
                # FILTRO 3: Semestre (rango de preferencias)
                candidate_semester = candidate_info['semester']
                if not (user_prefs['semester_min'] <= candidate_semester <= user_prefs['semester_max']):
                    print(f"    ❌ SEMESTRE RANGO: {candidate_semester} fuera de [{user_prefs['semester_min']}, {user_prefs['semester_max']}]")
                    filtered_counts['semester'] += 1
                    continue
                print(f"    ✓ Semestre rango OK ({candidate_semester})")
                
                # DISTANCIA: Solo informativo
                distance_km = self._calculate_distance(user_info, candidate_info)
                print(f"    ℹ️  Distancia: {distance_km:.1f} km (informativo, no filtra)")
                
                # Score con BONUS ALTO por semestre
                base_similarity = max(0, 1 - distance)
                semester_bonus = 0
                if semester_diff == 0:
                    semester_bonus = 0.20
                elif semester_diff == 1:
                    semester_bonus = 0.15
                
                final_score = min(1.0, base_similarity + semester_bonus)
                
                recommendation = self._build_recommendation(
                    candidate_id, final_score, candidate_data, 
                    user_idx, idx, semester_diff, distance_km
                )
                recommendations.append(recommendation)
                filtered_counts['accepted'] += 1
                print(f"    ✅ ACEPTADO - Score: {final_score:.3f}\n")
                
                if len(recommendations) >= limit:
                    break
            
            # Resumen
            print(f"\n{'='*70}")
            print(f"📊 RESUMEN DE FILTRADO:")
            print(f"   Total evaluados: {len(indices[0]) - 1}")
            print(f"   Excluidos manualmente: {filtered_counts['excluded']}")
            print(f"   ❌ Rechazados por SEMESTRE: {filtered_counts['semester']}")
            print(f"   📍 Distancia y edad NO utilizadas como filtro")
            print(f"   ✅ ACEPTADOS: {filtered_counts['accepted']}")
            print(f"{'='*70}\n")
            
            compatibility_metrics = self._calculate_compatibility_metrics(recommendations, user_info)
            
            return {
                "recommendations": recommendations,
                "compatibility_metrics": compatibility_metrics,
                "total_filtered": len(recommendations),
                "user_preferences_applied": user_prefs,
                "debug_filters": filtered_counts,
                "filter_priority": "SEMESTRE (principal) > Distancia (informativo)"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    
    def _calculate_distance(self, user_info, candidate_info):
        """Calcula distancia solo para información"""
        try:
            user_coords = user_info.get('location', settings.DEFAULT_COORDINATES)
            candidate_coords = candidate_info.get('location', settings.DEFAULT_COORDINATES)
            
            distance = geodesic(
                (user_coords[1], user_coords[0]),
                (candidate_coords[1], candidate_coords[0])
            ).km
            
            return distance
        except:
            return 0.0
    
    def _build_recommendation(self, candidate_id: str, similarity_score: float, 
                            candidate_data, user_idx: int, candidate_idx: int,
                            semester_diff: int, distance_km: float = None) -> Dict:
        """Construye recomendación con información de distancia"""
        profile = candidate_data.get('profile', {})
        
        recommendation = {
            "user_id": candidate_id,
            "similarity_score": float(similarity_score),
            "compatibility_indicators": {
                "semester_difference": semester_diff,
                "semester_compatible": semester_diff <= 1,
                "age": profile.get('age'),
                "semester": profile.get('semester')
            },
            "match_reasons": self.preprocessor.get_match_reasons(
                self.user_data, user_idx, candidate_idx
            ),
            "profile_preview": {
                "firstName": profile.get('firstName', 'Usuario'),
                "top_skills": candidate_data.get('skills', {}).get('technical', [])[:5],
                "objectives": candidate_data.get('objectives', {}).get('primary', [])[:4],
                "time_availability": candidate_data.get('objectives', {}).get('timeAvailability', 'No especificado'),
                "commitment_level": candidate_data.get('objectives', {}).get('commitmentLevel', 'No especificado'),
                "semester": profile.get('semester'),
                "university": profile.get('university', 'No especificada')
            }
        }
        
        # Agregar distancia como información adicional
        if distance_km is not None:
            recommendation["distance_info"] = {
                "distance_km": round(distance_km, 1),
                "note": "Distancia informativa, no afecta el matching"
            }
        
        return recommendation
    
    def _calculate_compatibility_metrics(self, recommendations: List[Dict], user_info: Dict) -> Dict:
        """Calcula métricas de compatibilidad"""
        if not recommendations:
            return {
                "pct_semester_compatible": 0,
                "avg_semester_difference": 0,
                "total_matches": 0
            }
        
        total = len(recommendations)
        semester_compatible = sum(1 for r in recommendations if r['compatibility_indicators']['semester_compatible'])
        avg_semester_diff = sum(r['compatibility_indicators']['semester_difference'] for r in recommendations) / total
        
        return {
            "pct_semester_compatible": round((semester_compatible / total) * 100, 2),
            "avg_semester_difference": round(avg_semester_diff, 2),
            "total_matches": total,
            "primary_filter": "semester_difference"
        }
    
    def get_model_stats(self):
        """Estadísticas del modelo"""
        if not self.model_trained:
            raise HTTPException(status_code=400, detail="Modelo no entrenado")
        
        return {
            "total_users": len(self.user_data),
            "feature_dimensions": self.feature_matrix.shape[1],
            "k_neighbors": self.knn_model.n_neighbors,
            "feature_weights": self.preprocessor.feature_weights,
            "filter_strategy": "Semester-focused (distance informative only)"
        }
    
    def is_healthy(self):
        """Health check"""
        return {
            "model_trained": self.model_trained,
            "users_loaded": len(self.user_data) if self.user_data is not None else 0,
            "filtering_mode": "semester_priority"
        }