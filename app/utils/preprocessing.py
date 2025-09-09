import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config.settings import settings

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf_skills = TfidfVectorizer(max_features=settings.MAX_SKILLS_FEATURES)
        self.tfidf_objectives = TfidfVectorizer(max_features=settings.MAX_OBJECTIVES_FEATURES)
        
        # Mapeos para features categóricas
        self.time_mapping = {
            'Mañana': 0, 'Tarde': 1, 'Noche': 2, 
            'Fines de semana': 3, 'Flexible': 4
        }
        self.commitment_mapping = {
            'Casual': 0, 'Moderado': 1, 'Intensivo': 2
        }
    
    def extract_user_features(self, users_data):
        """Extrae y procesa características de los usuarios"""
        if not users_data:
            raise ValueError("No hay datos de usuarios para procesar")
        
        user_df = pd.DataFrame(users_data)
        features = []
        
        for _, user in user_df.iterrows():
            try:
                feature_dict = self._process_single_user(user)
                features.append(feature_dict)
            except Exception as e:
                print(f"Error procesando usuario {user.get('user_id', 'unknown')}: {e}")
                continue
        
        print(f"Features procesadas para {len(features)} usuarios")
        return features, user_df
    
    def _process_single_user(self, user):
        """Procesa un usuario individual"""
        # 1. Skills como texto para TF-IDF
        skills_technical = user.get('skills', {}).get('technical', [])
        skills_interests = user.get('skills', {}).get('interests', [])
        skills_text = ' '.join(skills_technical + skills_interests) if skills_technical or skills_interests else 'sin skills'
        
        # 2. Objetivos como texto para TF-IDF  
        objectives = user.get('objectives', {}).get('primary', [])
        objectives_text = ' '.join(objectives) if objectives else 'sin objetivos'
        
        # 3. Features numéricas con valores por defecto
        profile = user.get('profile', {})
        location = profile.get('location', {})
        coordinates = location.get('coordinates', settings.DEFAULT_COORDINATES)
        
        numeric_features = [
            profile.get('age', 20),
            profile.get('semester', 5),
            coordinates[0] if len(coordinates) > 0 else settings.DEFAULT_COORDINATES[0],  # lng
            coordinates[1] if len(coordinates) > 1 else settings.DEFAULT_COORDINATES[1],  # lat
        ]
        
        # 4. Features categóricas con valores por defecto
        objectives_data = user.get('objectives', {})
        categorical_features = [
            self.time_mapping.get(objectives_data.get('timeAvailability'), 1),  # Tarde por defecto
            self.commitment_mapping.get(objectives_data.get('commitmentLevel'), 1)  # Moderado por defecto
        ]
        
        return {
            'user_id': user['user_id'],
            'skills_text': skills_text,
            'objectives_text': objectives_text,
            'numeric': numeric_features,
            'categorical': categorical_features
        }
    
    def create_feature_matrix(self, features):
        """Crea la matriz final de características para ML"""
        if len(features) < 2:
            raise ValueError("No se pudieron procesar suficientes características")
        
        print("📊 Creando matrices de características...")
        
        # Extraer textos
        skills_texts = [f['skills_text'] for f in features]
        objectives_texts = [f['objectives_text'] for f in features]
        
        # Aplicar TF-IDF
        skills_matrix = self.tfidf_skills.fit_transform(skills_texts)
        objectives_matrix = self.tfidf_objectives.fit_transform(objectives_texts)
        
        # Combinar features numéricas y categóricas
        numeric_categorical = np.array([f['numeric'] + f['categorical'] for f in features])
        numeric_categorical_scaled = self.scaler.fit_transform(numeric_categorical)
        
        # Matriz final combinada
        feature_matrix = np.hstack([
            skills_matrix.toarray(),
            objectives_matrix.toarray(), 
            numeric_categorical_scaled
        ])
        
        return feature_matrix
    
    def get_match_reasons(self, user_data, user_idx, candidate_idx):
        """Calcula las razones específicas del match entre dos usuarios"""
        try:
            user = user_data.iloc[user_idx]
            candidate = user_data.iloc[candidate_idx]
            
            reasons = []
            
            # Skills en común
            user_skills = self._get_user_skills(user)
            candidate_skills = self._get_user_skills(candidate)
            common_skills = user_skills.intersection(candidate_skills)
            
            if common_skills:
                reasons.append(f"Habilidades en común: {', '.join(list(common_skills)[:3])}")
            
            # Objetivos similares
            user_objectives = set(user.get('objectives', {}).get('primary', []))
            candidate_objectives = set(candidate.get('objectives', {}).get('primary', []))
            common_objectives = user_objectives.intersection(candidate_objectives)
            
            if common_objectives:
                reasons.append(f"Objetivos similares: {', '.join(list(common_objectives)[:2])}")
            
            # Proximidad de semestre
            user_semester = user.get('profile', {}).get('semester', 0)
            candidate_semester = candidate.get('profile', {}).get('semester', 0)
            semester_diff = abs(user_semester - candidate_semester)
            
            if semester_diff <= 1:
                reasons.append("Semestres cercanos")
            
            return reasons if reasons else ["Perfil compatible"]
            
        except Exception as e:
            print(f"Error calculando razones de match: {e}")
            return ["Perfil compatible"]
    
    def _get_user_skills(self, user):
        """Obtiene todas las habilidades de un usuario"""
        skills_data = user.get('skills', {})
        technical_skills = skills_data.get('technical', [])
        interest_skills = skills_data.get('interests', [])
        return set(technical_skills + interest_skills)