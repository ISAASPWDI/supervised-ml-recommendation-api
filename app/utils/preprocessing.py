import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config.settings import settings

class FeaturePreprocessor:
    def __init__(self):
        # TF-IDF SOLO para skills y objectives - ENFOQUE EXCLUSIVO
        self.tfidf_skills = TfidfVectorizer(
            max_features=settings.MAX_SKILLS_FEATURES,
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2)  # Bigramas para mejor matching
        )
        self.tfidf_objectives = TfidfVectorizer(
            max_features=settings.MAX_OBJECTIVES_FEATURES,
            lowercase=True,
            strip_accents='unicode'
        )
        
        # PESOS SIMPLIFICADOS: SOLO Skills + Objectives
        self.feature_weights = {
            'skills_technical': 0.35,   # 35% - Technical skills
            'skills_interests': 0.30,   # 30% - Interests
            'objectives': 0.35,         # 35% - Primary objectives
        }
    
    def extract_user_features(self, users_data):
        """Extrae SOLO skills.technical, skills.interests y objectives.primary"""
        if not users_data:
            raise ValueError("No hay datos de usuarios para procesar")
        
        user_df = pd.DataFrame(users_data)
        features = []
        
        print(f"\n{'='*70}")
        print(f"🎯 EXTRAYENDO FEATURES - MODO: SKILLS & OBJECTIVES ONLY")
        print(f"{'='*70}\n")
        
        for _, user in user_df.iterrows():
            try:
                feature_dict = self._process_single_user(user)
                features.append(feature_dict)
            except Exception as e:
                print(f"⚠️ Error procesando usuario {user.get('user_id', 'unknown')}: {e}")
                continue
        
        print(f"\n✅ Features procesadas para {len(features)} usuarios")
        print(f"   📊 Componentes: Technical Skills + Interests + Objectives\n")
        return features, user_df
    
    def _process_single_user(self, user):
        """Procesa usuario - SOLO SKILLS Y OBJECTIVES"""
        user_id = user.get('user_id', 'unknown')
        
        # 1. SKILLS TECHNICAL (PRIORIDAD ALTA)
        skills_technical = user.get('skills', {}).get('technical', [])
        skills_technical_text = ' '.join(skills_technical) if skills_technical else ''
        
        # 2. SKILLS INTERESTS (PRIORIDAD ALTA)
        skills_interests = user.get('skills', {}).get('interests', [])
        skills_interests_text = ' '.join(skills_interests) if skills_interests else ''
        
        # 3. OBJECTIVES PRIMARY (PRIORIDAD ALTA)
        objectives = user.get('objectives', {}).get('primary', [])
        objectives_text = ' '.join(objectives) if objectives else ''
        
        # Validación: al menos debe tener algo
        if not (skills_technical_text or skills_interests_text or objectives_text):
            print(f"⚠️ Usuario {user_id}: Sin skills ni objectives - usando placeholder")
            skills_technical_text = 'sin_skills'
            objectives_text = 'sin_objetivos'
        
        # METADATA (NO SE USA EN MATCHING, solo para referencia)
        profile = user.get('profile', {})
        
        feature_dict = {
            'user_id': user_id,
            # FEATURES PARA MATCHING
            'skills_technical_text': skills_technical_text,
            'skills_interests_text': skills_interests_text,
            'objectives_text': objectives_text,
            # METADATA (solo informativa, NO entra al KNN)
            'semester': profile.get('semester', 5),
            'age': profile.get('age', 20),
            'location': profile.get('location', {}).get('coordinates', settings.DEFAULT_COORDINATES),
            'university': profile.get('university', 'No especificada'),
            'time_availability': user.get('objectives', {}).get('timeAvailability', 'No especificado'),
        }
        
        return feature_dict
    
    def create_feature_matrix(self, features):
        """Crea matriz SOLO con Skills (technical + interests) + Objectives"""
        if len(features) < 2:
            raise ValueError("Insuficientes características procesadas")
        
        print(f"\n{'='*70}")
        print(f"🔧 CONSTRUYENDO MATRIZ DE FEATURES - PONDERACIÓN")
        print(f"{'='*70}\n")
        
        # 1. TF-IDF para TECHNICAL SKILLS - 35%
        technical_texts = [f['skills_technical_text'] for f in features]
        technical_matrix = self.tfidf_skills.fit_transform(technical_texts).toarray()
        technical_weighted = technical_matrix * self.feature_weights['skills_technical']
        
        print(f"✅ Technical Skills:")
        print(f"   Dimensiones: {technical_matrix.shape}")
        print(f"   Peso aplicado: {self.feature_weights['skills_technical']*100:.0f}%")
        print(f"   Vocabulario: {len(self.tfidf_skills.vocabulary_)} términos únicos\n")
        
        # 2. TF-IDF para INTERESTS - 30%
        interests_texts = [f['skills_interests_text'] for f in features]
        # Usamos el mismo vectorizador pero con fit separado para mantener independencia
        tfidf_interests = TfidfVectorizer(
            max_features=50,  # Menos features para interests
            lowercase=True,
            strip_accents='unicode'
        )
        interests_matrix = tfidf_interests.fit_transform(interests_texts).toarray()
        interests_weighted = interests_matrix * self.feature_weights['skills_interests']
        
        print(f"✅ Interests:")
        print(f"   Dimensiones: {interests_matrix.shape}")
        print(f"   Peso aplicado: {self.feature_weights['skills_interests']*100:.0f}%")
        print(f"   Vocabulario: {len(tfidf_interests.vocabulary_)} términos únicos\n")
        
        # 3. TF-IDF para OBJECTIVES - 35%
        objectives_texts = [f['objectives_text'] for f in features]
        objectives_matrix = self.tfidf_objectives.fit_transform(objectives_texts).toarray()
        objectives_weighted = objectives_matrix * self.feature_weights['objectives']
        
        print(f"✅ Objectives:")
        print(f"   Dimensiones: {objectives_matrix.shape}")
        print(f"   Peso aplicado: {self.feature_weights['objectives']*100:.0f}%")
        print(f"   Vocabulario: {len(self.tfidf_objectives.vocabulary_)} términos únicos\n")
        
        # 4. CONCATENAR: Technical + Interests + Objectives
        feature_matrix = np.hstack([
            technical_weighted,
            interests_weighted,
            objectives_weighted
        ])
        
        print(f"{'='*70}")
        print(f"✅ MATRIZ FINAL CONSTRUIDA")
        print(f"{'='*70}")
        print(f"   Shape total: {feature_matrix.shape}")
        print(f"   Total features: {feature_matrix.shape[1]}")
        print(f"   Usuarios: {feature_matrix.shape[0]}")
        print(f"   Distribución:")
        print(f"     • Technical Skills: {technical_matrix.shape[1]} features (35%)")
        print(f"     • Interests: {interests_matrix.shape[1]} features (30%)")
        print(f"     • Objectives: {objectives_matrix.shape[1]} features (35%)")
        print(f"\n   ⚠️  Edad, semestre, tiempo y ubicación NO incluidos en matching")
        print(f"   ℹ️  Solo se usan como metadata informativa\n")
        
        return feature_matrix
    
    def get_match_reasons(self, user_data, user_idx, candidate_idx):
        """Calcula razones del match basadas SOLO en Skills + Objectives"""
        try:
            user = user_data.iloc[user_idx]
            candidate = user_data.iloc[candidate_idx]
            
            reasons = []
            
            # 1. TECHNICAL SKILLS en común
            user_technical = set(user.get('skills', {}).get('technical', []))
            candidate_technical = set(candidate.get('skills', {}).get('technical', []))
            common_technical = user_technical.intersection(candidate_technical)
            
            if common_technical:
                tech_list = list(common_technical)[:4]  # Top 4
                reasons.append(f"💻 Technical: {', '.join(tech_list)}")
            
            # 2. INTERESTS en común
            user_interests = set(user.get('skills', {}).get('interests', []))
            candidate_interests = set(candidate.get('skills', {}).get('interests', []))
            common_interests = user_interests.intersection(candidate_interests)
            
            if common_interests:
                interests_list = list(common_interests)[:3]
                reasons.append(f"💡 Interests: {', '.join(interests_list)}")
            
            # 3. OBJECTIVES similares
            user_objectives = set(user.get('objectives', {}).get('primary', []))
            candidate_objectives = set(candidate.get('objectives', {}).get('primary', []))
            common_objectives = user_objectives.intersection(candidate_objectives)
            
            if common_objectives:
                obj_list = list(common_objectives)[:2]
                reasons.append(f"🎯 Objectives: {', '.join(obj_list)}")
            
            # 4. INFO COMPLEMENTARIA (NO afecta el matching)
            candidate_profile = candidate.get('profile', {})
            candidate_semester = candidate_profile.get('semester', 'N/A')
            candidate_university = candidate_profile.get('university', 'N/A')
            
            reasons.append(f"ℹ️ Semestre {candidate_semester} - {candidate_university}")
            
            return reasons if reasons else ["✅ Perfil compatible por skills y objetivos"]
            
        except Exception as e:
            print(f"⚠️ Error calculando razones: {e}")
            return ["✅ Perfil compatible"]
    
    def _get_user_skills(self, user):
        """Extrae todos los skills (technical + interests) normalizados"""
        skills_data = user.get('skills', {})
        technical = [s.lower().strip() for s in skills_data.get('technical', []) if s]
        interests = [s.lower().strip() for s in skills_data.get('interests', []) if s]
        return set(technical + interests)