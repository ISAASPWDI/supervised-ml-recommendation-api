import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class Settings:
    # MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:17032004@localhost:27017/studysync?authSource=admin")
    DATABASE_NAME = "studysync"
    COLLECTION_NAME = "users"
    
    # API
    API_TITLE = "Academic Match ML Service"
    API_VERSION = "2.0.0"  # Nueva versión optimizada
    
    # CORS
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
    
    # Machine Learning
    MIN_USERS_FOR_TRAINING = 2
    PROFILE_COMPLETION_MIN = 50
    LAST_ACTIVE_DAYS = 90
    
    # TF-IDF Features
    MAX_SKILLS_FEATURES = 100
    MAX_OBJECTIVES_FEATURES = 50
    
    # KNN - CONFIGURACIÓN CRÍTICA SEGÚN TU INVESTIGACIÓN
    OPTIMAL_K_NEIGHBORS = 3  # ✅ Tu análisis mostró que K=3 es óptimo
    MAX_K_NEIGHBORS = 10     # Límite máximo para búsquedas amplias
    KNN_METRIC = "cosine"
    KNN_ALGORITHM = "brute"
    
    # NUEVO: Filtros según indicadores de tu matriz de operacionalización
    MAX_SEMESTER_DIFFERENCE = 1  # ✅ Indicador clave: "diferencia ≤1 semestre"
    MAX_AGE_DIFFERENCE = 5       # Diferencia máxima de edad (años)
    MIN_SKILL_OVERLAP = 1        # Mínimo de habilidades en común
    
    # Recommendations
    DEFAULT_RECOMMENDATION_LIMIT = 10
    
    # Locations (Lima default)
    DEFAULT_COORDINATES = [-77.0428, -12.0464]
    
    # NUEVO: Umbrales para métricas de validación (tu matriz)
    MIN_ACCURACY_THRESHOLD = 0.80       # 80% accuracy mínima
    MIN_PRECISION_THRESHOLD = 0.75      # 75% precision mínima
    MIN_RECALL_THRESHOLD = 0.70         # 70% recall mínimo
    
    # NUEVO: Ponderación de features (ajustable según experimentos)
    FEATURE_WEIGHTS = {
        'skills': 0.30,        # 30% - Habilidades complementarias
        'objectives': 0.25,    # 25% - Objetivos similares
        'semester': 0.20,      # 20% - CRÍTICO para tu matriz
        'age': 0.10,          # 10% - Compatibilidad etaria
        'time': 0.10,         # 10% - Disponibilidad horaria
        'commitment': 0.05    # 5% - Nivel de compromiso
    }

settings = Settings()