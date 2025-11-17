import os
from dotenv import load_dotenv
from ast import literal_eval

load_dotenv()

class Settings:
    """Configuración general del servicio ML"""

    # 🗄️ MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "studysync")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "users")

    # 🚀 API
    API_TITLE = "Academic Match ML Service"
    API_VERSION = "2.0.0"

    # 🔐 Webhook Security
    WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY")  # Opcional, para validar requests del backend

    # 🌐 CORS
    try:
        CORS_ORIGINS = literal_eval(os.getenv("CORS_ORIGINS", "['http://localhost:3000']"))
    except Exception:
        CORS_ORIGINS = ["http://localhost:3000"]

    # 🧠 Machine Learning
    MIN_USERS_FOR_TRAINING = int(os.getenv("MIN_USERS_FOR_TRAINING", 2))
    PROFILE_COMPLETION_MIN = int(os.getenv("PROFILE_COMPLETION_MIN", 50))
    LAST_ACTIVE_DAYS = int(os.getenv("LAST_ACTIVE_DAYS", 90))

    # 🔢 TF-IDF
    MAX_SKILLS_FEATURES = int(os.getenv("MAX_SKILLS_FEATURES", 100))
    MAX_OBJECTIVES_FEATURES = int(os.getenv("MAX_OBJECTIVES_FEATURES", 50))

    # 🧩 KNN Configuración
    OPTIMAL_K_NEIGHBORS = int(os.getenv("OPTIMAL_K_NEIGHBORS", 3))
    MAX_K_NEIGHBORS = int(os.getenv("MAX_K_NEIGHBORS", 10))
    KNN_METRIC = os.getenv("KNN_METRIC", "cosine")
    KNN_ALGORITHM = os.getenv("KNN_ALGORITHM", "brute")

    # 📊 Filtros
    MAX_SEMESTER_DIFFERENCE = int(os.getenv("MAX_SEMESTER_DIFFERENCE", 1))
    MAX_AGE_DIFFERENCE = int(os.getenv("MAX_AGE_DIFFERENCE", 5))
    MIN_SKILL_OVERLAP = int(os.getenv("MIN_SKILL_OVERLAP", 1))

    # 🔁 Recomendaciones
    DEFAULT_RECOMMENDATION_LIMIT = int(os.getenv("DEFAULT_RECOMMENDATION_LIMIT", 10))

    # 📍 Coordenadas por defecto
    DEFAULT_COORDINATES = [-77.0428, -12.0464]

    # 📈 Métricas de validación
    MIN_ACCURACY_THRESHOLD = float(os.getenv("MIN_ACCURACY_THRESHOLD", 0.80))
    MIN_PRECISION_THRESHOLD = float(os.getenv("MIN_PRECISION_THRESHOLD", 0.75))
    MIN_RECALL_THRESHOLD = float(os.getenv("MIN_RECALL_THRESHOLD", 0.70))

    # ⚖️ Ponderación de características
    FEATURE_WEIGHTS = {
        'skills': float(os.getenv("WEIGHT_SKILLS", 0.30)),
        'objectives': float(os.getenv("WEIGHT_OBJECTIVES", 0.25)),
        'semester': float(os.getenv("WEIGHT_SEMESTER", 0.20)),
        'age': float(os.getenv("WEIGHT_AGE", 0.10)),
        'time': float(os.getenv("WEIGHT_TIME", 0.10)),
        'commitment': float(os.getenv("WEIGHT_COMMITMENT", 0.05)),
    }

settings = Settings()