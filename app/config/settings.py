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
    API_VERSION = "1.0.0"
    
    # CORS
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
    
    # Machine Learning
    MIN_USERS_FOR_TRAINING = 2
    PROFILE_COMPLETION_MIN = 50
    LAST_ACTIVE_DAYS = 90
    
    # TF-IDF
    MAX_SKILLS_FEATURES = 100
    MAX_OBJECTIVES_FEATURES = 50
    
    # KNN
    DEFAULT_K_NEIGHBORS = 5
    KNN_METRIC = "cosine"
    KNN_ALGORITHM = "brute"
    
    # Recommendations
    DEFAULT_RECOMMENDATION_LIMIT = 10
    
    # Locations (Lima default)
    DEFAULT_COORDINATES = [-77.0428, -12.0464]

settings = Settings()