import pymongo
from fastapi import HTTPException
from ..config.settings import settings

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.collection = None
    
    def connect(self):
        """Establece conexión a MongoDB"""
        try:
            self.client = pymongo.MongoClient(settings.MONGODB_URI)
            db = self.client[settings.DATABASE_NAME]
            self.collection = db[settings.COLLECTION_NAME]
            return self.collection
        except Exception as e:
            print(f"Error conectando a MongoDB: {e}")
            raise HTTPException(status_code=500, detail="Error de conexión a base de datos")
    
    def get_active_users(self):
        """
        Obtiene usuarios activos con SOLO LOS CAMPOS NECESARIOS
        ✅ ELIMINADOS: skills.level, commitmentLevel, preferences, activity, privacy
        """
        # ❌ ANTES: if not self.collection:
        # ✅ AHORA: comparar con None
        if self.collection is None:
            self.connect()
        
        try:
            pipeline = [
                {
                    "$match": {
                        "activity.profileCompletion": {"$gte": settings.PROFILE_COMPLETION_MIN}
                    }
                },
                {
                    "$project": {
                        # ✅ Campos necesarios
                        "user_id": {"$toString": "$_id"},
                        
                        # Skills
                        "skills.technical": 1,
                        "skills.interests": 1,
                        
                        # Objectives
                        "objectives.primary": 1,
                        "objectives.timeAvailability": 1,
                        
                        # Profile
                        "profile.firstName": 1,
                        "profile.age": 1,
                        "profile.university": 1,
                        "profile.location": 1,
                    }
                }
            ]
            
            users = list(self.collection.aggregate(pipeline))
            print(f"✅ {len(users)} usuarios cargados (solo campos necesarios)")
            return users
        
        except Exception as e:
            print(f"❌ Error obteniendo usuarios: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_user_activity_stats(self):
        """Estadísticas básicas de usuarios"""
        if self.collection is None:
            self.connect()
            
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_users": {"$sum": 1},
                        "avg_completion": {"$avg": "$activity.profileCompletion"}
                    }
                }
            ]
            
            stats = list(self.collection.aggregate(pipeline))
            return stats[0] if stats else {}
            
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def get_user_by_id(self, user_id: str):
        """Obtiene un usuario específico por ID"""
        if self.collection is None:
            self.connect()
        
        try:
            from bson import ObjectId
            user = self.collection.find_one({"_id": ObjectId(user_id)})
            return user
        except Exception as e:
            print(f"Error obteniendo usuario {user_id}: {e}")
            return None
        
    def close(self):
        """Cierra la conexión a MongoDB"""
        if self.client is not None:
            self.client.close()
            print("🔌 Conexión a MongoDB cerrada")