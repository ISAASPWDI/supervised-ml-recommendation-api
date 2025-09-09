import pymongo
from fastapi import HTTPException
from datetime import datetime, timedelta
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
            """Obtiene todos los usuarios con perfil completo para entrenamiento"""
            if not self.collection:
                self.connect()
            
            try:
                # Información de diagnóstico
                total_users = self.collection.count_documents({})
                print(f"Total de usuarios en BD: {total_users}")
                
                # Pipeline simplificado - usa todos los usuarios con perfil completo
                pipeline = [
                    {
                        "$match": {
                            "activity.profileCompletion": {"$gte": settings.PROFILE_COMPLETION_MIN}
                            # ✅ Removido el filtro de actividad reciente
                        }
                    },
                    {
                        "$project": {
                            "user_id": {"$toString": "$_id"},
                            "skills.technical": 1,
                            "skills.interests": 1,
                            "objectives.primary": 1,
                            "profile.age": 1,
                            "profile.semester": 1,
                            "profile.location": 1,
                            "objectives.timeAvailability": 1,
                            "objectives.commitmentLevel": 1,
                            # Opcionalmente, incluir datos de actividad para análisis posterior
                            "activity.lastActive": 1,
                            "activity.profileCompletion": 1
                        }
                    },
                    {
                        "$sort": {
                            "activity.lastActive": -1,  # Ordenar por actividad reciente primero
                            "activity.profileCompletion": -1  # Luego por completitud de perfil
                        }
                    }
                ]
                
                users = list(self.collection.aggregate(pipeline))
                
                # Información adicional sobre la distribución de actividad
                recent_users = len([u for u in users if u.get('activity', {}).get('lastActive', datetime.min) >= (datetime.now() - timedelta(days=90))])
                print(f"Usuarios cargados: {len(users)}")
                print(f"  - Activos últimos 90 días: {recent_users}")
                print(f"  - Usuarios adicionales (menos activos): {len(users) - recent_users}")
                
                return users
                
            except Exception as e:
                print(f"Error cargando usuarios: {e}")
                raise HTTPException(status_code=500, detail=f"Error cargando datos: {str(e)}")
    
    def get_user_activity_stats(self):
        """Método adicional para obtener estadísticas de actividad"""
        if not self.collection:
            self.connect()
            
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_users": {"$sum": 1},
                        "avg_completion": {"$avg": "$activity.profileCompletion"},
                        "recent_active": {
                            "$sum": {
                                "$cond": [
                                    {"$gte": ["$activity.lastActive", datetime.now() - timedelta(days=90)]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }
                }
            ]
            
            stats = list(self.collection.aggregate(pipeline))
            return stats[0] if stats else {}
            
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {}
        """Obtiene usuarios activos con pipeline optimizado"""
        if not self.collection:
            self.connect()
        
        try:
            total_users = self.collection.count_documents({})
            print(f"Total de usuarios en BD: {total_users}")
            
            # Contar usuarios por criterio
            completion_filter = {"activity.profileCompletion": {"$gte": settings.PROFILE_COMPLETION_MIN}}
            users_with_completion = self.collection.count_documents(completion_filter)
            print(f"Usuarios con profileCompletion >= {settings.PROFILE_COMPLETION_MIN}: {users_with_completion}")
            
            activity_filter = {"activity.lastActive": {"$gte": datetime.now() - timedelta(days=settings.LAST_ACTIVE_DAYS)}}
            users_with_activity = self.collection.count_documents(activity_filter)
            print(f"Usuarios activos últimos {settings.LAST_ACTIVE_DAYS} días: {users_with_activity}")
            # Pipeline para obtener usuarios activos con perfil completo
            pipeline = [
                {
                    "$match": {
                        "activity.profileCompletion": {"$gte": settings.PROFILE_COMPLETION_MIN},
                        "activity.lastActive": {
                            "$gte": datetime.now() - timedelta(days=settings.LAST_ACTIVE_DAYS)
                        }
                    }
                },
                {
                    "$project": {
                        "user_id": {"$toString": "$_id"},
                        "skills.technical": 1,
                        "skills.interests": 1,
                        "objectives.primary": 1,
                        "profile.age": 1,
                        "profile.semester": 1,
                        "profile.location": 1,
                        "objectives.timeAvailability": 1,
                        "objectives.commitmentLevel": 1
                    }
                }
            ]
            
            users = list(self.collection.aggregate(pipeline))
            print(f"Usuarios que pasan ambos filtros: {len(users)}")
            # Si no hay usuarios con el filtro estricto, usar todos los usuarios disponibles
            if len(users) < settings.MIN_USERS_FOR_TRAINING:
                print("Pocos usuarios encontrados, usando todos los usuarios disponibles...")
                pipeline_simple = [
                    {
                        "$project": {
                            "user_id": {"$toString": "$_id"},
                            "skills.technical": 1,
                            "skills.interests": 1,
                            "objectives.primary": 1,
                            "profile.age": 1,
                            "profile.semester": 1,
                            "profile.location": 1,
                            "objectives.timeAvailability": 1,
                            "objectives.commitmentLevel": 1
                        }
                    }
                ]
                users = list(self.collection.aggregate(pipeline_simple))
            
            print(f"Usuarios cargados: {len(users)}")
            return users
            
        except Exception as e:
            print(f"Error cargando usuarios: {e}")
            raise HTTPException(status_code=500, detail=f"Error cargando datos: {str(e)}")
    
    def close(self):
        """Cierra la conexión a MongoDB"""
        if self.client:
            self.client.close()