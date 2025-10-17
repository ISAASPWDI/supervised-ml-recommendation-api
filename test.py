import pymongo
from datetime import datetime, timedelta
import random
from faker import Faker
import uuid

# Configuración
fake = Faker(['es_ES', 'es_MX'])

# Conexión a MongoDB
client = pymongo.MongoClient("mongodb://admin:17032004@localhost:27017/studysync?authSource=admin")
db = client['studysync']
collection = db['users']

# Datos para generar usuarios aleatorios
TECHNICAL_SKILLS = [
    'Flutter', 'React', 'Python', 'JavaScript', 'Java', 'C++',
    'Node.js', 'MongoDB', 'MySQL', 'PostgreSQL', 'Git', 'Docker',
    'AWS', 'Azure', 'Spring', 'Django', 'FastAPI', 'TensorFlow',
    'Machine Learning', 'Deep Learning', 'SQL', 'NoSQL', 'Kotlin', 'Swift', 'Typescript'
]

INTERESTS = [
    'Desarrollo Web', 'Desarrollo Móvil', 'Inteligencia Artificial',
    'Data Science', 'Ciberseguridad', 'Diseño UX/UI', 'DevOps',
    'Blockchain', 'IoT', 'Cloud Computing', 'Game Development',
    'Realidad Virtual', 'Realidad Aumentada'
]

OBJECTIVES = [
    'Preparar exámenes',
    'Desarrollar proyectos',
    'Aprender nuevas tecnologías',
    'Networking profesional',
    'Hackathons y competencias',
    'Tutorías y mentorías',
    'Investigación académica',
    'Práctica de programación',
    'Formar equipo para proyecto'
]

TIME_AVAILABILITY = [
    'Menos de 5 horas/semana',
    '5-10 horas/semana',
    '10-15 horas/semana',
    '15-20 horas/semana',
    'Más de 20 horas/semana',
]

GROUP_SIZES = ['Pequeño (2-3)', 'Mediano (4-6)', 'Grande (7+)']
COMMITMENT_LEVELS = ['Bajo', 'Moderado', 'Alto']
SKILL_LEVELS = ['Básico', 'Intermedio', 'Avanzado']

UNIVERSITIES = [
    'Universidad Nacional del Centro del Perú',
    'Universidad Nacional Mayor de San Marcos',
    'Pontificia Universidad Católica del Perú',
    'Universidad Nacional de Ingeniería',
    'Universidad de Lima'
]

FACULTIES = [
    'Ingeniería de Sistemas',
    'Ingeniería de Software',
    'Ciencias de la Computación',
    'Ingeniería Informática',
    'Ingeniería Electrónica'
]

DISTRICTS_LIMA = [
    {'name': 'San Borja', 'coords': [-12.0464, -77.0428]},
    {'name': 'Miraflores', 'coords': [-12.1219, -77.0292]},
    {'name': 'San Isidro', 'coords': [-12.0947, -77.0361]},
    {'name': 'Surco', 'coords': [-12.1458, -77.0181]},
    {'name': 'La Molina', 'coords': [-12.0826, -76.9422]},
    {'name': 'Jesús María', 'coords': [-12.0742, -77.0486]},
    {'name': 'Pueblo Libre', 'coords': [-12.0756, -77.0639]},
    {'name': 'Lince', 'coords': [-12.0839, -77.0353]},
    {'name': 'San Miguel', 'coords': [-12.0772, -77.0861]},
    {'name': 'Magdalena', 'coords': [-12.0906, -77.0753]}
]

LANGUAGE_PREFS = ['Español', 'Inglés', 'Ambos']

def generate_random_user():
    """Genera un usuario aleatorio completo con email único usando UUID"""
    
    first_name = fake.first_name()
    last_name = fake.last_name()
    # Email único usando UUID corto (8 caracteres)
    unique_id = str(uuid.uuid4())[:8]
    email = f"{first_name.lower()}.{last_name.lower()}.{unique_id}@uni.edu.pe"
    
    # Seleccionar habilidades técnicas aleatorias (3-6)
    num_skills = random.randint(3, 6)
    technical_skills = random.sample(TECHNICAL_SKILLS, num_skills)
    skill_levels = [random.choice(SKILL_LEVELS) for _ in range(num_skills)]
    
    # Seleccionar intereses aleatorios (2-5)
    num_interests = random.randint(2, 5)
    interests = random.sample(INTERESTS, num_interests)
    
    # Seleccionar objetivos primarios (2-4)
    num_objectives = random.randint(2, 4)
    primary_objectives = random.sample(OBJECTIVES, num_objectives)
    
    # Datos del perfil
    age = random.randint(18, 28)
    semester = random.randint(1, 10)
    district = random.choice(DISTRICTS_LIMA)
    
    # Fechas
    join_date = datetime.now() - timedelta(days=random.randint(1, 365))
    last_active = datetime.now() - timedelta(hours=random.randint(1, 72))
    
    user = {
        "email": email,
        "password": "$2b$12$V673spOMgshVJwmk6RDetuOusVU0XV.2NQRbN4omrhkVGdQi2ybQa",
        "createdAt": join_date,
        "updatedAt": datetime.now(),
        "__v": 0,
        "activity": {
            "isOnline": random.choice([True, False]),
            "profileCompletion": 100,
            "lastSeenAt": last_active,
            "joinDate": join_date,
            "lastActive": last_active
        },
        "objectives": {
            "primary": primary_objectives,
            "timeAvailability": random.choice(TIME_AVAILABILITY),
            "preferredGroupSize": random.choice(GROUP_SIZES),
            "commitmentLevel": random.choice(COMMITMENT_LEVELS)
        },
        "profile": {
            "age": age,
            "semester": semester,
            "university": random.choice(UNIVERSITIES),
            "faculty": random.choice(FACULTIES),
            "bio": fake.text(max_nb_chars=150),
            "location": {
                "district": district['name'],
                "coordinates": district['coords']
            },
            "firstName": first_name,
            "lastName": last_name,
            "profilePicture": f"https://i.pravatar.cc/150?u={email}"
        },
        "skills": {
            "technical": technical_skills,
            "interests": interests,
            "level": skill_levels
        },
        "picture": f"https://i.pravatar.cc/300?img={random.randint(1, 70)}",
        "preferences": {
            "ageRange": {
                "min": max(18, age - 5),
                "max": min(30, age + 5)
            },
            "semesterRange": {
                "min": max(1, semester - 3),
                "max": min(10, semester + 3)
            },
            "maxDistance": random.choice([10, 20, 30, 50]),
            "languagePreference": [random.choice(LANGUAGE_PREFS)]
        },
        "privacy": {
            "showAge": random.choice([True, False]),
            "showLocation": random.choice([True, False]),
            "showSemester": random.choice([True, False])
        }
    }
    
    return user

def main():
    print("🚀 Generando 100 usuarios aleatorios con emails únicos...\n")
    
    users_to_insert = []
    
    for i in range(1, 101):
        user = generate_random_user()
        users_to_insert.append(user)
        
        if i % 10 == 0:
            print(f"✓ Generados {i}/100 usuarios...")
    
    print(f"\n📥 Insertando {len(users_to_insert)} usuarios en la base de datos...")
    
    try:
        result = collection.insert_many(users_to_insert, ordered=False)
        print(f"✅ ¡Éxito! Se insertaron {len(result.inserted_ids)} usuarios")
        print(f"\n📊 RESUMEN:")
        print(f"  - Total insertados: {len(result.inserted_ids)}")
        print(f"  - Todos con profileCompletion: 100%")
        print(f"  - Campos completos: ✓")
        
        # Verificar
        total_users = collection.count_documents({})
        complete_users = collection.count_documents({"activity.profileCompletion": 100})
        print(f"\n🔍 VERIFICACIÓN:")
        print(f"  - Total usuarios en BD: {total_users}")
        print(f"  - Usuarios completos (100%): {complete_users}")
        
    except pymongo.errors.BulkWriteError as e:
        # Obtener cuántos se insertaron exitosamente
        inserted_count = e.details.get('nInserted', 0)
        print(f"⚠️  Se insertaron {inserted_count} usuarios antes del error")
        print(f"❌ Error: Algunos emails ya existían en la base de datos")
        
        # Mostrar el primer error
        if e.details.get('writeErrors'):
            first_error = e.details['writeErrors'][0]
            print(f"   Primer email duplicado: {first_error['keyValue']['email']}")
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    
    finally:
        client.close()
        print("\n🔌 Conexión cerrada")

if __name__ == "__main__":
    main()