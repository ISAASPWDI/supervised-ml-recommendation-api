# StudySync ML-Service: Sistema de Recomendación Académica

## 📋 Resumen Ejecutivo

**StudySync ML-Service** es un microservicio de machine learning diseñado para generar recomendaciones personalizadas de compañeros de estudio basado en compatibilidad de habilidades técnicas y objetivos académicos. Utiliza algoritmos de **K-Nearest Neighbors (KNN)** con procesamiento de lenguaje natural mediante **TF-IDF** para identificar usuarios académicamente compatibles.

## 🏗️ Arquitectura del Sistema

### Estructura del Proyecto
```
ML-SERVICE/
├── app/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuraciones del sistema
│   ├── models/
│   │   ├── __init__.py
│   │   ├── matcher.py           # Algoritmo principal KNN
│   │   └── schemas.py           # Modelos Pydantic para API
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── database.py          # Gestión de MongoDB
│   │   └── preprocessing.py     # Procesamiento de características
│   └── main.py                  # FastAPI application
├── tests/
├── requirements.txt
└── README.md
```

## 🤖 Componentes Técnicos Principales

### 1. **Feature Preprocessor** (`utils/preprocessing.py`)

**Responsabilidad**: Transformar datos de usuario en características numéricas para machine learning.

**Técnicas Implementadas**:
- **TF-IDF Vectorization**: Convierte skills técnicos y objetivos académicos en vectores numéricos
- **Feature Engineering**: Normalización y limpieza de datos de entrada
- **N-gram Processing**: Captura skills compuestos (ej: "machine learning", "data science")

**Optimizaciones**:
```python
TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=1,            # Incluir skills raros
    max_df=0.8           # Excluir skills muy comunes
)
```

### 2. **Academic Matcher** (`models/matcher.py`)

**Responsabilidad**: Algoritmo principal de recomendación basado en KNN.

**Características Técnicas**:
- **Algoritmo**: K-Nearest Neighbors con métrica coseno
- **Enfoque**: Similaridad basada únicamente en skills y objetivos académicos
- **Optimización**: Eliminación de features irrelevantes (edad, ubicación, semestre)

**Pipeline de Procesamiento**:
1. **Entrenamiento**: Carga 506 usuarios activos desde MongoDB
2. **Vectorización**: Convierte texto a matrices TF-IDF
3. **Fitting**: Entrena modelo KNN con features combinadas
4. **Predicción**: Genera recomendaciones con scores de similitud

### 3. **Database Manager** (`utils/database.py`)

**Responsabilidad**: Interfaz con MongoDB para datos de usuarios.

**Funcionalidades**:
- Conexión asíncrona a MongoDB
- Filtrado de usuarios activos
- Gestión de esquemas de datos complejos

## 🎯 Algoritmo de Recomendación

### Metodología

El sistema implementa un enfoque **content-based filtering** optimizado para compatibilidad académica:

```python
# Matrix de características combinadas
feature_matrix = [
    skills_tfidf_matrix,      # 40% del peso total
    objectives_tfidf_matrix,  # 30% del peso total  
    categorical_features      # 30% del peso total
]
```

### Proceso de Matching

1. **Input**: `user_id` del usuario solicitante
2. **Preprocessing**: Extracción de features del usuario objetivo
3. **KNN Search**: Búsqueda de k-vecinos más cercanos usando distancia coseno
4. **Scoring**: Conversión de distancia a score de similitud (1 - distancia)
5. **Ranking**: Ordenamiento por score descendente
6. **Filtering**: Exclusión de usuarios ya contactados

### Criterios de Compatibilidad

El algoritmo prioriza usuarios con:

- **Alta Coincidencia de Skills**: ≥50% de habilidades técnicas compartidas
- **Objetivos Académicos Alineados**: Proyectos o metas de estudio similares
- **Disponibilidad Temporal Compatible**: Horarios de estudio coincidentes
- **Nivel de Compromiso Similar**: Intensidad de dedicación académica

## 📊 Métricas y Performance

### Datos de Entrenamiento
- **Dataset Size**: 506 usuarios activos
- **Feature Dimensions**: ~200-300 características (dinámico según vocabulario)
- **K-Neighbors**: Configuración adaptativa (min: 5, max: dataset_size-1)

### Match Reason Intelligence

El sistema genera explicaciones automáticas del matching:

```python
match_reasons = [
    "Alta compatibilidad técnica: Python, React, Machine Learning, Docker",
    "Objetivos similares: Tesis de pregrado, Paper académico", 
    "Horario compatible: Flexible"
]
```

**Niveles de Compatibilidad**:
- **Alta compatibilidad técnica**: ≥50% skills compartidos
- **Múltiples habilidades**: ≥3 skills en común  
- **Habilidades clave**: ≥2 skills en común
- **Habilidad específica**: 1 skill compartido

## 🚀 API Endpoints

### Core Endpoints

```http
POST /retrain
# Entrena el modelo con datos actualizados

POST /recommendations  
Content-Type: application/json
{
  "user_id": "user_id_string",
  "exclude_users": ["excluded_id1", "excluded_id2"],
  "limit": 10
}

GET /health
# Status del modelo y estadísticas
```

### Response Format

```json
{
  "recommendations": [
    {
      "user_id": "68bcc19dc7e05abd45d0933e",
      "similarity_score": 0.8456,
      "match_reasons": [
        "Alta compatibilidad técnica: Backend, Mobile, DataScience",
        "Objetivos similares: Paper académico",
        "Horario compatible: Flexible"
      ],
      "profile_preview": {
        "top_skills": ["React", "Python", "MongoDB"],
        "objectives": ["Tesis de pregrado", "Paper académico"],
        "time_availability": "Flexible",
        "commitment_level": "Intensivo"
      }
    }
  ],
  "model_version": "1.0.0",
  "generated_at": "2025-09-09T17:50:58.544464"
}
```

## ⚙️ Stack Tecnológico

### Machine Learning
- **scikit-learn**: KNN, TF-IDF, preprocessing
- **NumPy**: Operaciones matriciales y vectorización
- **Pandas**: Manipulación y análisis de datos

### Backend Framework  
- **FastAPI**: API REST moderna y asíncrona
- **Pydantic**: Validación de datos y serialización
- **Motor**: Driver asíncrono para MongoDB

### Base de Datos
- **MongoDB**: NoSQL para esquemas flexibles de usuario
- **Aggregation Pipeline**: Queries complejas para filtrado de usuarios

## 🔧 Configuración y Despliegue

### Variables de Entorno
```bash
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=studysync
COLLECTION_NAME=users
MAX_SKILLS_FEATURES=100
MAX_OBJECTIVES_FEATURES=50
DEFAULT_K_NEIGHBORS=10
```

### Instalación
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📈 Optimizaciones Implementadas

### Performance
- **Vectorización eficiente** con TF-IDF sparse matrices
- **Eliminación de features irrelevantes** (edad, ubicación, semestre)
- **Caching de modelos** entrenados para respuesta rápida
- **Algoritmo KNN optimizado** con brute-force para precisión

### Calidad de Recomendaciones
- **N-gram analysis** para skills compuestos
- **Normalización inteligente** de texto (lowercase, trim)
- **Filtrado de skills vacíos** y duplicados
- **Scoring granular** con múltiples niveles de compatibilidad

### Escalabilidad
- **Arquitectura de microservicio** independiente
- **API asíncrona** con FastAPI
- **Modelo re-entrenable** sin downtime
- **Paginación y límites** configurables

## 🎯 Casos de Uso Principales

1. **Matching para Proyectos**: Usuarios buscando colaboradores con skills complementarios
2. **Grupos de Estudio**: Formación de grupos con objetivos académicos alineados  
3. **Mentorship**: Conexión entre usuarios con diferentes niveles de experiencia
4. **Research Partners**: Matching para papers académicos y tesis

## 📋 Roadmap Futuro

- **Hybrid Filtering**: Incorporar collaborative filtering
- **Deep Learning**: Embeddings neurales para mejor representación semántica
- **Real-time Updates**: Actualización incremental del modelo
- **A/B Testing**: Framework para experimentación de algoritmos
- **Feedback Loop**: Incorporar ratings de usuarios para supervised learning

---

**Desarrollado para StudySync Platform** - Sistema de recomendación académica de nueva generación basado en machine learning y procesamiento de lenguaje natural.