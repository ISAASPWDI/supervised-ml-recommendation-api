"""
Modelos y esquemas de datos
"""

from .matcher import AcademicMatcher
from .schemas import (
    UserProfile, RecommendationRequest, RecommendationResponse,
    TrainingResult, HealthResponse, ModelStatsResponse
)

__all__ = [
    'AcademicMatcher',
    'UserProfile', 
    'RecommendationRequest', 
    'RecommendationResponse',
    'TrainingResult', 
    'HealthResponse', 
    'ModelStatsResponse'
]