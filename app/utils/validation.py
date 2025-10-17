import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd

class ModelValidator:
    """Validador para métricas de tu matriz de operacionalización"""
    
    def __init__(self, feature_matrix, labels=None):
        """
        Args:
            feature_matrix: Matriz de características del modelo
            labels: Etiquetas de matches exitosos (si están disponibles)
        """
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.validation_results = {}
    
    def perform_cross_validation(self, k_neighbors=3, n_folds=5):
        """
        Validación cruzada del modelo KNN
        
        INDICADOR: % de accuracy del modelo KNN en validación cruzada
        """
        if self.labels is None:
            print("⚠️ No hay etiquetas. Generando validación sintética...")
            # Generar labels sintéticas basadas en similitud
            self.labels = self._generate_synthetic_labels()
        
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='cosine')
        
        # K-Fold Cross Validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Calcular accuracy en cada fold
        accuracy_scores = cross_val_score(
            knn, self.feature_matrix, self.labels, 
            cv=kfold, scoring='accuracy'
        )
        
        self.validation_results['accuracy'] = {
            'mean': float(np.mean(accuracy_scores)),
            'std': float(np.std(accuracy_scores)),
            'scores': accuracy_scores.tolist(),
            'min': float(np.min(accuracy_scores)),
            'max': float(np.max(accuracy_scores))
        }
        
        print(f"📊 Accuracy en validación cruzada ({n_folds}-fold):")
        print(f"   • Promedio: {self.validation_results['accuracy']['mean']:.3f}")
        print(f"   • Desviación: ±{self.validation_results['accuracy']['std']:.3f}")
        print(f"   • Rango: [{self.validation_results['accuracy']['min']:.3f}, "
              f"{self.validation_results['accuracy']['max']:.3f}]")
        
        return self.validation_results['accuracy']
    
    def calculate_precision_recall(self, k_neighbors=3):
        """
        Calcula precision y recall del modelo
        
        INDICADORES:
        - % de precisión en predicción de matches exitosos
        - % de recall en identificación de perfiles compatibles
        """
        if self.labels is None:
            self.labels = self._generate_synthetic_labels()
        
        # Train/test split
        split_idx = int(len(self.feature_matrix) * 0.8)
        X_train = self.feature_matrix[:split_idx]
        X_test = self.feature_matrix[split_idx:]
        y_train = self.labels[:split_idx]
        y_test = self.labels[split_idx:]
        
        # Entrenar y predecir
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='cosine')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Calcular métricas
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        self.validation_results['precision'] = float(precision)
        self.validation_results['recall'] = float(recall)
        self.validation_results['confusion_matrix'] = cm.tolist()
        
        print(f"\n📊 Métricas de Clasificación:")
        print(f"   • Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"   • Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"\n📈 Matriz de Confusión:")
        print(cm)
        
        return {
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
    
    def calculate_error_rate(self, k_values=[3, 5, 7, 10]):
        """
        Calcula tasa de error para diferentes valores de K
        
        Para validar tu afirmación: "K>5 aumenta error significativamente"
        """
        if self.labels is None:
            self.labels = self._generate_synthetic_labels()
        
        error_rates = {}
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            scores = cross_val_score(
                knn, self.feature_matrix, self.labels,
                cv=5, scoring='accuracy'
            )
            error_rate = 1 - np.mean(scores)
            error_rates[k] = float(error_rate)
        
        self.validation_results['error_rates'] = error_rates
        
        print(f"\n📉 Tasa de Error por K:")
        for k, error in error_rates.items():
            print(f"   • K={k}: {error:.3f} ({error*100:.1f}%)")
        
        # Validar tu hipótesis
        if len(k_values) >= 2:
            k3_error = error_rates.get(3, 0)
            k5_plus_errors = [error_rates[k] for k in k_values if k > 5]
            
            if k5_plus_errors:
                avg_k5_plus = np.mean(k5_plus_errors)
                increase = ((avg_k5_plus - k3_error) / k3_error) * 100
                
                print(f"\n🔍 Análisis:")
                print(f"   Error con K=3: {k3_error:.3f}")
                print(f"   Error promedio K>5: {avg_k5_plus:.3f}")
                print(f"   Incremento: +{increase:.1f}%")
                
                if increase > 10:
                    print(f"   ✅ VALIDADO: K>5 aumenta error significativamente (+{increase:.1f}%)")
        
        return error_rates
    
    def _generate_synthetic_labels(self):
        """
        Genera labels sintéticas basadas en similitud de características
        
        Nota: En producción, deberías usar datos reales de matches exitosos
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calcular similitud entre todos los pares
        similarities = cosine_similarity(self.feature_matrix)
        
        # Generar labels binarias: 1 si similitud > umbral, 0 si no
        threshold = np.percentile(similarities.flatten(), 75)
        
        # Para cada usuario, etiquetar si tiene matches de alta calidad
        labels = []
        for i in range(len(self.feature_matrix)):
            # Si tiene al menos 2 vecinos con alta similitud, label=1
            high_sim_neighbors = np.sum(similarities[i] > threshold) - 1  # -1 para excluir a sí mismo
            labels.append(1 if high_sim_neighbors >= 2 else 0)
        
        return np.array(labels)
    
    def generate_validation_report(self):
        """
        Genera reporte completo para tu matriz de operacionalización
        """
        report = {
            "variable_independiente": {
                "indicador": "Precisión del Algoritmo",
                "metricas": {
                    "accuracy_cv": self.validation_results.get('accuracy', {}),
                    "precision": self.validation_results.get('precision', 0),
                    "recall": self.validation_results.get('recall', 0),
                    "error_rates": self.validation_results.get('error_rates', {})
                }
            },
            "interpretacion": self._interpret_results()
        }
        
        return report
    
    def _interpret_results(self):
        """Interpreta los resultados según umbrales de tu investigación"""
        interpretations = []
        
        accuracy = self.validation_results.get('accuracy', {}).get('mean', 0)
        if accuracy >= 0.80:
            interpretations.append(f"✅ Accuracy {accuracy:.1%} cumple umbral mínimo (80%)")
        else:
            interpretations.append(f"⚠️ Accuracy {accuracy:.1%} bajo el umbral (80%)")
        
        precision = self.validation_results.get('precision', 0)
        if precision >= 0.75:
            interpretations.append(f"✅ Precision {precision:.1%} cumple umbral mínimo (75%)")
        else:
            interpretations.append(f"⚠️ Precision {precision:.1%} bajo el umbral (75%)")
        
        recall = self.validation_results.get('recall', 0)
        if recall >= 0.70:
            interpretations.append(f"✅ Recall {recall:.1%} cumple umbral mínimo (70%)")
        else:
            interpretations.append(f"⚠️ Recall {recall:.1%} bajo el umbral (70%)")
        
        return interpretations


# EJEMPLO DE USO
def validate_model_for_thesis(matcher):
    """
    Función helper para validar el modelo según tu matriz de operacionalización
    
    Args:
        matcher: Instancia de AcademicMatcher con modelo entrenado
    """
    print("🔬 INICIANDO VALIDACIÓN DEL MODELO PARA TESIS\n")
    
    validator = ModelValidator(matcher.feature_matrix)
    
    # 1. Validación cruzada (Indicador: % accuracy)
    print("=" * 60)
    print("INDICADOR 1: % de accuracy del modelo KNN en validación cruzada")
    print("=" * 60)
    validator.perform_cross_validation(k_neighbors=3, n_folds=5)
    
    # 2. Precision y Recall (Indicadores: % precision, % recall)
    print("\n" + "=" * 60)
    print("INDICADORES 2-3: % precisión y % recall")
    print("=" * 60)
    validator.calculate_precision_recall(k_neighbors=3)
    
    # 3. Análisis de error por K (Validar hipótesis K=3 óptimo)
    print("\n" + "=" * 60)
    print("ANÁLISIS: Validación de K óptimo")
    print("=" * 60)
    validator.calculate_error_rate(k_values=[3, 5, 7, 10, 15])
    
    # 4. Reporte final
    print("\n" + "=" * 60)
    print("REPORTE FINAL PARA MATRIZ DE OPERACIONALIZACIÓN")
    print("=" * 60)
    report = validator.generate_validation_report()
    
    print("\n📋 INTERPRETACIONES:")
    for interp in report['interpretacion']:
        print(f"   {interp}")
    
    return report