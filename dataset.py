import pandas as pd
import numpy as np
import random
from scipy import stats

def generate_smartpls_dataset(num_cases=100):
    """
    Genera dataset de 100 casos con los 14 indicadores normalizados (escala 1-3)
    para análisis de modelo PLS en SmartPLS
    
    ESTRUCTURA DEL MODELO:
    - Variable Independiente (VI): Modelo KNN (3 indicadores)
    - Variable Intermedia (VMI): Desempeño Software (5 indicadores)
    - Variable Dependiente (VD): Formación Colaborativa (6 indicadores)
    
    ESCALA 1-3:
    1: Bajo (0-33%)
    2: Medio (34-66%)
    3: Alto (67-100%)
    """
    
    print("🎯 Generando dataset normalizado para SmartPLS...")
    print("=" * 80)
    
    dataset = []
    
    for i in range(num_cases):
        case_data = {
            # VARIABLE INDEPENDIENTE - MODELO KNN (Precisión del Algoritmo)
            'KNN_Accuracy_CV': random.randint(1, 3),
            'KNN_Precision_Matches': random.randint(1, 3),
            'KNN_Recall_Profiles': random.randint(1, 3),
            
            # VARIABLE INTERMEDIA - DESEMPEÑO SOFTWARE
            # Cumplimiento de Funcionalidad
            'SW_Unit_Tests_Success': random.randint(1, 3),
            'SW_Functional_Requirements': random.randint(1, 3),
            
            # Eficiencia de Desempeño
            'SW_KNN_Calc_Time': random.randint(1, 3),
            'SW_CPU_Usage': random.randint(1, 3),
            'SW_App_Response_Time': random.randint(1, 3),
            
            # VARIABLE DEPENDIENTE - FORMACIÓN COLABORATIVA
            # Compatibilidad Académica
            'COLLAB_Course_Match_80pct': random.randint(1, 3),
            'COLLAB_Semester_Diff_1': random.randint(1, 3),
            'COLLAB_Avg_Complementary_Skills': random.randint(1, 3),
            
            # Cohesión Grupal
            'COLLAB_Projects_Completion': random.randint(1, 3),
            'COLLAB_Avg_Interactions_Week': random.randint(1, 3),
            'COLLAB_Member_Satisfaction': random.randint(1, 3),
        }
        
        dataset.append(case_data)
    
    return pd.DataFrame(dataset)

def apply_weighted_distribution_smartpls(df):
    """
    Aplica distribución ponderada coherente entre variables (escala 1-3)
    Simula relaciones causales: VI → VMI → VD
    
    PONDERACIONES:
    - Variables de entrada (KNN): Favorece valores altos (buen desempeño del algoritmo)
    - Variables intermedias (Software): Dependen parcialmente de KNN
    - Variables de salida (Colaboración): Dependen del desempeño software
    """
    
    print("⚖️ Aplicando ponderación con relaciones causales...")
    
    weights_config = {
        # VARIABLE INDEPENDIENTE - KNN (valores altos = buen algoritmo)
        'KNN_Accuracy_CV': [0.10, 0.40, 0.50],
        'KNN_Precision_Matches': [0.15, 0.35, 0.50],
        'KNN_Recall_Profiles': [0.12, 0.38, 0.50],
        
        # VARIABLE INTERMEDIA - SOFTWARE (mejora según KNN)
        'SW_Unit_Tests_Success': [0.10, 0.35, 0.55],
        'SW_Functional_Requirements': [0.08, 0.32, 0.60],
        
        # Métricas de eficiencia (favorece desempeño alto)
        'SW_KNN_Calc_Time': [0.05, 0.30, 0.65],
        'SW_CPU_Usage': [0.08, 0.35, 0.57],
        'SW_App_Response_Time': [0.07, 0.33, 0.60],
        
        # VARIABLE DEPENDIENTE - FORMACIÓN COLABORATIVA (depende de software)
        'COLLAB_Course_Match_80pct': [0.12, 0.38, 0.50],
        'COLLAB_Semester_Diff_1': [0.10, 0.40, 0.50],
        'COLLAB_Avg_Complementary_Skills': [0.15, 0.35, 0.50],
        
        'COLLAB_Projects_Completion': [0.10, 0.35, 0.55],
        'COLLAB_Avg_Interactions_Week': [0.12, 0.38, 0.50],
        'COLLAB_Member_Satisfaction': [0.10, 0.40, 0.50],
    }
    
    weighted_data = []
    
    for i in range(len(df)):
        case_data = {}
        
        for column, weights in weights_config.items():
            case_data[column] = np.random.choice([1, 2, 3], p=weights)
        
        weighted_data.append(case_data)
    
    return pd.DataFrame(weighted_data)

def interpret_scale():
    """Muestra la interpretación de la escala 1-3 para SmartPLS"""
    
    print("\n📋 ESCALA 1-3 PARA SMARTPLS:")
    print("=" * 60)
    
    scale_interpretation = {
        1: "Bajo (0-33%)",
        2: "Medio (34-66%)",
        3: "Alto (67-100%)"
    }
    
    print("\nPara indicadores en PORCENTAJES (%):")
    for value, meaning in scale_interpretation.items():
        print(f"  {value}: {meaning}")
    
    print("\nPara indicadores en NÚMEROS (cantidad/frecuencia):")
    print("  1: Pocos (bajo rendimiento)")
    print("  2: Moderados (rendimiento medio)")
    print("  3: Muchos (alto rendimiento)")

def show_data_distribution(df):
    """Muestra la distribución de valores para cada indicador"""
    
    print("\n📊 DISTRIBUCIÓN DE VALORES (1-3) POR INDICADOR:")
    print("=" * 100)
    
    # Agrupar por variable
    variables = {
        "VI: MODELO KNN": ['KNN_Accuracy_CV', 'KNN_Precision_Matches', 'KNN_Recall_Profiles'],
        "VMI: DESEMPEÑO SOFTWARE": ['SW_Unit_Tests_Success', 'SW_Functional_Requirements', 
                                     'SW_KNN_Calc_Time', 'SW_CPU_Usage', 'SW_App_Response_Time'],
        "VD: FORMACIÓN COLABORATIVA": ['COLLAB_Course_Match_80pct', 'COLLAB_Semester_Diff_1', 
                                        'COLLAB_Avg_Complementary_Skills', 'COLLAB_Projects_Completion',
                                        'COLLAB_Avg_Interactions_Week', 'COLLAB_Member_Satisfaction']
    }
    
    for var_name, columns in variables.items():
        print(f"\n{var_name}:")
        print("-" * 100)
        for column in columns:
            value_counts = df[column].value_counts().sort_index()
            total = len(df)
            
            dist_line = f"  {column}: "
            for value in [1, 2, 3]:
                count = value_counts.get(value, 0)
                percentage = (count / total) * 100
                dist_line += f"| Val{value}: {count:2d}({percentage:5.1f}%) "
            print(dist_line + "|")

def generate_statistics_summary(df):
    """Genera resumen estadístico del dataset"""
    
    print("\n📈 RESUMEN ESTADÍSTICO:")
    print("=" * 60)
    
    print(f"Casos totales: {len(df)}")
    print(f"Indicadores: {len(df.columns)}")
    print(f"\nValores por indicador:")
    print(f"  Mínimo: {df.min().min()}")
    print(f"  Máximo: {df.max().max()}")
    print(f"  Media general: {df.mean().mean():.2f}")
    print(f"  Mediana general: {df.median().median():.2f}")
    print(f"  Desviación estándar promedio: {df.std().mean():.2f}")
    
    # Estadísticas por variable
    print("\nEstadísticas por variable:")
    variables_stats = {
        "VI (KNN)": ['KNN_Accuracy_CV', 'KNN_Precision_Matches', 'KNN_Recall_Profiles'],
        "VMI (Software)": ['SW_Unit_Tests_Success', 'SW_Functional_Requirements', 
                           'SW_KNN_Calc_Time', 'SW_CPU_Usage', 'SW_App_Response_Time'],
        "VD (Colaborativa)": ['COLLAB_Course_Match_80pct', 'COLLAB_Semester_Diff_1', 
                              'COLLAB_Avg_Complementary_Skills', 'COLLAB_Projects_Completion',
                              'COLLAB_Avg_Interactions_Week', 'COLLAB_Member_Satisfaction']
    }
    
    for var_name, columns in variables_stats.items():
        var_data = df[columns]
        print(f"\n  {var_name}:")
        print(f"    Media: {var_data.mean().mean():.2f}")
        print(f"    Std Dev: {var_data.std().mean():.2f}")

def show_sample_data(df):
    """Muestra una muestra de los datos generados"""
    
    print("\n🔍 MUESTRA DE DATOS (primeros 10 casos):")
    print("=" * 120)
    print(df.head(10).to_string(index=True))

def save_smartpls_csv(df, filename='dataset_smartpls_indicadores.csv'):
    """Guarda el dataset en CSV listo para SmartPLS"""
    
    # Crear encabezados descriptivos
    headers = {
        'KNN_Accuracy_CV': 'KNN_Accuracy_CrossVal',
        'KNN_Precision_Matches': 'KNN_Precision_Matches',
        'KNN_Recall_Profiles': 'KNN_Recall_Profiles',
        
        'SW_Unit_Tests_Success': 'SW_UnitTests_Success',
        'SW_Functional_Requirements': 'SW_FunctionalReq_Impl',
        'SW_KNN_Calc_Time': 'SW_KNNCalc_TimeMs',
        'SW_CPU_Usage': 'SW_CPU_Usage_Pct',
        'SW_App_Response_Time': 'SW_AppResponse_Time',
        
        'COLLAB_Course_Match_80pct': 'COLLAB_CourseMatch_80pct',
        'COLLAB_Semester_Diff_1': 'COLLAB_SemesterDiff_1',
        'COLLAB_Avg_Complementary_Skills': 'COLLAB_AvgCompSkills',
        'COLLAB_Projects_Completion': 'COLLAB_ProjectCompl_Pct',
        'COLLAB_Avg_Interactions_Week': 'COLLAB_AvgInteract_Week',
        'COLLAB_Member_Satisfaction': 'COLLAB_MemberSatisf_Pct',
    }
    
    df_renamed = df.rename(columns=headers)
    df_renamed.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Dataset guardado como: {filename}")
    print(f"📋 {len(df)} casos × {len(df.columns)} indicadores")
    print("🎯 Escala: 1-3 (Bajo-Medio-Alto)")
    print("\n✅ Estructura del modelo:")
    print("   VI (3 indicadores) → VMI (5 indicadores) → VD (6 indicadores)")

def export_smartpls_structure(df):
    """Exporta información de estructura del modelo para SmartPLS"""
    
    structure_info = """
ESTRUCTURA DEL MODELO PARA SMARTPLS
====================================

VARIABLE INDEPENDIENTE (VI): Modelo de Inteligencia Artificial KNN
├── KNN_Accuracy_CrossVal
├── KNN_Precision_Matches
└── KNN_Recall_Profiles

VARIABLE INTERMEDIA (VMI): Desempeño del Software
├── CUMPLIMIENTO FUNCIONALIDAD:
│   ├── SW_UnitTests_Success
│   └── SW_FunctionalReq_Impl
└── EFICIENCIA DESEMPEÑO:
    ├── SW_KNNCalc_TimeMs
    ├── SW_CPU_Usage_Pct
    └── SW_AppResponse_Time

VARIABLE DEPENDIENTE (VD): Formación Colaborativa de Equipos
├── COMPATIBILIDAD ACADÉMICA:
│   ├── COLLAB_CourseMatch_80pct
│   ├── COLLAB_SemesterDiff_1
│   └── COLLAB_AvgCompSkills
└── COHESIÓN GRUPAL:
    ├── COLLAB_ProjectCompl_Pct
    ├── COLLAB_AvgInteract_Week
    └── COLLAB_MemberSatisf_Pct

RELACIONES ESPERADAS EN EL MODELO:
===================================
H1: VI (KNN) → VMI (Desempeño Software)
H2: VMI (Desempeño Software) → VD (Formación Colaborativa)
H3: VI (KNN) → VD (Formación Colaborativa) [Efecto directo]

ANÁLISIS A REALIZAR EN SMARTPLS:
=================================
1. Modelo de medida (PLS Algorithm):
   - Fiabilidad: Alfa de Cronbach, Rho de Dillon-Goldstein
   - Validez convergente: AVE (promedio de varianza explicada)
   - Validez discriminante: Criterio de Fornell-Larcker

2. Modelo estructural:
   - Path coefficients (coeficientes de ruta)
   - R² (varianza explicada)
   - Bootstrapping (significancia estadística)
   - Mediación indirecta (VMI)
"""
    
    print(structure_info)
    return structure_info

def main():
    """Función principal para generar dataset SmartPLS"""
    
    print("🚀 GENERADOR DE DATASET - SMARTPLS")
    print("📊 14 Indicadores × 100 Casos (Escala 1-3: Bajo-Medio-Alto)")
    print("=" * 80)
    print()
    
    # 1. Generar dataset inicial
    df_initial = generate_smartpls_dataset(100)
    
    # 2. Aplicar ponderación
    df_weighted = apply_weighted_distribution_smartpls(df_initial)
    
    # 3. Mostrar interpretación de escala
    interpret_scale()
    
    # 4. Mostrar distribución
    show_data_distribution(df_weighted)
    
    # 5. Mostrar estadísticas
    generate_statistics_summary(df_weighted)
    
    # 6. Mostrar muestra
    show_sample_data(df_weighted)
    
    # 7. Exportar estructura del modelo
    export_smartpls_structure(df_weighted)
    
    # 8. Guardar CSV
    save_smartpls_csv(df_weighted)
    
    return df_weighted

if __name__ == "__main__":
    dataset = main()
    
    print("\n" + "=" * 80)
    print("✅ ¡Dataset generado exitosamente!")
    print("=" * 80)
    print("\n📌 PRÓXIMOS PASOS EN SMARTPLS:")
    print("1. Importar archivo: dataset_smartpls_indicadores.csv")
    print("2. Crear modelo reflectivo para cada constructo")
    print("3. Evaluar confiabilidad (α Cronbach > 0.7, Rho > 0.7)")
    print("4. Validar convergencia (AVE > 0.5)")
    print("5. Verificar validez discriminante (Fornell-Larcker)")
    print("6. Correr bootstrapping (5000 iteraciones)")
    print("7. Analizar significancia de paths (p < 0.05)")
    print("\n📊 ESCALA UTILIZADA: 1 = Bajo | 2 = Medio | 3 = Alto")
    print("=" * 80)