import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def train_model():
    # Cargar datos procesados
    if not os.path.exists('data/processed/cleaned_data.csv'):
        print("Error: No se encuentra el archivo procesado. Ejecuta data_processing.py primero.")
        return

    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Separar Features (X) y Target (y)
    '''
    Features: 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    Target: 'Exited' es la variable objetivo (1 = churn, 0 = no churn)
    '''
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Split Estratificado (Mantiene la proporción de 20% churn en ambos sets)
    '''
    Dado que el dataset tiene un desbalance (20% de los clientes abandonan), 
    es fundamental realizar un split estratificado para asegurar que ambos conjuntos (entrenamiento y prueba) reflejen esta proporción.
        - Lógica de Negocio: El churn es un evento relativamente raro, y sin estratificación, 
            podríamos terminar con un set de entrenamiento que no tenga suficientes ejemplos de clientes que abandonan, 
            lo que dificultaría el aprendizaje del modelo.
        - Impacto Esperado: Al mantener la proporción de churn en ambos sets, 
            el modelo podrá aprender patrones relevantes para predecir tanto a los clientes que abandonan como a los que no, 
            mejorando así su capacidad de generalización y su rendimiento en datos no vistos.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Configuración del Modelo
    '''
     - Random Forest es una elección sólida para este tipo de problema debido a su capacidad para manejar datos tabulares, 
        capturar relaciones no lineales y su robustez frente al ruido.
     - n_estimators=150 y max_depth=12 son hiperparámetros que se han ajustado para equilibrar el sesgo y la varianza, 
        buscando un modelo que generalice bien sin sobreajustarse a los datos de entrenamiento.
     - class_weight='balanced' es crucial para abordar el desbalance en las clases, 
        asignando un mayor peso a la clase minoritaria (churn) para que el modelo preste más atención a estos casos durante el entrenamiento.
    '''
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        random_state=42,
        class_weight='balanced' 
    )
    
    # Validación Cruzada (Demuestra robustez)
    '''
    La validación cruzada con 5 folds nos permite evaluar la estabilidad del modelo y su capacidad de generalización. 
    Al promediar los F1-scores obtenidos en cada fold, podemos tener una estimación más confiable del rendimiento del modelo en datos no vistos, 
    lo que es esencial para garantizar que el modelo no esté sobreajustado a un conjunto específico de entrenamiento.
    '''
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"✅ F1-Score Promedio (CV): {np.mean(cv_scores):.4f}")
    
    # Entrenamiento Final
    model.fit(X_train, y_train)
    
    # Evaluación Detallada
    '''
    El reporte de clasificación proporciona métricas clave como precisión, recall y F1-score para cada
    clase, lo que nos permite entender mejor el rendimiento del modelo, especialmente en la clase de churn.
    La matriz de confusión nos muestra el número de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos, 
    lo que es crucial para evaluar el impacto de los errores del modelo en la toma de decisiones.
    '''
    y_pred = model.predict(X_test)
    print("\n--- Reporte de Clasificación Final ---")
    print(classification_report(y_test, y_pred))
    
    # Importancia de las Variables (Para contar la historia en el Dashboard)
    # Nos ayuda a identificar cuáles son las características más influyentes en la predicción del churn, lo que es valioso para la interpretación del modelo y para comunicar insights a los stakeholders.
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- Top 5 Variables Predictoras ---")
    print(importances.head(5))
    
    # Guardar Modelo y Columnas (Muy importante para la consistencia)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    print("\n🚀 Modelo guardado exitosamente en /models")

if __name__ == "__main__":
    train_model()