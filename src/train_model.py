import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def train_model():
    # 1. Cargar datos procesados
    if not os.path.exists('data/processed/cleaned_data.csv'):
        print("Error: No se encuentra el archivo procesado. Ejecuta data_processing.py primero.")
        return

    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # 2. Separar Features (X) y Target (y)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # 3. Split Estratificado (Mantiene la proporción de 20% churn en ambos sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Configuración del Modelo
    # Usamos class_weight='balanced' para manejar el desbalance de ese 20%
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12, 
        random_state=42,
        class_weight='balanced' 
    )
    
    # 5. Validación Cruzada (Demuestra robustez)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"✅ F1-Score Promedio (CV): {np.mean(cv_scores):.4f}")
    
    # 6. Entrenamiento Final
    model.fit(X_train, y_train)
    
    # 7. Evaluación Detallada
    y_pred = model.predict(X_test)
    print("\n--- Reporte de Clasificación Final ---")
    print(classification_report(y_test, y_pred))
    
    # 8. Importancia de las Variables (Para contar la historia en el Dashboard)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- Top 5 Variables Predictoras ---")
    print(importances.head(5))
    
    # 9. Guardar Modelo y Columnas (Muy importante para la consistencia)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    print("\n🚀 Modelo guardado exitosamente en /models")

if __name__ == "__main__":
    train_model()