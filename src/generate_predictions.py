import pandas as pd
import joblib

def generate_dashboard_data():
    print("1. Cargando datos y modelo...")
    # Cargar los datos limpios
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # "Descongelar" el modelo y las columnas
    modelo = joblib.load('models/churn_model.pkl')
    columnas_modelo = joblib.load('models/model_columns.pkl')
    
    # Asegurarnos de que usamos exactamente las columnas que el modelo conoce
    X = df[columnas_modelo]
    
    print("2. Generando predicciones...")
    # Predecir si se va (1) o se queda (0)
    df['Prediccion_Fuga'] = modelo.predict(X)
    
    # Probabilidad de fuga
    probabilidades = modelo.predict_proba(X)
    df['Probabilidad_Fuga_%'] = (probabilidades[:, 1] * 100).round(2)
    
    # 3. Exportar para Power BI
    df.to_csv('data/processed/dashboard_data.csv', index=False)
    print("✅ ¡Listo! Datos exportados a data/processed/dashboard_data.csv")

if __name__ == "__main__":
    generate_dashboard_data()