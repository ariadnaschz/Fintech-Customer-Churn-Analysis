import pandas as pd
import joblib

def generate_dashboard_data():
    print("1. Cargando datos...")
    # Cargar los datos limpios (para el modelo)
    df = pd.read_csv('data/processed/cleaned_data.csv')
    
    # Cargar los datos crudos (solo para rescatar el ID y el Apellido)
    raw_df = pd.read_csv('data/raw/Churn_Modelling.csv')
    
    # Cargar el modelo
    modelo = joblib.load('models/churn_model.pkl')
    columnas_modelo = joblib.load('models/model_columns.pkl')
    
    X = df[columnas_modelo]
    
    print("2. Generando predicciones...")
    df['Prediccion_Fuga'] = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    df['Probabilidad_Fuga_%'] = (probabilidades[:, 1] * 100).round(2)
    
    # 3. Pegar el ID y el Apellido al principio del DataFrame
    df.insert(0, 'CustomerId', raw_df['CustomerId'])
    df.insert(1, 'Surname', raw_df['Surname'])
    
    # 4. Exportar
    df.to_csv('data/processed/dashboard_data.csv', index=False)
    print("✅ ¡Listo! Datos exportados con CustomerId y Surname incluidos.")

if __name__ == "__main__":
    generate_dashboard_data()