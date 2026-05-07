import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def transform_data(df):
    # Eliminar columnas que no aportan valor predictivo
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Feature Engineering - Crear nuevas características a partir de las existentes
    '''
    Justificación de las Nuevas Variables (Features)

    Para mejorar la capacidad predictiva del modelo, se implementaron las siguientes métricas:
    - Is_Zero_Balance: 
        -Lógica de Negocio: Identifica binariamente a clientes con saldo 0
        -Impacto Esperado: Capturar el comportamiento específico del segmento inactivo (cuentas zombie)
    - Balance_Salary_Ratio: 
        -Lógica de Negocio: Proporción del saldo actual respecto al salario estimado
        -Impacto Esperado: Determinar si el banco es la entidad financiera principal del cliente o solo una cuenta secundaria
    - Tenure_Age_Ratio: 
        -Lógica de Negocio: Años de permanencia divididos por la edad del cliente
        -Impacto Esperado: Medir la lealtad relativa. Un ratio alto indica un cliente que ha pasado gran parte de su vida adulta con la institución

    '''
    df['Is_Zero_Balance'] = (df['Balance'] == 0).astype(int)
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Tenure_Age_Ratio'] = df['Tenure'] / df['Age']
    
    # Convertimos Gender a binario y Geography a variables Dummy 
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    
    return df

if __name__ == "__main__":
    # Prueba local
    raw_data = load_data('data/raw/Churn_Modelling.csv')
    processed_data = transform_data(raw_data)
    processed_data.to_csv('data/processed/cleaned_data.csv', index=False)
    print("Transformación completada y guardada en data/processed/")