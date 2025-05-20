import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    df = pd.read_csv('dados.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'dados.csv' não encontrado. Verifique o caminho e o nome do arquivo.")
    exit()


df.columns = ['salario_anual', 'total_dividas', 'historico_pagamento', 'idade', 'credito_solicitado', 'elegibilidade']

def clean_historico_pagamento(value):
    try:
        return float(value)
    except (ValueError, AttributeError):
        return np.nan
 
    
df['historico_pagamento'] = df['historico_pagamento'].apply(clean_historico_pagamento)

df = df.dropna()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['salario_anual', 'total_dividas', 'credito_solicitado']:
    df = remove_outliers(df, col)

X = df[['salario_anual', 'total_dividas', 'historico_pagamento', 'idade', 'credito_solicitado']]
y = df['elegibilidade']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

modelo = KNeighborsClassifier(n_neighbors=9)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_val)
acuracia = accuracy_score(y_val, y_pred)
print(f'Acurácia na validação: {acuracia * 100:.2f}%')

joblib.dump(modelo, 'modelo_knn.joblib')
joblib.dump(scaler, 'scaler.joblib')

exemplo = [[50000, 20000, 8.5, 35, 15000]]
exemplo_df = pd.DataFrame(exemplo, columns=['salario_anual', 'total_dividas', 'historico_pagamento', 'idade', 'credito_solicitado'])
exemplo_scaled = scaler.transform(exemplo_df)
saida = modelo.predict(exemplo_scaled)
print(f'Previsão para entrada {exemplo} → Resultado: {saida[0]}')