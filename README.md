# Modelo de Classificação KNN para Elegibilidade de Crédito

## Tipo de modelo
- KNN (K-Nearest Neighbors)
- Número de vizinhos (K): 9

## Variáveis utilizadas no modelo (ordem no vetor de entrada)
1. salário anual (em dólares)
2. total de dívidas (em dólares)
3. histórico de pagamento (proporção entre 0 e 1)
4. idade
5. crédito solicitado (em dólares)

## Exemplo de previsão
- Entrada: [50000, 20000, 0.85, 35, 15000]
- Saída: [2]  
> Categoria 2 = Elegível com análise

## Normalização
As variáveis foram normalizadas utilizando `StandardScaler` da biblioteca `sklearn.preprocessing`.

## Outliers
Foram removidos outliers com base na regra do IQR.

## Mapeamento das classes
- 1: Não Elegível
- 2: Elegível com análise
- 3: Elegível
