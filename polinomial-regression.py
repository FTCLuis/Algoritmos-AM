import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Carregar os dados
data = pd.read_csv('./dados/2021_tarde.csv')

# Filtrar dados para o poluente MP10
data_mp10 = data[data['Poluente'] == 'NO2']

# Mostrar a quantidade total de dados analisados
print("Quantidade total de dados analisados:", len(data_mp10))

# Remover outliers (definido como valores fora de 3 desvios padrão da média)
mean_val = data_mp10['Valor'].mean()
std_val = data_mp10['Valor'].std()
data_mp10 = data_mp10[(data_mp10['Valor'] >= mean_val - 3 * std_val) & (data_mp10['Valor'] <= mean_val + 3 * std_val)]

# Mostrar a quantidade total de dados analisados após a remoção de outliers
print("Quantidade total de dados analisados após remoção de outliers:", len(data_mp10))

# Verificar a distribuição dos valores de "Valor"
plt.figure(figsize=(10, 6))
hist, bins, _ = plt.hist(data_mp10['Valor'], bins=50, edgecolor='k', width=2.0)
plt.title('Distribuição dos Valores')
plt.xlabel('Valor')
plt.ylabel('Frequência')

# Adicionando rótulos aos intervalos no eixo x
bin_centers = 0.5 * (bins[:-1] + bins[1:])
for count, (x, y) in enumerate(zip(bin_centers, hist)):
    if y > 0:
        plt.text(x, y, f'{int(x)}-{int(bins[count+1])}', ha='center', va='bottom', rotation=45)

# Alterar o título da janela
plt.gcf().canvas.manager.set_window_title('Análise de Poluição em Estações de São Paulo')

plt.show()

# Definir as features (X) e o alvo (y)
# Vamos usar o 'Estacao' como uma característica fictícia para fins de exemplo
X = data_mp10[['Estacao']]
y = data_mp10['Valor']

# Transformar a variável categórica 'Estacao' para numérica (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Adicionar uma coluna de índice para representar a ordem dos dados
X['Index'] = np.arange(len(X))

# Transformar os dados para incluir termos polinomiais
poly = PolynomialFeatures(degree=2)  # Grau 2 para termos quadráticos
X_poly = poly.fit_transform(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ajustar o modelo de Regressão Linear aos dados polinomiais
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcular o Erro Médio Quadrático (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Erro Médio Quadrático (MSE):", mse)

# Calcular o Erro Médio Absoluto (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Erro Médio Absoluto (MAE):", mae)

# Preparar a nova estação para previsão
# Certificar que as colunas estão em ordem correta e presentes
nova_estacao = pd.DataFrame({'Estacao': ['Nome_da_nova_estacao']})
nova_estacao_encoded = pd.get_dummies(nova_estacao, drop_first=True)

# Adicionar colunas faltantes com valor 0
for col in X.columns:
    if col not in nova_estacao_encoded.columns:
        nova_estacao_encoded[col] = 0

# Garantir que a nova estação tenha as colunas na mesma ordem que X
nova_estacao_encoded = nova_estacao_encoded[X.columns]

# Transformar a nova estação usando PolynomialFeatures
nova_estacao_poly = poly.transform(nova_estacao_encoded)

# Fazer a previsão para a nova estação
previsao_nova_estacao = model.predict(nova_estacao_poly)
print("Previsão de poluição para a nova estação:", previsao_nova_estacao)
