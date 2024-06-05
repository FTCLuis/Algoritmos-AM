import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mplcursors  # Biblioteca para interatividade

# Leitura do arquivo CSV
data = pd.read_csv('./dados/2021_manha.csv')

# Remover outliers
# Imprimir quantidade de dados antes da remoção dos outliers
print("Quantidade de dados antes da remoção dos outliers:", len(data))

# Remover outliers (definido como valores fora de 3 desvios padrão da média)
mean_val = data['Valor'].mean()
std_val = data['Valor'].std()
data = data[(data['Valor'] >= mean_val - 3 * std_val) & (data['Valor'] <= mean_val + 3 * std_val)]

# Imprimir quantidade de dados depois da remoção dos outliers
print("Quantidade de dados depois da remoção dos outliers:", len(data))


# Selecionando as colunas relevantes
data_relevant = data[['Estacao', 'Poluente', 'Valor']]

# Agrupando por estação e poluente, e calculando a média
data_grouped = data_relevant.groupby(['Estacao', 'Poluente']).mean().reset_index()

# Normalização dos dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_grouped[['Valor']])

# Aplicação do KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # Ajuste o número de clusters conforme necessário
data_grouped['Cluster'] = kmeans.fit_predict(data_scaled)

# Obter os centróides dos clusters
cluster_centers = kmeans.cluster_centers_

# Preparar dados para plotagem
# Precisamos calcular a média da poluição para cada estação, não apenas por poluente
data_mean_by_station = data_relevant.groupby('Estacao')['Valor'].mean().reset_index()
data_mean_by_station['Cluster'] = data_grouped.groupby('Estacao')['Cluster'].first().values

# Identificar o cluster com menor poluição
cluster_means = data_mean_by_station.groupby('Cluster')['Valor'].mean()
least_polluted_cluster = cluster_means.idxmin()
least_polluted_stations = data_mean_by_station[data_mean_by_station['Cluster'] == least_polluted_cluster]['Estacao'].unique()

estacoes = data_mean_by_station['Estacao']
valores = data_mean_by_station['Valor']
clusters = data_mean_by_station['Cluster']
estacao_indices = range(len(estacoes))

# Visualizar os clusters
plt.figure(figsize=(14, 8))
scatter = plt.scatter(estacao_indices, valores, c=clusters, cmap='viridis')

# Adicionar rótulos aos centróides
for i, txt in enumerate(cluster_means.index):
    plt.annotate(f'Cluster {txt}', (cluster_centers[i, 0], cluster_means[i]), xytext=(cluster_centers[i, 0] + 0.1, cluster_means[i]), fontsize=8, ha='left')


# Plotar os centróides dos clusters
plt.scatter(cluster_centers[:, 0], cluster_means.values, c='red', marker='X', s=200, label='Centróides dos Clusters')

# Configurar o eixo X com os nomes das estações
plt.xticks(estacao_indices, estacoes, rotation=90)

# Destacar as estações menos poluídas no eixo X
for i, estacao in enumerate(estacoes):
    if estacao in least_polluted_stations:
        plt.gca().get_xticklabels()[i].set_color('red')
        plt.gca().get_xticklabels()[i].set_fontweight('bold')

plt.xlabel('Estação')
plt.ylabel('Valor Médio de Poluição')
plt.title('Clusters de Estações de Monitoramento de Poluição')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.tight_layout()  # Ajusta o layout para que os rótulos do eixo X sejam completamente visíveis

# Adicionar interatividade
cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", 
lambda sel: sel.annotation.set_text(f"{estacoes[sel.index]} \nPoluente: {data_grouped['Poluente'].loc[(data_grouped['Estacao'] == estacoes[sel.index]) & (data_grouped['Cluster'] == clusters[sel.index])].values[0]} \nValor médio: {valores[sel.index]:.2f} \nCluster: {clusters[sel.index]}"))

# Alterar o título da janela
plt.gcf().canvas.manager.set_window_title('Análise de Poluição em Estações de São Paulo')

plt.show()
