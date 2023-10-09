from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# Tu DataFrame
data = {
    'mail': ['adolfo@gmail.com', 'jesedh@gmail.com', 'marco@gmail.com'],
    'hoobies': [['3', '4', '5'], ['1', '2', '3'], ['2', '5']],
    'estado_civil': [1, 1, 1],
    'deportes': [['2', '4', '1'], ['1', '3'], ['1', '2']],
    'edad': ['18', '20', '22'],
    'genero': ['M', 'M', 'M'],
    'ingresos': ['1000', '2000', '3000'],
    'municipio': ['1', '1', '3'],
    'pregunta_1': [['1','3'], ['2'], ['3','2']],
}

df = pd.DataFrame(data)

# Convert the lists in the 'hoobies' and 'deportes' columns to binary arrays
mlb = MultiLabelBinarizer()
df['hoobies'] = mlb.fit_transform(df['hoobies'])
df['deportes'] = mlb.fit_transform(df['deportes'])
df['edad'] = mlb.fit_transform(df['edad'])
df['genero'] = mlb.fit_transform(df['genero'])
df['ingresos'] = mlb.fit_transform(df['ingresos'])
df['municipio'] = mlb.fit_transform(df['municipio'])
df['pregunta_1'] = mlb.fit_transform(df['pregunta_1'])

# Calcular similitud coseno entre usuarios
user_similarity = cosine_similarity(df[['hoobies', 'estado_civil', 'deportes', 'edad', 'genero', 'ingresos', 'municipio', 'pregunta_1']])

# Puedes usar esto para obtener usuarios similares
similar_users = pd.DataFrame(user_similarity, columns=df['mail'], index=df['mail'])
print(similar_users)


import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo dirigido (DiGraph)
G = nx.DiGraph()

# Agregar nodos al grafo
for user in df['mail']:
    G.add_node(user)

# Agregar bordes ponderados (similitud) al grafo
for i, user1 in enumerate(df['mail']):
    for j, user2 in enumerate(df['mail']):
        if i != j:
            similarity = user_similarity[i, j]
            G.add_edge(user1, user2, weight=similarity)

# Dibujar el grafo
pos = nx.spring_layout(G)  # Puedes ajustar el diseño según tus preferencias
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, edge_color='gray', width=1, alpha=0.7)

# Agregar etiquetas de peso en los bordes
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Mostrar el gráfico
plt.show()
