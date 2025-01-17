# Instale os pacotes necessários
# pip install pandas streamlit openpyxl xlrd

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from gower import gower_matrix
from scipy.spatial.distance import squareform

# Configurações iniciais do Streamlit
st.title("Análise de Clusters - Navegação de Compradores Online")
st.sidebar.header("Configurações")

# Carregando o dataset
st.header("Carregando os Dados")
uploaded_file = st.sidebar.file_uploader("Faça o upload do dataset (CSV, XLS ou XLSX)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Identificar o tipo de arquivo
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")  # Especificar o motor para arquivos .xls
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")  # Usar openpyxl para arquivos .xlsx

    st.write("Dados Carregados com Sucesso!")
    st.dataframe(df.head())
else:
    st.warning("Por favor, faça o upload de um arquivo válido (CSV, XLS ou XLSX).")
    st.stop()

# Análise Descritiva
st.header("Análise Descritiva")
st.write("Distribuição das variáveis e análise de valores faltantes:")
st.write(df.isna().sum())

# Criação de nova coluna para visualização
st.write("Criando grupos baseados em categorias das sessões:")
def categorize_row(row):
    categories = []
    if row['Administrative'] != 0:
        categories.append('Administrative')
    if row['Informational'] != 0:
        categories.append('Informational')
    if row['ProductRelated'] != 0:
        categories.append('ProductRelated')
    return " & ".join(categories) if categories else "None"

df['Variable_Group'] = df.apply(categorize_row, axis=1)
st.write("Frequência dos grupos criados:")
st.bar_chart(df['Variable_Group'].value_counts())

# Correlação entre variáveis numéricas
st.header("Matriz de Correlação")
if st.checkbox("Exibir matriz de correlação"):
    corr_matrix = df.select_dtypes('number').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Seleção de Variáveis para Agrupamento
st.header("Preparação dos Dados para Agrupamento")
X = df[['VisitorType', 'SpecialDay', 'Weekend', 'Month', 
        'Administrative', 'Informational', 'ProductRelated', 
        'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']]

# Tratamento de valores categóricos
X['SpecialDay'] = X['SpecialDay'].astype(float).astype('category')
X_dummies = pd.get_dummies(X)

# Escalonamento de variáveis numéricas
numerical_columns = X_dummies.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_dummies[numerical_columns] = scaler.fit_transform(X_dummies[numerical_columns])

# Matriz de Distância Gower
st.write("Calculando matriz de distância Gower...")
vars_cat = [dtype == 'bool' for dtype in X_dummies.dtypes]
distancia_gower = gower_matrix(X_dummies, cat_features=vars_cat)
gdv = squareform(distancia_gower, force='tovector')

# Clustering Hierárquico
st.header("Clusterização Hierárquica")
method = st.sidebar.selectbox("Selecione o método de clusterização", ["complete", "average", "single"])
Z = linkage(gdv, method=method)
st.write(f"Linkage realizado com o método: {method}")

# Exibição do Dendrograma
st.write("Dendrograma:")
fig, ax = plt.subplots(figsize=(12, 12))
dn = dendrogram(Z, truncate_mode='level', p=30, show_leaf_counts=True, ax=ax, color_threshold=.30)
st.pyplot(fig)

# Adicionando Grupos ao Dataset
st.sidebar.header("Definir Número de Grupos")
num_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=10, value=4)
df[f'grupo_{num_clusters}'] = fcluster(Z, num_clusters, criterion='maxclust')
st.write(f"Grupos para {num_clusters} clusters adicionados aos dados.")
st.dataframe(df.head())

# Análise dos Clusters
st.header("Análise dos Clusters")
crosstab = pd.crosstab(df.Revenue, df[f'grupo_{num_clusters}'])
percentuais = crosstab.div(crosstab.sum(axis=0), axis=1) * 100
crosstab.loc['Percentual_compra'] = percentuais.loc[True]
st.write(f"Distribuição dos Clusters para {num_clusters} grupos:")
st.dataframe(crosstab)

# Final
st.write("Obrigado por usar o aplicativo de análise!")

