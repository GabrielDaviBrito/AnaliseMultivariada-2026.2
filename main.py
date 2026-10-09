import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import amvlib as amv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.write("# Análise Multivariada")
st.write("Autores: Carlos Alberto de Queiroz Júnior, Gabriel Davi Brito")

# Caixa de upload de arquivo CSV
arquivo_carregado = st.file_uploader("Escolha um arquivo CSV", type="csv")
# Verifica se o arquivo foi carregado
if arquivo_carregado is None:
    st.stop()
# Lê o arquivo CSV em um DataFrame do pandas, corrigindo o separador decimal para vírgula
df = pd.read_csv(arquivo_carregado, decimal=',')
# Filtra as variáveis de interesse e renomeia as colunas para facilitar a análise
variaveis_de_interesse = [
    'Código da Instituição', 'Nome da Instituição','Taxa de Permanência - TAP', 'Taxa de Conclusão Acumulada - TCA',
    'Taxa de Desistência Acumulada - TDA', 'Taxa de Conclusão Anual - TCAN',
    'Taxa de Desistência Anual - TADA'
]
indicadores_de_trajetoria = ['TAP','TCA','TDA','TCAN','TADA']
df_filtrado = df[variaveis_de_interesse].copy()
df_filtrado = df_filtrado.rename(
    columns = {
        'Código da Instituição':'CO_IES',
        'Nome da Instituição':'NO_IES',
        'Taxa de Permanência - TAP':'TAP',
        'Taxa de Conclusão Acumulada - TCA':'TCA',
        'Taxa de Desistência Acumulada - TDA':'TDA',
        'Taxa de Conclusão Anual - TCAN':'TCAN',
        'Taxa de Desistência Anual - TADA':'TADA'
    }
)
st.success(f"{len(df_filtrado)} observações de {len(df_filtrado.columns)} variáveis foram carregadas")
st.dataframe(df_filtrado)

# Seleção de variáveis para análise
variaveis_disponiveis = df_filtrado.columns.tolist()
with st.form("Variáveis para Análise"):
    variaveis_selecionadas = st.multiselect("Selecione as variáveis para análise", variaveis_disponiveis, default=['TAP', 'TCA', 'TDA', 'TCAN', 'TADA'])
    aplicar_btn = st.form_submit_button("Aplicar")
    
if (len(variaveis_selecionadas) <= 1):
    st.warning("Selecione pelo menos duas variáveis para realizar a análise.")
    st.stop()
else:
    df_selecionado = df_filtrado[variaveis_selecionadas].dropna()
    st.write(f"Analisando {len(df_selecionado)} observações de {len(variaveis_selecionadas)} variáveis")
    
    # Padronização dos dados
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_selecionado)
    
    # PCA
    pca = PCA()
    scores = pca.fit_transform(dados_padronizados)
    loadings = pca.components_.T
    

### COMEÇO DA VISUALIZAÇÃO ###

# Pair Plot
plt.figure(figsize=(10, 8))
grafico_pairwise = sns.pairplot(
    df_filtrado[variaveis_selecionadas],
    kind="scatter",
    markers=["o","s","D"],
    palette="Set2"
)
st.pyplot(grafico_pairwise)


# Biplot
grafico_biplot, ax = plt.subplots(figsize=(10, 8))
amv.biplot(scores, loadings, df_filtrado.iloc[:, 1], variaveis_selecionadas, ax=ax)
st.pyplot(grafico_biplot)