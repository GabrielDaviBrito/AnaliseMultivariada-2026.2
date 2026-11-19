import os
import amvlib as amv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Configuração inicial da página
st.set_page_config(
    page_title="Análise Multivariada - Psicologia",
    layout="wide",
)

st.title("Análise Multivariada — Trajetória do Curso de Psicologia")
st.caption("Autores: Carlos Alberto de Queiroz Júnior, Gabriel Davi Brito")


@st.cache_data
def carregar_dados(fonte):
    df = pd.read_csv(fonte, decimal=",")

    variaveis_de_interesse = [
        "Código da Instituição",
        "Nome da Instituição",
        "Taxa de Permanência - TAP",
        "Taxa de Conclusão Acumulada - TCA",
        "Taxa de Desistência Acumulada - TDA",
        "Taxa de Conclusão Anual - TCAN",
        "Taxa de Desistência Anual - TADA",
    ]

    df_filtrado = df[variaveis_de_interesse].copy()
    df_filtrado = df_filtrado.rename(
        columns={
            "Código da Instituição": "CO_IES",
            "Nome da Instituição": "NO_IES",
            "Taxa de Permanência - TAP": "TAP",
            "Taxa de Conclusão Acumulada - TCA": "TCA",
            "Taxa de Desistência Acumulada - TDA": "TDA",
            "Taxa de Conclusão Anual - TCAN": "TCAN",
            "Taxa de Desistência Anual - TADA": "TADA",
        }
    )
    return df_filtrado


# --- BARRA LATERAL (Configurações e Upload) ---
with st.sidebar:
    st.header("1. Carregamento de Dados")
    arquivo_carregado = st.file_uploader("Escolha o arquivo CSV", type="csv")

    CAMINHO_REPO = "DADOS_INEP_ANALISE_MULTIVARIADA_SIGLAS.csv"
    usar_repo = st.checkbox(
        "Usar arquivo padrão do repositório",
        value=True,
        disabled=(arquivo_carregado is not None),
    )

    if arquivo_carregado is not None:
        df_filtrado = carregar_dados(arquivo_carregado)
        st.caption("Fonte: Arquivo enviado via upload")
    elif usar_repo and os.path.exists(CAMINHO_REPO):
        df_filtrado = carregar_dados(CAMINHO_REPO)
        st.caption(f"Fonte: `{CAMINHO_REPO}` (repositório)")
    else:
        if usar_repo and not os.path.exists(CAMINHO_REPO):
            st.error(
                f"Arquivo `{CAMINHO_REPO}` não encontrado na pasta do projeto."
            )
        else:
            st.info(
                "Faça o upload de um CSV ou marque a opção do repositório acima."
            )
        st.stop()

    indicadores_de_trajetoria = ["TAP", "TCA", "TDA", "TCAN", "TADA"]

    st.header("2. Seleção de Variáveis")
    variaveis_selecionadas = st.multiselect(
        "Indicadores para análise:",
        options=indicadores_de_trajetoria,
        default=indicadores_de_trajetoria,
    )

if len(variaveis_selecionadas) <= 1:
    st.warning("Selecione pelo menos duas variáveis na barra lateral.")
    st.stop()

# Remove linhas com valores ausentes nas colunas selecionadas (mantendo alinhamento com NO_IES)
df_analise = df_filtrado[
    ["CO_IES", "NO_IES"] + variaveis_selecionadas
].dropna()
df_numerico = df_analise[variaveis_selecionadas]

# Padronização (usada no PCA e no Agrupamento)
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(df_numerico)

# --- CRIAÇÃO DAS ABAS ---
aba_dados, aba_box, aba_corr, aba_pca, aba_cluster = st.tabs([
    "📋 Dados",
    "📦 Boxplots",
    "🔥 Matriz de Correlação",
    "🧭 Componentes Principais (PCA)",
    "🧩 Agrupamento (Cluster)",
])

# ABA 1: DADOS 
with aba_dados:
    st.subheader("Base de Dados Filtrada")
    st.success(
        f"{len(df_analise)} instituições carregadas com "
        f"{len(variaveis_selecionadas)} indicadores selecionados."
    )
    st.dataframe(df_analise, use_container_width=True)

    st.subheader("Resumo Estatístico das Variáveis")
    st.dataframe(df_numerico.describe().T.round(2), use_container_width=True)

    st.subheader("Gráfico de Dispersão Par a Par (Pairplot)")
    grafico_pairwise = sns.pairplot(
        df_numerico,
        kind="scatter",
        palette="Set2",
    )
    st.pyplot(grafico_pairwise.figure)

# ABA 2: BOXPLOTS
with aba_box:
    st.subheader("Distribuição e Outliers (Boxplots)")
    st.info("Próximo passo: montar os boxplots de cada variável aqui.")

# ABA 3: CORRELAÇÃO (Código trazido do Colab)
with aba_corr:
    st.subheader("Correlograma das Variáveis")

    metodo_corr = st.radio(
        "Método de correlação:",
        options=["pearson", "spearman"],
        format_func=lambda x: (
            "Correlação de Pearson"
            if x == "pearson"
            else "Correlação de Spearman"
        ),
        horizontal=True,
    )

    corr_matriz = df_numerico.corr(method=metodo_corr)

    col_graf, col_tab = st.columns([1.4, 1])

    with col_graf:
        fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            corr_matriz,
            annot=True,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.5,
            ax=ax_corr,
        )
        titulo_metodo = "Pearson" if metodo_corr == "pearson" else "Spearman"
        ax_corr.set_title(
            f"Correlograma das Variáveis (Correlação de {titulo_metodo})"
        )
        plt.tight_layout()
        st.pyplot(fig_corr)

    with col_tab:
        st.markdown("**Valores da Matriz**")
        st.dataframe(corr_matriz.round(4), use_container_width=True)

# ABA 4: PCA
with aba_pca:
    st.subheader("Análise de Componentes Principais (PCA)")
    pca = PCA()
    scores = pca.fit_transform(dados_padronizados)
    loadings = pca.components_.T

    grafico_biplot, ax = plt.subplots(figsize=(10, 7))
    amv.biplot(
        scores, loadings, df_analise["NO_IES"], variaveis_selecionadas, ax=ax
    )
    st.pyplot(grafico_biplot)

# ABA 5: AGRUPAMENTO
with aba_cluster:
    st.subheader("Análise de Agrupamento")
    st.info("Próximo passo: definir o método (Hierárquico ou K-Means) e montar.")