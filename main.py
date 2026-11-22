import os
import amvlib as amv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import io

# Dicionário de constantes e variáveis
# # Constantes
# CAMINHO_REPO: Caminho do arquivo CSV padrão do repositório

# # Variáveis
# arquivo_carregado:          Arquivo CSV carregado pelo usuário
# dados_padronizados:         Dados numéricos padronizados
# df_analise:                 DataFrame com as observações e variáveis selecionadas, sem valores ausentes
# df_filtrado:                DataFrame com as variáveis selecionadas e renomeadas
# df_numerico:                DataFrame apenas com as variáveis numéricas selecionadas
# cargas:                     Cargas do PCA (componentes principais)
# corr_matriz:                Matriz de correlação das variáveis selecionadas
# indicadores_de_trajetoria:  Lista de indicadores de trajetória disponíveis para análise
# variaveis_selecionadas:     Lista de variáveis selecionadas pelo usuário para análise


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
    
    CAMINHO_REPO = "AnaliseMultivariada-2026.2/DADOS_INEP_ANALISE_MULTIVARIADA_SIGLAS.csv"
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

# ABA 4: PCA (Código trazido do Colab)
with aba_pca:
    # Preparação dos dados para PCA
    pca = PCA()
    scores = pca.fit_transform(dados_padronizados)
    cargas = pca.components_.T
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)
    
    tabela_var = pd.DataFrame({
        'Componente': [f'CP{i+1}' for i in range(len(var_explicada))],
        'Variância (%)': np.round(var_explicada, 2),
        'Acumulada (%)': np.round(var_acumulada, 2)
    })
    tabela_loadings = pd.DataFrame(
        cargas,
        index=[indicadores_de_trajetoria],
        columns=[f'CP{i+1}' for i in range(len(cargas))]
    )
    
    st.subheader("Análise de Componentes Principais (PCA)")
    st.markdown(
        """
        Iniciamos nossos estudos com um conjunto de 84 observações de 5 indicadores de trajetória de cursos de Psicologia, administrados por instituições de ensino público de todo o país: TAP, a taxa de permanência de discentes no curso; TCA, a taxa de conclusão acumulada do curso; TDA, a taxa de desistência acumulada; TCAN, a taxa de conclusão anual do curso; e TADA, a taxa de desistência anual. Nosso objetivo nessa seção é encontrar relações e identificar padrões entre essas variáveis, a fim de podermos descrever a trajetória dos dados de forma mais concisa e informativa. Por meio da *Análise de Componentes Principais* conseguimos reduzir a dimensionalidade do conjunto de dados, e ao mesmo tempo, manter a maior parte da variabilidade presente nas variáveis originais. A seguir, apresentamos a variância explicada por cada componente principal e as *cargas* que indicam a contribuição de cada variável para os componentes.
        """
    )
    
    st.markdown("## Variância Explicada pelos Componentes Principais")
    st.markdown(
        """
        Seguem as variâncias explicadas por cada componente principal, calculadas a partir da matriz de correlação das variáveis selecionadas, bem como a variância acumulada.
        """
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(tabela_var, width=960)
    st.markdown(
        """
        De acordo com os dados da tabela acima, os dois primeiros componentes principais, CP1 e CP2, explicam juntos aproximadamente 80% da variância total dos dados. Logo, podemos descriminar os componentes CP3, CP4 e CP5, que juntos explicam uma parcela muito menor, e negligenciável, da variância total.
        
        Definidas as componentes do modelo, podemos analisar as *cargas* de cada variável em cada componente, que indicam a contribuição de cada indicador de trajetória para a formação dos componentes principais.
        """
    )
    
    st.markdown("## Cargas dos Componentes Principais")
    st.markdown(
            """
            A seguir, estão as cargas de cada variável nos componentes principais.
            """
        )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(tabela_loadings, width=960)
    st.markdown(
    """
    Podemos observar que as cargas das variáveis TAP, TCA e TCAN são, em módulo, as de maior grandeza na formação do primeiro componente principal, sendo elas, respectivamente, -0.5153, 0.6136 e 0.5787. Portanto, podemos interpretar o primeiro componente como um constraste entre a taxa de permanência e as taxas de conclusão --- tanto acumulada quanto anual --- desconsiderando as variáveis de desistência, que possuem cargas de grandeza insignificante. E então definimos o primeiro componente como um escore de "Conclusão de Curso vs Permanência".
    """
    )
    
    grafico_biplot, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    buf = io.BytesIO()
    amv.biplot(
        scores, cargas, df_analise["NO_IES"], variaveis_selecionadas, ax=ax
    )
    grafico_biplot.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(buf, width=960)

# ABA 5: AGRUPAMENTO
with aba_cluster:
    st.subheader("Análise de Agrupamento")
    st.info("Próximo passo: definir o método (Hierárquico ou K-Means) e montar.")