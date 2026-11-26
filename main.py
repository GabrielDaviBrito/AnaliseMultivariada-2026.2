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


# --- BARRA LATERAL (Configurações) ---
CAMINHO_REPO = "DADOS_INEP_ANALISE_MULTIVARIADA_SIGLAS.csv"

with st.sidebar:
    st.header("Configurações da Análise")

    if os.path.exists(CAMINHO_REPO):
        df_filtrado = carregar_dados(CAMINHO_REPO)
        st.success("Base do repositório carregada.")
    else:
        st.error(f"Arquivo `{CAMINHO_REPO}` não encontrado na pasta do projeto.")
        st.stop()

    indicadores_de_trajetoria = ["TAP", "TCA", "TDA", "TCAN", "TADA"]

    variaveis_selecionadas = st.multiselect(
        "Indicadores para análise:",
        options=indicadores_de_trajetoria,
        default=indicadores_de_trajetoria,
    )

    st.divider()

    with st.expander("Sobre o Projeto", expanded=False):
        st.markdown(
            """
            **Objetivo:** Análise multivariada (Boxplots, Correlação, PCA e Agrupamento) 
            dos indicadores de trajetória dos cursos de **Psicologia** no Brasil.
            
            **Fonte:** Indicadores de Trajetória na Educação Superior (**INEP**).

            **Variáveis:**
            * **TAP:** Taxa de Permanência
            * **TCA:** Taxa de Conclusão Acumulada
            * **TDA:** Taxa de Desistência Acumulada
            * **TCAN:** Taxa de Conclusão Anual
            * **TADA:** Taxa de Desistência Anual
            """
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
    "Dados",
    "Boxplots",
    "Matriz de Correlação",
    "Componentes Principais (PCA)",
    "Agrupamento (Cluster)",
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
    st.subheader("Distribuição e Outliers por Variável (Boxplots)")

    # Gráfico geral com os boxplots lado a lado
    fig_box_geral, ax_geral = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df_numerico, palette="Set2", ax=ax_geral)
    ax_geral.set_title("Comparação Geral dos Indicadores (%)")
    ax_geral.set_ylabel("Taxa (%)")
    ax_geral.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig_box_geral)

    st.divider()
    st.markdown("**Detalhamento Individual e Identificação de Outliers**")

    # Boxplot individual de cada variável selecionada + tabela de outliers (critério IQR)
    colunas = st.columns(len(variaveis_selecionadas))
    for i, var in enumerate(variaveis_selecionadas):
        with colunas[i]:
            fig_ind, ax_ind = plt.subplots(figsize=(3.5, 4))
            sns.boxplot(y=df_analise[var], color="#66c2a5", ax=ax_ind)
            ax_ind.set_title(var)
            ax_ind.set_ylabel("Taxa (%)")
            ax_ind.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig_ind)

            # Cálculo de outliers pelo critério interquartil (IQR)
            q1 = df_analise[var].quantile(0.25)
            q3 = df_analise[var].quantile(0.75)
            iqr = q3 - q1
            outliers = df_analise[
                (df_analise[var] < q1 - 1.5 * iqr)
                | (df_analise[var] > q3 + 1.5 * iqr)
            ][["NO_IES", var]]

            if not outliers.empty:
                st.caption(f"Outliers ({len(outliers)}):")
                st.dataframe(
                    outliers.sort_values(by=var, ascending=False),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("Sem outliers.")
    st.divider()
    st.subheader("Principais Conclusões da Análise Exploratória")

    card1, card2 = st.columns(2)

    with card1:
        with st.container(border=True):
            st.markdown("#### Dispersão de Permanência e Conclusão")
            st.markdown(
                """
                * **TAP, TCA e TCAN** não apresentam *outliers*, mas exibem **elevada amplitude interquartil**.
                * A mediana da **TAP** gira em torno de **40%**, mostrando grande heterogeneidade no ritmo de retenção e conclusão entre as IES brasileiras.
                """
            )

        with st.container(border=True):
            st.markdown("#### Desistência Acumulada (TDA)")
            st.markdown(
                """
                * A maioria das IES concentra-se entre **16% e 33%** (mediana próxima de **23%**).
                * Foram identificados **5 *outliers* superiores**, com destaque para **UFSC (95,2%)**, **IMESSM (90,0%)** e **USP (71,4%)**.
                """
            )

    with card2:
        with st.container(border=True):
            st.markdown("#### Assimetria na Desistência Anual (TADA)")
            st.markdown(
                """
                * Apresenta forte **assimetria positiva** e baixa dispersão na maior parte da amostra (mediana abaixo de **2%**).
                * Possui **8 *outliers* superiores**, liderados por **IMESSM (60,0%)** e **USP (22,9%)**.
                """
            )

        with st.container(border=True):
            st.markdown("#### Instituições em Destaque")
            st.markdown(
                """
                * **IMESSM**, **USP** e **FMPFMMG** figuram simultaneamente como *outliers* tanto na evasão anual (`TADA`) quanto na acumulada (`TDA`).
                * Esse comportamento atípico explica o afastamento dessas IES nas componentes principais (PCA).
                """
            )            

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

    st.divider()
    st.subheader("Principais Conclusões da Matriz de Correlação")

    card_c1, card_c2 = st.columns(2)

    with card_c1:
        with st.container(border=True):
            st.markdown("#### Conclusão Acumulada vs. Anual (TCA × TCAN)")
            st.markdown(
                """
                * Apresentam a **correlação positiva mais forte** da base (**0,83** em Pearson e **0,84** em Spearman).
                * Instituições que acumulam maior percentual de diplomados também mantêm elevado fluxo anual de concluintes.
                """
            )

        with st.container(border=True):
            st.markdown("#### Trade-off Permanência vs. Conclusão (TAP × TCA)")
            st.markdown(
                """
                * Forte **correlação negativa** (**-0,78** em Pearson e **-0,76** em Spearman; e **-0,62 / -0,55** com **TCAN**).
                * Reflete a dinâmica complementar da coorte: conforme os estudantes concluem a graduação, reduz-se proporcionalmente o estoque de alunos ainda matriculados.
                """
            )

    with card_c2:
        with st.container(border=True):
            st.markdown("#### Evasão Acumulada vs. Anual (TDA × TADA)")
            st.markdown(
                """
                * Correlação **positiva moderada**, caindo de **0,57** (Pearson) para **0,45** (Spearman).
                * A diferença entre os coeficientes confirma o impacto dos *outliers* extremos (como IMESSM e USP) sobre a correlação linear.
                """
            )

        with st.container(border=True):
            st.markdown("#### Implicação para o PCA")
            st.markdown(
                """
                * A forte colinearidade entre **TAP, TCA e TCAN** e o bloco separado de **TDA e TADA** antecipam duas dimensões principais bem definidas.
                * Isso justifica o uso da **Análise de Componentes Principais** para sintetizar os indicadores sem redundância.
                """
            )
# ABA 4: PCA
with aba_pca:
    st.subheader("Análise de Componentes Principais (PCA)")
    pca = PCA()
    scores = pca.fit_transform(dados_padronizados)
    cargas = pca.components_.T
    componentes_pricipais_labels = [f"CP{i+1}" for i in range(len(variaveis_selecionadas))]
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)
    
    coluna_tabelaVar, coluna_cards = st.columns([1, 1])
    with coluna_tabelaVar:
        st.subheader("Variância Explicada pelos Componentes Principais")
        tabela_var = pd.DataFrame({
            'Componente': componentes_pricipais_labels,
            'Variância (%)': np.round(var_explicada, 2),
            'Acumulada (%)': np.round(var_acumulada, 2)
        })
        st.dataframe(tabela_var, width=960)
    
    with coluna_cards:
        st.subheader("Critérios de Seleção dos Componentes Principais")
        total_retentores_da_variancia = sum([1 for cp in var_explicada if cp >= 25.0])
        with st.container(border=True):
            st.markdown(
                f"""
                - Consideraremos somente os componentes principais que somarem pelo menos **25% da variância individual**. \n
                - Os componentes **CP1 e CP2** explicam, respectivamente, **50,22%** e **35,89%** da variância dos dados, totalizando **86,11%** da variância total. \n
                - Os demais componentes, **CP3, CP4, CP5**, explicam somente **13,89%** da variância total, portanto, serão desconsiderados.
                """
            )

    st.divider()
        
    coluna_tabelaLoad, coluna_vazia = st.columns([1, 1])
    with coluna_tabelaLoad:
        st.subheader("Cargas dos Componentes Principais")
        tabela_loadings = pd.DataFrame(
            cargas,
            index=[variaveis_selecionadas],
            columns=[f'CP{i+1}' for i in range(len(cargas))]
        )
        st.dataframe(tabela_loadings, width=960)
    
    st.subheader("Interpretação das Cargas e Identificação dos Componentes Principais")
    coluna_card1, coluna_card2 = st.columns([1, 1])
    with coluna_card1:
        with st.container(border=True):
            st.markdown(
                f"""
                #### Permanência x Conclusão
                - O primeiro componente principal, **CP1**, tem uma forte influência negativa de **TAP** com -0.5153 de carga enquanto demonstra uma forte influência positiva de **TCA** e **TCAN** com cargas de 0.6136 e 0.5787, respectivamente.\n
                - Podemos observar que quanto menor a permanẽncia, e maior a conclusão, maior o valor do **CP1**. Caso contrário, observamos valores mais baixos do componente.\n
                - Chegamos à conclusão de que o **CP1** representa o contraste entre **Permanência x Conclusão**.
                """
            )
    
    with coluna_card2:
        with st.container(border=True):
            st.markdown(
                f"""
                #### Permanência x Desistência
                - O segundo componente principal, **CP2**, tem uma forte influência negativa de **TAP** com -0.3996 de carga enquanto demonstra uma forte influência positiva de **TDA** e **TADA** com cargas de 0.6824 e 0.6057, respectivamente.\n
                - De forma análoga ao primeiro componente, valores menores em permanência, e maiores nas variáeis de desistência, implicam em valores maiores do **CP2**.\n
                - Dadas as características de suas cargas, podemos concluir que o **CP2** representa o contraste entre **Permanência x Desistência**.
                """
            )
    
    st.divider()
    st.subheader("Biplot da Análise de Componentes Principais - Dispersão das Instituições")
    
    grafico_biplot, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    buf = io.BytesIO()
    amv.biplot(
        scores, cargas, df_analise["NO_IES"], variaveis_selecionadas, ax=ax
    )
    grafico_biplot.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    coluna_vazia1, biplot, coluna_vazia2 = st.columns([1, 2, 1])
    with biplot:
        st.image(buf, width=960)

# ABA 5: AGRUPAMENTO
with aba_cluster:
    st.subheader("Análise de Agrupamento")
    st.info("Próximo passo: definir o método (Hierárquico ou K-Means) e montar.")