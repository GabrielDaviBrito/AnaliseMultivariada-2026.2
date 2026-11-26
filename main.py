import os
import amvlib as amv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
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
    "Componentes Principais",
    "Agrupamento",
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
    st.subheader("Análise de Componentes Principais")
    pca = PCA()
    scores = pca.fit_transform(dados_padronizados)
    cargas = pca.components_.T
    componentes_pricipais_labels = [f"CP{i+1}" for i in range(len(variaveis_selecionadas))]
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Variância Explicada pelos Componentes Principais")
        tabela_var = pd.DataFrame({
            'Componente': componentes_pricipais_labels,
            'Variância (%)': np.round(var_explicada, 2),
            'Acumulada (%)': np.round(var_acumulada, 2)
        })
        st.dataframe(tabela_var, width=960)
        with st.container(border=True):
            st.markdown(
                f"""
                #### Critérios de Seleção dos Componentes Principais
                - Consideraremos somente os componentes principais que somarem pelo menos **25% da variância individual**. \n
                - Os componentes **CP1 e CP2** explicam, respectivamente, **50,22%** e **35,89%** da variância dos dados, totalizando **86,11%** da variância total. \n
                - Os demais componentes, **CP3, CP4, CP5**, explicam somente **13,89%** da variância total, portanto, serão desconsiderados.
                """
            )
    
    with col2:
        st.subheader("Gráfico da Variância Explicada pelos Componentes Principais")
        grafico_variancia_explicada, ax = plt.subplots(figsize=(10, 5.36))
        ax.bar(componentes_pricipais_labels, var_explicada, color="#66c2a5", alpha=0.7, label="Variância Individual")
        ax.plot(componentes_pricipais_labels, var_acumulada, color="darkorange", marker="o", label="Variância Acumulada")
        ax.set_xlabel("Componentes Principais")
        ax.set_ylabel("Variância (%)")
        ax.set_title("Variância Explicada pelos Componentes Principais")
        ax.axhline(25, color="red", linestyle="--", label="25% da Variância Individual")
        ax.legend()
        buf = io.BytesIO()
        grafico_variancia_explicada.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        st.image(buf, use_container_width=True)

    st.divider()
        
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Cargas dos Componentes Principais")
        tabela_loadings = pd.DataFrame(
            cargas,
            index=[variaveis_selecionadas],
            columns=[f'CP{i+1}' for i in range(len(cargas))]
        )
        st.dataframe(tabela_loadings, width=960)
        
        st.subheader("Interpretação das Cargas e Identificação dos Componentes Principais")
        with st.container(border=True):
            st.markdown(
                f"""
                #### Permanência x Conclusão
                - **CP1** demonstra forte influência negativa de **TAP** com -0.5153 de carga e forte influência positiva de **TCA** e **TCAN** com cargas de 0.6136 e 0.5787.\n
                - Interpretamos que menores escores de permanência, e maiores escores de conclusão de curso, resultam em valores maiores do **CP1**.
                - Concluímos que o **CP1** representa o contraste entre **Permanência x Conclusão**.
                """
            )
            
        with st.container(border=True):
            st.markdown(
                f"""
                #### Permanência x Desistência
                - **CP2** demonstra forte influência negativa de **TAP** com -0.3996 de carga e forte influência positiva de **TDA** e **TADA** com cargas de 0.6824 e 0.6057, respectivamente.\n
                - Interpretamos que menores escores de permanência, e maiores escores de desistência, resultam em valores maiores do **CP2**.
                - Concluímos que o **CP2** representa o contraste entre **Permanência x Desistência**.
                """
            )
    
    with col2:
        st.subheader("Biplot da Análise de Componentes Principais")
        
        grafico_biplot, ax = plt.subplots(figsize=(10, 8.5), dpi=100)
        buf = io.BytesIO()
        amv.biplot(
            scores, cargas, df_analise["NO_IES"], variaveis_selecionadas, ax=ax,
            label_fontsize=8
        )
        grafico_biplot.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        st.image(buf, use_container_width=True)

# ABA 5: AGRUPAMENTO
with aba_cluster:
    st.subheader("Análise de Agrupamento")
 
    k_grupos = st.slider("Número de grupos (k):", min_value=2, max_value=6, value=4)

    st.divider()
    st.subheader("Método Hierárquico — Vizinho Mais Distante")
 
    ligacao = linkage(dados_padronizados, method="complete", metric="euclidean")
    correlacao_cofenetica, _ = cophenet(ligacao, pdist(dados_padronizados))
    grupos_hierarquicos = fcluster(ligacao, t=k_grupos, criterion="maxclust")
    altura_corte_dendrograma = (ligacao[-(k_grupos - 1), 2] + ligacao[-k_grupos, 2]) / 2
 
    fig_dend, ax_dend = plt.subplots(figsize=(14, 5))
    dendrogram(
        ligacao,
        labels=df_analise["NO_IES"].values,
        color_threshold=altura_corte_dendrograma,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax_dend,
    )
    ax_dend.axhline(altura_corte_dendrograma, color="red", linestyle="--", label=f"Corte ({k_grupos} grupos)")
    ax_dend.set_title("Dendrograma — Ligação Completa (Distância Euclidiana)")
    ax_dend.set_ylabel("Distância")
    ax_dend.legend()
    st.pyplot(fig_dend)
 
    coluna_tabela_perfis, coluna_silhueta_hierarquica = st.columns([1, 1])
    with coluna_tabela_perfis:
        st.markdown(f"**Perfil das médias dos grupos — Correlação cofenética: {correlacao_cofenetica:.4f}**")
        perfil_grupos_hierarquicos = df_numerico.groupby(grupos_hierarquicos).mean().round(2)
        perfil_grupos_hierarquicos.insert(0, "N° de IES", pd.Series(grupos_hierarquicos).value_counts().sort_index().values)
        perfil_grupos_hierarquicos.index.name = "Grupo"
        st.dataframe(perfil_grupos_hierarquicos, use_container_width=True)
 
    with coluna_silhueta_hierarquica:
        sil_hier = [
            silhouette_score(dados_padronizados, fcluster(ligacao, t=k, criterion="maxclust"))
            for k in range(2, 7)
        ]
        fig_sil_hier, ax_silhueta_hierarquica = plt.subplots(figsize=(6, 3.5))
        ax_silhueta_hierarquica.plot(range(2, 7), sil_hier, marker="o")
        ax_silhueta_hierarquica.axvline(k_grupos, color="red", linestyle="--", alpha=0.6)
        ax_silhueta_hierarquica.set_xlabel("Número de grupos")
        ax_silhueta_hierarquica.set_ylabel("Silhueta média")
        ax_silhueta_hierarquica.set_title("Silhueta por número de grupos - Hierárquico")
        ax_silhueta_hierarquica.grid(linestyle="--", alpha=0.5)
        st.pyplot(fig_sil_hier)
 
    st.divider()
    st.subheader("Método K-Means")
 
    soma_quadrados_entre_clusters = []
    silhueta_k_medias = []
    for k in range(2, 7):
        k_medias_teste = KMeans(n_clusters=k, n_init=10, random_state=42).fit(dados_padronizados)
        soma_quadrados_entre_clusters.append(k_medias_teste.inertia_)
        silhueta_k_medias.append(silhouette_score(dados_padronizados, k_medias_teste.labels_))
 
    coluna_cotovelo, coluna_sil_km = st.columns([1, 1])
    with coluna_cotovelo:
        fig_cotovelo, ax_cotovelo = plt.subplots(figsize=(6, 3.5))
        ax_cotovelo.plot(range(2, 7), soma_quadrados_entre_clusters, marker="o")
        ax_cotovelo.axvline(k_grupos, color="red", linestyle="--", alpha=0.6)
        ax_cotovelo.set_xlabel("Número de grupos (k)")
        ax_cotovelo.set_ylabel("SQ Dentro do Cluster")
        ax_cotovelo.grid(linestyle="--", alpha=0.5)
        st.pyplot(fig_cotovelo)
 
    with coluna_sil_km:
        fig_silhueta_k_medias, ax_silhueta_k_medias = plt.subplots(figsize=(6, 3.5))
        ax_silhueta_k_medias.plot(range(2, 7), silhueta_k_medias, marker="o", color="darkorange")
        ax_silhueta_k_medias.axvline(k_grupos, color="red", linestyle="--", alpha=0.6)
        ax_silhueta_k_medias.set_xlabel("Número de grupos (k)")
        ax_silhueta_k_medias.set_ylabel("Silhueta média")
        ax_silhueta_k_medias.grid(linestyle="--", alpha=0.5)
        st.pyplot(fig_silhueta_k_medias)
 
    k_medias = KMeans(n_clusters=k_grupos, n_init=10, random_state=42).fit(dados_padronizados)
    grupos_k_medias = k_medias.labels_ + 1
 
    st.markdown(f"**Perfil das médias dos grupos — Silhueta: {silhouette_score(dados_padronizados, k_medias.labels_):.4f}**")
    perfil_k_medias = df_numerico.groupby(grupos_k_medias).mean().round(2)
    perfil_k_medias.insert(0, "N° de IES", pd.Series(grupos_k_medias).value_counts().sort_index().values)
    perfil_k_medias.index.name = "Grupo"
    st.dataframe(perfil_k_medias, use_container_width=True)
 
    st.divider()
    st.subheader("Dispersão dos Grupos nas Componentes Principais")
 
    coluna_graf_hier, coluna_graf_km = st.columns([1, 1])
    for coluna, grupos, titulo in [
        (coluna_graf_hier, grupos_hierarquicos, "Hierárquico (Ligação Completa)"),
        (coluna_graf_km, grupos_k_medias, "K-Means"),
    ]:
        with coluna:
            fig_grupos, ax_grupos = plt.subplots(figsize=(6.4, 4.8))
            sns.scatterplot(
                x=scores[:, 0], y=scores[:, 1], hue=grupos,
                palette="Set2", s=45, edgecolor="k", linewidth=0.3, ax=ax_grupos,
            )
            ax_grupos.axhline(0, color="gray", lw=0.5)
            ax_grupos.axvline(0, color="gray", lw=0.5)
            ax_grupos.set_xlabel(f"CP1 ({var_explicada[0]:.1f}%)")
            ax_grupos.set_ylabel(f"CP2 ({var_explicada[1]:.1f}%)")
            ax_grupos.set_title(titulo)
            ax_grupos.legend(title="Grupo", fontsize=8)
            ax_grupos.grid(linestyle="--", alpha=0.4)
            st.pyplot(fig_grupos)
    
    coluna_tab_hier, coluna_tab_km = st.columns([1, 1])

    with st.container(border=True):
        st.subheader("Instituições por grupo")
        tabela_grupos = df_analise[["CO_IES", "NO_IES"]].copy()
        tabela_grupos["Grupo Hierárquico"] = grupos_hierarquicos
        tabela_grupos["Grupo K-Means"] = grupos_k_medias
        st.dataframe(tabela_grupos.sort_values(["Grupo Hierárquico", "NO_IES"]),
                     hide_index=True, use_container_width=True)
 
    st.divider()
    st.subheader("Conclusões da Análise de Agrupamento (k = 4)")
 
    card_g1, card_g2 = st.columns(2)
 
    with card_g1:
        with st.container(border=True):
            st.markdown("#### Hierárquico — Vizinho Mais Distante")
            st.markdown(
                """
                * Correlação cofenética de **0,7643** e silhueta média de **0,443** com **4 grupos**.
                * **39 IES** formam o grupo de **alta permanência** (TAP média de **67,9%**) e **37 IES** o de **alta conclusão** (TCA média de **55,3%**).
                * **7 IES** (**USP, FURB, UFAL, UFSC, UEFS, UFMS e FMPFMMG**) formam o grupo de **desistência elevada** (TDA média de **60,7%**).
                * **IMESSM** fica isolada em um grupo próprio (TDA de **90,0%** e TADA de **60,0%**).
                """
            )
 
        with st.container(border=True):
            st.markdown("#### Escolha do número de grupos")
            st.markdown(
                """
                * Com **k = 2**, a silhueta do hierárquico é alta (**0,705**), mas apenas porque separa a **IMESSM** das demais 83 IES, o que não traz informação útil.
                * **k = 4** é o menor valor que separa grupos com perfis distintos (permanência, conclusão e desistência), além do outlier.
                """
            )
 
    with card_g2:
        with st.container(border=True):
            st.markdown("#### K-Médias")
            st.markdown(
                """
                * Silhueta média de **0,445** com **4 grupos**, praticamente igual à do hierárquico.
                * **42 IES** no grupo de **permanência** (TAP média de **66,2%**) e **30 IES** no de **conclusão** (TCA de **59,1%** e TCAN de **51,8%**).
                * **11 IES** no grupo de **desistência elevada** (TDA de **55,9%** e TADA de **7,5%**).
                * **IMESSM** também forma um grupo isolado.
                """
            )
 
        with st.container(border=True):
            st.markdown("#### Concordância entre os métodos")
            st.markdown(
                """
                * Índice de Rand Ajustado de **0,776**, indicando partições bastante semelhantes.
                * A diferença está na fronteira do grupo de desistência: o K-Means inclui mais IES (**11**) do que o hierárquico (**7**).
                * Os grupos se alinham aos dois eixos do PCA: **CP1** (permanência x conclusão) e **CP2** (permanência x desistência).
                """
            )
