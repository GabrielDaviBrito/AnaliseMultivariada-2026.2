# AnaliseMultivariada-2026.2 
Projeto avaliativo da disciplina de Análise Multivariada.
Dashboard interativo desenvolvido em **Python** e **Streamlit** para análise estatística multivariada dos indicadores de trajetória estudantil dos cursos de graduação em **Psicologia** nas Instituições de Ensino Superior (IES) brasileiras.

**Autores:**
* Carlos Alberto de Queiroz Júnior
* Gabriel Davi Brito

---

## Sobre os Dados

Os dados utilizados (`DADOS_INEP_ANALISE_MULTIVARIADA_SIGLAS.csv`) são provenientes dos **Indicadores de Trajetória na Educação Superior** divulgados pelo **INEP**, filtrados para cursos de Psicologia, contemplando cinco variáveis quantitativas principais:

* **TAP:** Taxa de Permanência (%)
* **TCA:** Taxa de Conclusão Acumulada (%)
* **TDA:** Taxa de Desistência Acumulada (%)
* **TCAN:** Taxa de Conclusão Anual (%)
* **TADA:** Taxa de Desistência Anual (%)

---

## Pré-requisitos e Instalação

Certifique-se de ter o **Python 3.10+** instalado em sua máquina.

1. Clone este repositório:
   ```bash
   git clone [https://github.com/GabrielDaviBrito/AnaliseMultivariada-2026.2.git](https://github.com/GabrielDaviBrito/AnaliseMultivariada-2026.2.git)
   cd AnaliseMultivariada-2026.2
   ```

2. Instale as bibliotecas necessárias utilizando o arquivo `requirements.txt` (caso utilize os scripts automáticos abaixo, este passo já é executado automaticamente):
   ```bash
   pip install -r requirements.txt
   ```

---

## Como Executar a Aplicação

Você pode iniciar o dashboard de acordo com o seu sistema operacional:

### Opção 1: Execução automática no Windows (`rodar.bat`)
Dê um duplo clique no arquivo **`rodar.bat`** na pasta raiz do projeto (ou execute `.\rodar.bat` no terminal). O script atualiza o `pip`, instala as dependências do `requirements.txt` e inicia o Streamlit automaticamente.

### Opção 2: Execução automática no macOS / Linux (`rodar.sh`)
Na pasta raiz do projeto, conceda permissão de execução (apenas na primeira vez) e rode o script:
```bash
chmod +x rodar.sh
./rodar.sh
```
O script instala as dependências via `python3` e abre a aplicação no navegador.

### Opção 3: Via Terminal (manual)
```bash
streamlit run main.py
```

### Opção 4: Pelo VS Code (sem terminal)
Com a extensão **Python** instalada no VS Code, abra o projeto e pressione **`F5`** (utilizando a configuração `.vscode/launch.json`).

Após a inicialização, a aplicação abrirá automaticamente no navegador em `http://localhost:8501`.

---

## Descrição das Funcionalidades

O dashboard é organizado em uma barra lateral de controle e cinco abas principais:

* **Barra Lateral (Configurações):** Carrega automaticamente a base oficial do repositório (`DADOS_INEP_ANALISE_MULTIVARIADA_SIGLAS.csv`), permite selecionar quais indicadores (`TAP`, `TCA`, `TDA`, `TCAN`, `TADA`) entram nos cálculos e disponibiliza o resumo contextual na seção recolhível *Sobre o Projeto*.
* **Dados:** Exibe a tabela filtrada das instituições, o resumo estatístico descritivo e o gráfico de dispersão par a par.
* **Boxplots:** Apresenta o boxplot comparativo geral e os boxplots individuais de cada indicador, identificando automaticamente as instituições *outliers* pelo critério interquartil, acompanhados de cards interpretativos.
* **Matriz de Correlação:** Gera o correlograma e a tabela numérica exata, permitindo alternar dinamicamente entre a Correlação de Pearson e a Correlação de Spearman, com cards de interpretação.
* **Componentes Principais (PCA):** Realiza a padronização dos dados (Z-score) e apresenta a variância explicada, cargas fatoriais, gráfico de dispersão dos escores e o *Biplot* das instituições e variáveis.
* **Agrupamento (Cluster):** Aplica técnicas de análise de agrupamento sobre os indicadores padronizados para segmentar as instituições com perfis semelhantes de trajetória acadêmica, exibindo a visualização e a caracterização dos grupos formados.

---

## Exemplos de Uso

1. **Explorar a distribuição e identificar casos atípicos:**
   * Ao abrir a aplicação, acesse a aba **Boxplots** para visualizar o comportamento geral das taxas. Role até o detalhamento individual para consultar a lista de instituições classificadas como *outliers* em `TDA` e `TADA` (como IMESSM, USP e UFSC) e leia a síntese nos cards inferiores.
2. **Comparar correlação linear vs. monotônica:**
   * Na aba **Matriz de Correlação**, alterne o seletor superior entre *Correlação de Pearson* e *Correlação de Spearman* para observar o impacto dos *outliers* na relação entre `TDA` e `TADA`  enquanto a relação `TCA × TCAN` permanece estável.
3. **Analisar componentes principais e grupos de instituições:**
   * Navegue pelas abas **Componentes Principais (PCA)** e **Agrupamento (Cluster)** para identificar como as universidades se distribuem nos eixos de conclusão e desistência e quais grupos homogêneos são formados.
4. **Filtrar subconjuntos de indicadores:**
   * Na barra lateral esquerda, remova uma variável (por exemplo, `TADA`) no campo *Indicadores para análise*. Todas as tabelas, matrizes, componentes principais e agrupamentos são recalculados instantaneamente apenas com as variáveis selecionadas.