# 🚗 Prevendo Valores de Veículos

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-007ACC?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

![Tipo](https://img.shields.io/badge/Problema-Regressão-success?style=flat-square)
![Modelo Final](https://img.shields.io/badge/Modelo_Final-Gradient_Boosting-orange?style=flat-square)
![R²](https://img.shields.io/badge/R²-0.9049-brightgreen?style=flat-square)
![MAE](https://img.shields.io/badge/MAE-0.7260-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=flat-square)

> Projeto de Machine Learning supervisionado (regressão) para estimar o preço de venda de veículos usados, com comparação sistemática entre **4 algoritmos** e **3 estratégias de pré-processamento de escala**.

---

## 📑 Índice

- [Definição do Problema de Negócio](#-definição-do-problema-de-negócio)
- [Dataset](#-dataset)
- [Stack Tecnológica](#-stack-tecnológica)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Pipeline do Projeto](#-pipeline-do-projeto)
- [1. Análise Exploratória (EDA)](#1-análise-exploratória-eda)
- [2. Pré-Processamento](#2-pré-processamento)
- [3. Construção dos Modelos](#3-construção-dos-modelos)
- [4. Métricas de Avaliação](#4-métricas-de-avaliação)
- [5. Resultados e Comparação dos Modelos](#5-resultados-e-comparação-dos-modelos)
- [6. Experimento de Escala: Original vs. Normalização vs. Padronização](#6-experimento-de-escala-original-vs-normalização-vs-padronização)
- [7. Insights Principais](#-7-insights-principais)
- [8. Modelo Final e Persistência](#-8-modelo-final-e-persistência)
- [Pontos de Atenção e Melhorias Futuras](#-pontos-de-atenção-e-melhorias-futuras)
- [Como Executar](#-como-executar)

---

## 🎯 Definição do Problema de Negócio

Construir um modelo de Machine Learning capaz de **prever o valor de venda de veículos** a partir de suas características (idade, quilometragem, tipo de combustível, câmbio, tipo de vendedor e preço de tabela).

| Item | Definição |
| :--- | :--- |
| **Tipo de problema** | Aprendizado supervisionado — **Regressão** |
| **Variável alvo (target)** | `Selling_Price` (valor numérico contínuo) |
| **Variáveis preditoras** | `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`, `Age` |
| **Critério de sucesso** | Menor erro de previsão (MAE / MAPE) e maior poder explicativo (R²) |

**Aplicação prática:** precificação automática em marketplaces de usados, avaliação de estoque por concessionárias e suporte à negociação (detectar anúncios acima/abaixo do valor justo de mercado).

---

## 📊 Dataset

**Fonte:** [Kaggle — Car Data](https://www.kaggle.com/datasets/athirags/car-data)

**Dimensões originais:** `301 registros × 9 colunas` · **Valores nulos:** `0`

| Variável | Descrição | Tipo |
| :--- | :--- | :--- |
| `Car_Name` | Nome/modelo do carro (98 valores únicos) | Categórica |
| `Year` | Ano de fabricação/modelo (2003–2018) | Numérica |
| `Selling_Price` | **🎯 TARGET** — preço pelo qual o carro foi vendido | Numérica |
| `Present_Price` | Preço atual de tabela (valor do veículo novo) | Numérica |
| `Kms_Driven` | Total de quilômetros rodados | Numérica |
| `Fuel_Type` | Tipo de combustível: Petrol / Diesel / CNG | Categórica |
| `Seller_Type` | Tipo de vendedor: Dealer / Individual | Categórica |
| `Transmission` | Tipo de câmbio: Manual / Automatic | Categórica |
| `Owner` | Número de proprietários anteriores | Numérica |

> ℹ️ Os valores monetários estão na unidade original do dataset (**lakhs de rúpias indianas**, onde `1 lakh = 100.000 INR`). Portanto, um `MAE = 0.7260` equivale a um erro médio de ~0,73 lakh.

---

## 🛠 Stack Tecnológica

```python
pandas              # Manipulação e análise de dados
matplotlib / seaborn # Visualização de dados
scikit-learn        # Split, pré-processamento, modelos e métricas
xgboost             # Algoritmo de Gradient Boosting otimizado
joblib              # Serialização / persistência do modelo
```

---

## 📁 Estrutura do Repositório

```
.
├── vehicle_prices.ipynb          # Notebook principal (pipeline completo)
├── linear_regression_manual.ipynb # Implementação didática de regressão linear
├── cardata.csv                   # Base de dados original
├── model_gb_veiculos.pk          # Modelo campeão serializado (Gradient Boosting)
└── README.md
```

---

## 🔄 Pipeline do Projeto

```mermaid
flowchart LR
    A[📥 Carga<br/>cardata.csv] --> B[🔍 EDA<br/>estrutura e qualidade]
    B --> C[📦 Boxplots<br/>detecção de outliers]
    C --> D[⚙️ Feature Engineering<br/>criação de Age]
    D --> E[✂️ Tratamento<br/>de Outliers]
    E --> F[🔢 Encoding<br/>variáveis categóricas]
    F --> G[🎯 Split 75/25<br/>treino e teste]
    G --> H[🤖 Treino<br/>LR · RF · GB · XGB]
    H --> I[📏 Avaliação<br/>MAE · MAPE · R²]
    I --> J[🧪 Experimento<br/>de escala]
    J --> K[🏆 Seleção<br/>do modelo]
    K --> L[💾 Persistência<br/>joblib]
```

---

## 1. Análise Exploratória (EDA)

### 1.1 Qualidade dos dados

Primeira etapa: entender a estrutura antes de qualquer transformação.

| Verificação | Método | Resultado |
| :--- | :--- | :--- |
| Dimensões | `df.shape` | `(301, 9)` |
| Tipos de dados | `df.info()` | 2 `float64`, 3 `int64`, 4 `str` |
| Valores ausentes | `df.isna().sum()` | **0 nulos** — nenhuma imputação necessária |
| Cardinalidade | `df.nunique()` | `Car_Name` com 98 níveis (alta cardinalidade) |
| Estatística descritiva | `df.describe()` | Ver abaixo |

### 1.2 Estatísticas descritivas

| | Year | Selling_Price | Present_Price | Kms_Driven | Owner |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **mean** | 2013.63 | 4.66 | 7.63 | 36.947 | 0.04 |
| **std** | 2.89 | 5.08 | 8.64 | 38.887 | 0.25 |
| **min** | 2003 | 0.10 | 0.32 | 500 | 0 |
| **50% (mediana)** | 2014 | 3.60 | 6.40 | 32.000 | 0 |
| **max** | 2018 | **35.00** | **92.60** | **500.000** | 3 |

**Leitura crítica dos números:**
- `Selling_Price`: média (4,66) **acima** da mediana (3,60) → distribuição **assimétrica à direita**, puxada por poucos veículos de luxo.
- `Kms_Driven`: máximo de 500.000 km contra mediana de 32.000 km → presença de valores extremos.
- `Present_Price` chega a 92,60 enquanto o 3º quartil é 9,90 → cauda longa muito pronunciada.

### 1.3 Distribuição das variáveis categóricas

```
Fuel_Type                 Seller_Type              Transmission            Owner
├── Petrol ....... 239    ├── Dealer ....... 195   ├── Manual .... 261     ├── 0 .... 290
├── Diesel .......  60    └── Individual ... 106   └── Automatic .  40     ├── 1 ....  10
└── CNG ..........   2                                                     └── 3 ....   1
```

**Desbalanceamentos identificados:**
- `Fuel_Type = CNG` tem apenas **2 registros** → classe praticamente sem representatividade estatística.
- `Owner` é quase constante (**96,3%** com valor 0) → baixo poder discriminativo esperado.
- `Transmission` é dominado por manuais (**86,7%**).

### 1.4 Boxplots — detecção visual de outliers

Foram gerados boxplots para `Selling_Price`, `Present_Price`, `Year` e `Kms_Driven`, além de um boxplot comparativo entre preço de venda e preço de tabela.

**Por que boxplot?** Ele expõe simultaneamente mediana, quartis, dispersão (IQR) e pontos fora dos limites (`Q1 − 1.5×IQR` / `Q3 + 1.5×IQR`), permitindo decidir objetivamente o que tratar como valor discrepante.

**Conclusão visual:** o boxplot de `Selling_Price` mostrou um grupo isolado de registros acima de ~33, confirmando a assimetria sugerida pelas estatísticas descritivas.

---

## 2. Pré-Processamento

### 2.1 Feature Engineering — criação da variável `Age`

```python
ano_atual = datetime.datetime.now().year
df['Age'] = ano_atual - df['Year']
df.drop('Year', axis=1, inplace=True)
```

**Lógica da decisão:** o ano absoluto (`2014`) é um rótulo temporal arbitrário — não carrega significado intrínseco para o modelo. Já a **idade do veículo** (`11 anos`) é a informação que realmente dirige a depreciação. A transformação:

1. Converte uma referência temporal absoluta em uma **grandeza contínua interpretável**;
2. Alinha a variável ao mecanismo econômico real (depreciação ∝ idade);
3. Mantém a mesma informação com uma relação mais direta com o target.

A coluna `Year` original foi removida para evitar **redundância perfeita** (`Age` e `Year` são linearmente dependentes — manter as duas inflaria a multicolinearidade).

### 2.2 Tratamento de Outliers

Abordagem por **ordenação decrescente (sorting)** para inspecionar manualmente os valores extremos antes de decidir o corte:

```python
sorted(df['Selling_Price'], reverse=True)   # inspeção
df[df['Selling_Price'] >= 33.0]             # isolamento dos candidatos
df = df[df['Selling_Price'] < 33.0]         # aplicação do corte
```

| | Registros |
| :--- | :---: |
| Antes do corte | 301 |
| Após o corte | **299** |
| Removidos | 2 (0,66%) |

**Justificativa:** apenas 2 observações (0,66% da base) foram descartadas. Esses registros representavam veículos de altíssimo valor sem casos semelhantes na base para dar suporte ao aprendizado — ou seja, pontos que o modelo não conseguiria generalizar e que distorceriam desproporcionalmente a função de custo (especialmente em métricas quadráticas). O corte foi deliberadamente **conservador**: preserva 99,3% da informação.

### 2.3 Encoding das variáveis categóricas

Aplicado **mapeamento manual** (`.map()`) em vez de `OneHotEncoder` ou `LabelEncoder`:

```python
df['Fuel_Type']    = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Seller_Type']  = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
```

**Lógica da escolha:**
- **Controle explícito:** o mapeamento manual torna a codificação **determinística e auditável** — não depende da ordem alfabética ou de fit/transform, eliminando risco de inconsistência entre treino e produção.
- **Binárias sem custo dimensional:** `Seller_Type` e `Transmission` têm apenas 2 níveis — o encoding binário (0/1) é equivalente ao one-hot sem gerar colunas extras.
- **Compatibilidade com árvores:** para modelos baseados em árvore (RF, GB, XGB), que fazem partições por regras de corte, o encoding ordinal não introduz viés de ordem relevante, pois o algoritmo pode isolar qualquer categoria com cortes sucessivos.

### 2.4 Seleção de variáveis

```python
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)   # 7 features
y = df['Selling_Price']                              # target
```

**Por que remover `Car_Name`?** 98 valores únicos em 299 registros (**cardinalidade de ~33%**). Codificar essa variável geraria:
- Com one-hot → 98 colunas novas, explosão de dimensionalidade com apenas 224 amostras de treino (maldição da dimensionalidade);
- Com ordinal → uma ordem numérica artificial e sem sentido entre modelos de carro.

Além disso, a informação de "qual modelo é o carro" já está **parcialmente capturada** por `Present_Price`, que funciona como proxy do posicionamento de mercado do veículo.

### 2.5 Divisão Treino / Teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

| Conjunto | Observações | Proporção |
| :--- | :---: | :---: |
| **Treino** | 224 | 75% |
| **Teste** | 75 | 25% |

O `random_state=42` garante **reprodutibilidade** da divisão. O conjunto de teste é mantido intocado para medir a capacidade de **generalização** em dados inéditos.

> 📌 **Convenção de nomenclatura:** `X` maiúsculo = **matriz** (bidimensional, múltiplas features) · `y` minúsculo = **vetor** (unidimensional, apenas o target). Padrão herdado da álgebra linear e adotado pelo scikit-learn.

---

## 3. Construção dos Modelos

Foram treinados **4 algoritmos** com **hiperparâmetros default**, sob condições idênticas (mesmo split, mesmas features), garantindo uma comparação justa:

| # | Algoritmo | Família | Lógica de funcionamento |
| :-: | :--- | :--- | :--- |
| 1 | **Linear Regression** | Linear / Paramétrico | Ajusta um hiperplano minimizando a soma dos erros quadráticos. Serve como **baseline** — se modelos complexos não o superarem, a relação é essencialmente linear. |
| 2 | **Random Forest** | Ensemble / *Bagging* | Treina N árvores em paralelo sobre amostras bootstrap e subconjuntos aleatórios de features; a previsão é a **média** das árvores. Reduz variância. |
| 3 | **Gradient Boosting** | Ensemble / *Boosting* | Treina árvores **sequencialmente**, cada uma corrigindo o erro residual da anterior. Reduz viés progressivamente. |
| 4 | **XGBoost** | Ensemble / *Boosting* | Implementação otimizada de gradient boosting, com regularização L1/L2, tratamento nativo de dados esparsos e paralelização. |

```python
lr  = LinearRegression();                     lr.fit(X_train, y_train)
rf  = RandomForestRegressor(random_state=42);  rf.fit(X_train, y_train)
gb  = GradientBoostingRegressor(random_state=42); gb.fit(X_train, y_train)
xgb = XGBRegressor(random_state=42);          xgb.fit(X_train, y_train)
```

> 🔒 **`random_state=42` nos três modelos de ensemble.** O `RandomForestRegressor` é genuinamente estocástico (amostragem *bootstrap* + subconjunto aleatório de features por nó) e, sem semente fixa, produz métricas diferentes a cada execução. `GB` e `XGB` com hiperparâmetros default são efetivamente determinísticos (`subsample=1.0`, sem subamostragem de colunas), mas a semente é declarada de todo modo — se no futuro houver tuning de `subsample` ou `colsample_bytree`, a reprodutibilidade já está garantida por construção. A `LinearRegression` não tem componente aleatório.

Em seguida, a **inferência** foi executada de forma padronizada sobre `X_test` para todos os modelos, permitindo comparar os erros sob a mesma base de referência.

---

## 4. Métricas de Avaliação

Três métricas **complementares** foram escolhidas deliberadamente, porque cada uma expõe um aspecto diferente do erro:

| Métrica | O que mede | Interpretação | Direção |
| :--- | :--- | :--- | :---: |
| **MAE** <br/>*Mean Absolute Error* | Erro médio absoluto, na **unidade do target** | Erro típico em valor monetário. Trata todos os erros linearmente. | ↓ menor é melhor |
| **MAPE** <br/>*Mean Absolute % Error* | Erro médio **relativo** ao valor real | Erro em %. Revela falhas em veículos de baixo valor. | ↓ menor é melhor |
| **R²** <br/>*Coeficiente de Determinação* | Fração da **variância** explicada pelo modelo | `1.0` = ajuste perfeito. Penaliza erros ao **quadrado**. | ↑ maior é melhor |

**Por que três métricas e não apenas uma?** Porque elas podem **discordar** — e essa discordância é informação. O R² penaliza erros grandes quadraticamente (sensível a outliers), o MAE trata todos os erros igualmente (robusto), e o MAPE expõe erros proporcionalmente graves em valores pequenos. Neste projeto a discordância apareceu duas vezes de forma concreta: a **Regressão Linear** tem R² aceitável (`0.7107`) e MAPE catastrófico (`164.25%`), e o **GB vence o XGB em MAE/R² mas perde em MAPE** — dois diagnósticos que uma métrica isolada esconderia por completo (ver [seção 7](#-7-insights-principais)).

---

## 5. Resultados e Comparação dos Modelos

### 🏁 Dados na escala original

| Modelo | MAE ↓ | MAPE ↓ | R² Score ↑ | Avaliação |
| :--- | :---: | :---: | :---: | :--- |
| **🏆 Gradient Boosting** | **0.7260** | 23.05% | **0.9049** | **Melhor desempenho geral** — menor erro absoluto e maior poder explicativo |
| **🥈 XGBoost** | 0.7460 | **21.61%** | 0.8822 | Praticamente empatado em MAE; **melhor MAPE** da comparação |
| **🥉 Random Forest** | 0.9406 | 26.56% | 0.7745 | Ajuste razoável, mas erro ~30% maior que o líder |
| **⚠️ Linear Regression** | 1.3939 | **164.25%** | 0.7107 | *Baseline* — falha ao capturar relações não lineares |

> 🔁 Resultados reproduzíveis: `train_test_split(random_state=42)` + `random_state=42` em RF, GB e XGB.

### 📈 Visualização Real vs. Previsto

O notebook gera gráficos de dispersão (`y_test` × `y_pred`) com a linha diagonal de **previsão perfeita** (`y = x`) para os três cenários de escala. A leitura é direta: quanto mais próximos da diagonal os pontos de um modelo, melhor seu ajuste. Os modelos de *boosting* (laranja e roxo) concentram-se visivelmente mais perto da diagonal, enquanto a Regressão Linear (azul) apresenta dispersão acentuada — inclusive **previsões negativas** para veículos de baixo valor, algo impossível no domínio do problema.

---

## 6. Experimento de Escala: Original vs. Normalização vs. Padronização

Para validar empiricamente o impacto da escala das variáveis em diferentes famílias de algoritmos, os 4 modelos foram reavaliados em **3 cenários**:

| Cenário | Técnica | Transformação |
| :--- | :--- | :--- |
| **Original** | — | Dados brutos, sem reescala |
| **Normalização** | `MinMaxScaler` | Reescala para o intervalo `[0, 1]` |
| **Padronização** | `StandardScaler` | Média 0, desvio padrão 1 (*z-score*) |

> ⚠️ **Pré-requisito metodológico.** Em ambos os cenários transformados, o scaler é ajustado **somente no treino** (`fit_transform(X_train)`) e apenas **aplicado** ao teste (`transform(X_test)`). Ajustar o scaler no teste (`fit_transform(X_test)`) mapearia treino e teste para sistemas de coordenadas distintos e invalidaria toda a comparação — ver [Pontos de Atenção](#-pontos-de-atenção-e-melhorias-futuras).

### 📊 Resultados consolidados

| Modelo | Cenário | MAE ↓ | MAPE ↓ | R² Score ↑ |
| :--- | :--- | :---: | :---: | :---: |
| **Gradient Boosting** | Original | 0.7260 | 23.05% | 0.9049 |
| | Normalizado | 0.7261 | 23.09% | 0.9049 |
| | Padronizado | 0.7263 | 23.07% | 0.9049 |
| **XGBoost** | Original | 0.7460 | 21.61% | 0.8822 |
| | Normalizado | 0.7460 | 21.61% | 0.8822 |
| | Padronizado | 0.7460 | 21.61% | 0.8822 |
| **Random Forest** | Original | 0.9406 | 26.56% | 0.7745 |
| | Normalizado | 0.9355 | 26.31% | 0.7749 |
| | Padronizado | 0.9369 | 26.44% | 0.7748 |
| **Linear Regression** | Original | 1.3939 | 164.25% | 0.7107 |
| | Normalizado | 1.3939 | 164.25% | 0.7107 |
| | Padronizado | 1.3939 | 164.25% | 0.7107 |

### 🔬 Amplitude máxima entre os três cenários

| Modelo | Δ MAE | Δ R² | Veredito |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | `0.000000` | `0.000000` | Idêntico até a 6ª casa decimal |
| **XGBoost** | `0.000000` | `0.000000` | Idêntico até a 6ª casa decimal |
| **Gradient Boosting** | `0.000262` | `0.000054` | Ruído numérico (0,04% do MAE) |
| **Random Forest** | `0.005080` | `0.000368` | Ruído numérico (0,54% do MAE) |

**Resultado do experimento: a escala é irrelevante para os quatro modelos deste projeto.** Nenhum apresentou variação que não seja atribuível a desempate de cortes e aritmética de ponto flutuante. A explicação completa de *por que* isso acontece — inclusive para a Regressão Linear, contrariando a crença difundida de que modelos lineares exigem padronização — está no [insight 4](#4️⃣-a-escala-é-comprovadamente-irrelevante--e-o-motivo-é-diferente-para-cada-família).

---

## 💡 7. Insights Principais

### 1️⃣ Superioridade clara dos modelos de *Boosting*

O **Gradient Boosting** reduziu o erro médio absoluto quase à **metade** em relação ao baseline linear (`1.3939 → 0.7260`, queda de **47,9%**) e elevou o poder explicativo para **90,5%** (`R² 0.7107 → 0.9049`).

**O que isso significa:** a diferença de performance entre um modelo linear e modelos de ensemble é a evidência empírica de que **existem relações não lineares e interações entre variáveis** determinando o preço. A depreciação de um veículo não é linear no tempo (cai mais rápido nos primeiros anos), e o efeito da quilometragem depende do segmento do carro — exatamente o tipo de padrão que árvores capturam e um hiperplano não.

### 2️⃣ O MAPE de 164% da Regressão Linear: a anomalia mais reveladora

Enquanto o MAE da LR (`1.3939`) é apenas ~1,9× pior que o do GB, seu MAPE (`164.25%`) é **7× pior** que o dos modelos de árvore. Por que a discrepância?

O MAPE divide o erro pelo valor real: `|y − ŷ| / y`. Como a regressão linear ajusta **um único hiperplano global**, ela é forçada a usar a mesma inclinação em toda a faixa de preços — e extrapola para fora do domínio válido na ponta inferior. A inspeção das previsões confirma o diagnóstico de forma literal:

| Verificação no conjunto de teste | Resultado |
| :--- | :--- |
| Previsões **negativas** | **10 de 75** (13,3%) |
| Menor valor previsto | **`−8.76`** |
| Menor valor real da base | `0.12` |
| MAPE restrito aos **10 menores** preços reais | **933%** |

Um preço de venda negativo é **impossível no domínio do problema** — não é imprecisão, é violação de restrição. E como esses registros têm denominador próximo de zero, cada um deles injeta centenas de pontos percentuais no MAPE (o pior caso: real `0.17`, previsto `−8.76`, erro relativo de **5.253%**).

**Lição metodológica:** o R² de `0.7107` esconde isso por completo, porque é construído sobre o erro **quadrático** e portanto dominado pelos veículos caros, faixa em que o ajuste linear é razoável. Modelos de árvore são estruturalmente imunes a esse modo de falha: como cada previsão é a média de folhas construídas a partir de valores observados no treino, **a saída fica necessariamente dentro do intervalo visto** — nunca negativa. Esse é um argumento técnico concreto para preferi-los aqui, independente da comparação de métricas.

### 3️⃣ 🐛 O bug que quase se tornou uma conclusão

A primeira versão deste experimento continha um erro de uma palavra:

```python
sc = MinMaxScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm  = sc.fit_transform(X_test)   # ⚠️ deveria ser sc.transform(X_test)
```

Um scaler **é um modelo**: o `fit` estima parâmetros a partir dos dados e os guarda como estado (`data_min_`, `data_max_` no Min-Max; `mean_`, `scale_` no z-score). Chamar `fit_transform` no teste provoca **duas falhas simultâneas**:

1. **Reestima os parâmetros** a partir de dados que deveriam ser inéditos;
2. **Sobrescreve o estado aprendido no treino** — o objeto `sc` deixa de representar a transformação usada no ajuste do modelo.

O efeito é que treino e teste passam a viver em **sistemas de coordenadas diferentes**. O mínimo de `Kms_Driven` no treino não é o mesmo do teste, então o valor `0.089` significa coisas distintas nos dois conjuntos, e todas as fronteiras de decisão aprendidas apontam para o lugar errado.

**O que isso produziu nos resultados:** as três tabelas divergiam de forma aparentemente significativa (o R² do Random Forest caía de `0.79` para `0.67` na normalização, o do XGBoost de `0.88` para `0.82`). Com o bug corrigido, **toda essa variação desapareceu** — as diferenças reais são da 4ª casa decimal para baixo.

**Por que este é o insight mais importante do projeto:** a versão com bug gerava números perfeitamente plausíveis e uma narrativa convincente — "a normalização degradou os modelos de árvore". Essa conclusão era **falsa**, mas internamente coerente e difícil de questionar sem conhecer o comportamento esperado dos algoritmos. Foi a **expectativa teórica** (árvores são invariantes a transformações monotônicas, logo as três tabelas *deveriam* ser idênticas) que expôs o defeito — não a inspeção das métricas.

> 💡 **Lição transferível:** quando um resultado experimental contraria uma propriedade matemática conhecida do algoritmo, a hipótese mais provável não é uma descoberta — é um bug no pipeline. A prevenção estrutural para esta classe de erro é encapsular pré-processamento e modelo em um `sklearn.pipeline.Pipeline`, que torna o reajuste acidental do scaler **impossível por construção**.

### 4️⃣ A escala é comprovadamente irrelevante — e o motivo é diferente para cada família

Com o scaler corrigido e as sementes fixadas, o experimento dá um resultado limpo: **nenhum dos quatro modelos muda de desempenho** entre dados originais, normalizados e padronizados. Mas a razão **não é a mesma** para todos.

**Árvores (GB, XGB, RF) — invariância por monotonicidade.** Cada nó decide por uma regra da forma `atributo <= limiar`, que depende apenas da **ordenação relativa** das observações. `MinMaxScaler` e `StandardScaler` são transformações afins de inclinação positiva, portanto monotônicas crescentes: preservam a ordem. Um corte em `Kms_Driven <= 45000` simplesmente passa a ser `Kms_Driven_escalado <= 0.089` e gera **exatamente a mesma partição**. As diferenças residuais de GB (`ΔMAE = 0.000262`) e RF (`ΔMAE = 0.005080`) vêm de desempates entre cortes de ganho numericamente igual, que podem se resolver de forma diferente após o reescalonamento.

**Regressão Linear — invariância por álgebra linear, contrariando a crença comum.** Existe a ideia difundida de que "modelos lineares precisam de padronização". Isso é verdade em dois casos, e **nenhum deles é este**:

| Caso | Por que a escala importa | Aplica-se aqui? |
| :--- | :--- | :---: |
| Modelos **regularizados** (Ridge, Lasso, ElasticNet) | A penalidade `α·‖β‖` compara coeficientes de grandezas diferentes — sem padronizar, variáveis de magnitude alta são punidas desproporcionalmente | ❌ Não |
| Otimização por **descida de gradiente** (`SGDRegressor`, redes neurais) | A velocidade de convergência depende do condicionamento das escalas | ❌ Não |

A `LinearRegression` do Scikit-learn resolve o problema de mínimos quadrados **analiticamente**, por decomposição em valores singulares (SVD). Como uma transformação afim das colunas **não altera o subespaço gerado por** `[1, X]`, a projeção ortogonal de `y` sobre esse subespaço é a mesma. Os **coeficientes mudam** (passam a estar em outra unidade), a **previsão não muda**. Daí o `MAE = 1.393906` e o `R² = 0.710697` idênticos até a 6ª casa decimal nos três cenários.

> ✅ **Valor do experimento:** é **diagnóstico**, não otimização. Ele produz a evidência empírica de que a escala não é uma alavanca disponível neste pipeline, redirecionando o esforço para onde há retorno real — *tuning* de hiperparâmetros e validação cruzada.

### 5️⃣ `Present_Price` domina o problema — e isso redefine o que o modelo aprende

A correlação de Pearson com o target expõe a estrutura real da base:

| Variável | Correlação com `Selling_Price` |
| :--- | :---: |
| `Present_Price` | **+0.8804** |
| `Age` | −0.2731 |

O preço de tabela sozinho explica a maior parte da variação. **Na prática, o modelo não está aprendendo "quanto vale um carro" — está aprendendo uma taxa de retenção de valor sobre um preço de referência que já lhe é fornecido.** É uma tarefa substancialmente mais fácil, e ajuda a explicar o `R²` acima de `0.90` com apenas 224 amostras de treino.

**O ponto não óbvio está no `Age`.** Economicamente, a idade é *o* motor da depreciação — mas sua correlação linear com o preço é de apenas `−0.2731`. Isso não é contradição: a depreciação age **proporcionalmente**, não em valor absoluto. Um SUV de 7 anos ainda vale mais que um popular zero-quilômetro, então o efeito da idade sobre o preço absoluto fica diluído quando medido isoladamente. O efeito só emerge **condicionado ao `Present_Price`**, ou seja, como uma **interação entre as duas variáveis**.

E é exatamente aí que a Regressão Linear falha estruturalmente: sem termos de interação explícitos, ela só consegue somar contribuições independentes (`β₁·Present_Price + β₂·Age`), quando o mecanismo real é aproximadamente **multiplicativo** (`Present_Price × fator_de_retenção(Age)`). Árvores representam interações nativamente, encadeando cortes nas duas variáveis ao longo de um mesmo caminho. **O insight 1 e o insight 2 são, na origem, o mesmo fenômeno visto por ângulos diferentes.**

### 6️⃣ GB vs. XGBoost: o critério de desempate depende do objetivo de negócio

| Métrica | Gradient Boosting | XGBoost | Vencedor |
| :--- | :---: | :---: | :--- |
| MAE | **0.7260** | 0.7460 | GB (−2,7% de erro) |
| MAPE | 23.05% | **21.61%** | XGB (−6,2% de erro) |
| R² | **0.9049** | 0.8822 | GB (+2,6%) |

A diferença de MAE é de apenas **0.0200** (2,7%) — irrelevante diante da variância esperada em um conjunto de teste de 75 amostras. Mas as métricas **discordam de forma sistemática**, e essa discordância tem causa identificável: o MAE é dominado pelos veículos de alto valor (onde os erros absolutos são grandes), enquanto o MAPE é dominado pelos de baixo valor (onde o denominador é pequeno). O GB ganha no primeiro grupo, o XGB no segundo.

**Tradução para decisão de negócio:**
- Precificar **veículos de maior valor** (onde o erro absoluto pesa mais no resultado financeiro) → **Gradient Boosting**
- Precificar **veículos populares / entrada** (onde o erro percentual é o que o cliente percebe) → **XGBoost**

> ⚠️ **Ressalva honesta:** com 75 amostras de teste e um único *split*, uma diferença de 2,7% no MAE **não é estatisticamente distinguível de ruído**. A escolha do GB está corretamente justificada por liderar 2 das 3 métricas, mas afirmar que ele é *definitivamente* superior ao XGB exigiria validação cruzada com intervalo de confiança — item priorizado nas [melhorias futuras](#evoluções-do-modelo).

### 7️⃣ Limitações estruturais da base

| Observação | Implicação |
| :--- | :--- |
| `Fuel_Type = CNG` tem só 2 registros | O modelo não aprende esse segmento; previsões para GNV são extrapolação |
| `Owner` é 0 em 96,3% dos casos | Variável quase constante → contribuição preditiva marginal |
| Apenas 299 registros / 75 no teste | Métricas com alta variância; sensíveis ao `random_state` do split |
| `Present_Price` é a feature dominante (`r = 0.8804`) | O modelo aprende largamente uma **taxa de retenção** sobre o preço de tabela, e não o preço do zero (ver [insight 5](#5️⃣-present_price-domina-o-problema--e-isso-redefine-o-que-o-modelo-aprende)) |
| A base não informa a **data de cada venda** | Não é possível saber a que momento cada `Selling_Price` se refere, nem corrigir efeitos de inflação ao longo do período de coleta. Assume-se implicitamente que todos os preços são contemporâneos entre si |

---

## 🏆 8. Modelo Final e Persistência

### Decisão

**Gradient Boosting Regressor** treinado sobre os **dados na escala original**.

| Métrica | Resultado | Interpretação |
| :--- | :---: | :--- |
| **MAE** | `0.7260` | Erro médio de ~0,73 lakh por previsão |
| **MAPE** | `23.05%` | Desvio médio de ~23% do valor real |
| **R²** | `0.9049` | **~90% da variância dos preços é explicada pelo modelo** |

### Justificativa da escolha

1. **Melhor em 2 das 3 métricas** (MAE e R²), com MAPE competitivo;
2. **Desempenho idêntico nos três cenários de escala** — o resultado não depende de uma configuração de pré-processamento específica;
3. **Simplicidade de leitura:** como as três escalas produzem resultados equivalentes, a escala original mantém as métricas interpretáveis na unidade do negócio (lakhs);
4. **Resultado alcançado com hiperparâmetros default** — há margem de ganho adicional via tuning.

### Serialização

```python
joblib.dump(gb, 'model_gb_veiculos.pk')
```

O modelo treinado é persistido com `joblib`, permitindo carregá-lo para inferência sem retreino:

```python
import joblib
from sklearn.preprocessing import StandardScaler

modelo = joblib.load('model_gb_veiculos.pk')

# O artefato salvo corresponde ao último `fit` do notebook,
# executado sobre X_train_std -> espera entrada padronizada.
ss = StandardScaler().fit(X_train)

# Ordem das features: Present_Price, Kms_Driven, Fuel_Type,
#                     Seller_Type, Transmission, Owner, Age
entrada = ss.transform([[9.85, 6900, 0, 0, 0, 0, 8]])
predicao = modelo.predict(entrada)
```

> ⚠️ **Contrato do artefato.** Ao carregar o `.pk`, a entrada deve respeitar três condições, todas herdadas do treino: a **ordem** das features, a **codificação** das categóricas (mesmos mapeamentos) e o cálculo de `Age`, e a **escala** — o arquivo versionado contém o modelo ajustado sobre `X_train_std`, cujos limiares estão em z-score (ex.: 64% dos cortes de `Kms_Driven` são negativos). O `StandardScaler` não é serializado junto, de modo que `μ` e `σ` precisam ser reconstituídos a partir de `X_train`. O pickle foi gravado com **scikit-learn 1.9.0**; versões diferentes emitem `InconsistentVersionWarning` na carga.

---

## 🔧 Pontos de Atenção e Melhorias Futuras

Itens identificados durante a análise, documentados para transparência e próximas iterações:

### ✅ Correções já aplicadas

| # | Item | Diagnóstico | Correção aplicada |
| :-: | :--- | :--- | :--- |
| 1 | **Scaler reajustado no teste** | `sc.fit_transform(X_test)` reestimava os parâmetros a partir de dados inéditos *e* sobrescrevia o estado ajustado no treino, mapeando os dois conjuntos para sistemas de coordenadas distintos | `sc.transform(X_test)` — o scaler é ajustado **somente** no treino. Ver [insight 3](#3️⃣--o-bug-que-quase-se-tornou-uma-conclusão) |
| 2 | **Ausência de `random_state`** nos modelos | O `RandomForestRegressor` é estocástico (*bootstrap* + subamostragem de features), produzindo métricas diferentes a cada execução e tornando as tabelas não reproduzíveis | `random_state=42` em RF, GB e XGB |

### 📌 Características da implementação

Comportamentos do código que condicionam a leitura dos resultados ou o reuso do artefato:

| # | Item | Descrição |
| :-: | :--- | :--- |
| 3 | **Escala do modelo serializado** | A célula de persistência salva a variável `gb`, que nesse ponto corresponde ao **último `fit` executado**, sobre `X_train_std`. O artefato portanto opera em z-score, e não na escala original em que as métricas da seção 8 foram reportadas — equivalentes entre si, mas não intercambiáveis na entrada. O `StandardScaler` não é serializado junto; ver o contrato do artefato na [seção 8](#serialização) |
| 4 | **Análise de correlação não executada** | As células de matriz de correlação e *heatmap* contêm apenas comentários, sem código. Os coeficientes citados no [insight 5](#5️⃣-present_price-domina-o-problema--e-isso-redefine-o-que-o-modelo-aprende) foram calculados diretamente sobre a base |
| 5 | **`Age` calculado com `datetime.now()`** | A idade é derivada do ano de execução, não de uma referência fixa. **Sem impacto nas métricas** — `Age = C − Year` é um deslocamento constante, invariante para árvores (transformação monotônica) e absorvido pelo intercepto na regressão linear. A consequência prática se restringe a treinar em um ano e inferir em outro, quando o mesmo veículo passa a receber um `Age` diferente do visto no treino |

### Evoluções do modelo

- [ ] **Validação cruzada** (`KFold` / `cross_val_score`) — com apenas 75 amostras no teste, um único split produz métricas instáveis. CV daria estimativas com intervalo de confiança.
- [ ] **Tuning de hiperparâmetros** via `GridSearchCV` ou `RandomizedSearchCV` (`n_estimators`, `learning_rate`, `max_depth`, `subsample`) — todos os resultados atuais usam defaults.
- [ ] **Feature importance** (`gb.feature_importances_` ou SHAP) — quantificar a contribuição de cada variável e confirmar a hipótese de dominância de `Present_Price`.
- [ ] **Análise de resíduos** — verificar se os erros são homocedásticos ou se o modelo erra sistematicamente em alguma faixa de preço.
- [ ] **Transformação logarítmica do target** (`log1p(Selling_Price)`) — dada a assimetria à direita, pode reduzir o impacto da cauda longa e melhorar o MAPE.
- [ ] **Pipeline do scikit-learn** (`Pipeline` + `ColumnTransformer`) — encapsular encoding, scaling e modelo em um único objeto, eliminando por construção o risco de inconsistência entre treino e inferência.
- [ ] **Reaproveitar `Car_Name`** via *target encoding* ou extração da marca (primeira palavra) — recupera informação de segmento sem explodir a dimensionalidade.
- [ ] **API de inferência** (FastAPI / Streamlit) para disponibilizar o modelo.

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost joblib notebook
```

### Execução

```bash
# 1. Clonar o repositório
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>

# 2. Abrir o notebook
jupyter notebook vehicle_prices.ipynb
```

> 📌 O arquivo `cardata.csv` deve estar no **mesmo diretório** do notebook, ou o caminho em `pd.read_csv()` deve ser ajustado.

### Executar as células na ordem

O notebook é sequencial: imports → EDA → pré-processamento → split → treino → avaliação → experimento de escala → persistência. Executar fora de ordem causa erros de variáveis não definidas.

---

## 📚 Conceitos Aplicados

`Análise Exploratória de Dados (EDA)` · `Detecção e Tratamento de Outliers` · `Feature Engineering` ·
`Categorical Encoding` · `Train/Test Split` · `Regressão Linear` · `Ensemble Learning` ·
`Bagging (Random Forest)` · `Boosting (Gradient Boosting, XGBoost)` · `Normalização (Min-Max)` ·
`Padronização (Z-Score)` · `Métricas de Regressão (MAE, MAPE, R²)` · `Invariância de Escala em Árvores` ·
`Comparação Multi-Métrica de Modelos` · `Serialização de Modelos`

---

## 📄 Licença

Projeto desenvolvido com finalidade educacional e de portfólio.
Dataset de origem pública, disponível no [Kaggle](https://www.kaggle.com/datasets/athirags/car-data).

---

<div align="center">

**⭐ Se este projeto foi útil, considere deixar uma estrela no repositório**

</div>
