# Detecção de Fraudes Financeiras

Trabalho de Conclusão de Curso — Ciência da Computação, Universidade Federal de Lavras (UFLA).

Comparação entre o algoritmo não supervisionado de agrupamento por densidade **HDBSCAN** (fraude tratada como ruído) e as abordagens supervisionadas de classificação **Random Forest** e **XGBoost**, sobre a base de transações de cartão de crédito do Machine Learning Group da Université Libre de Bruxelles.

**Autor:** Heitor Rodrigues Sabino
**Coorientadora:** Elaine Cecilia Gatto

---

## Conjunto de dados

`creditcard.csv` — 284.807 transações europeias de setembro de 2013, 31 variáveis, 492 fraudes (0,1727%). As variáveis `V1`–`V28` são componentes principais (PCA) anonimizados; `Time`, `Amount` e `Class` são originais.

O arquivo tem ~150 MB e **não é versionado** (acima do limite do GitHub). Baixe de
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
e coloque em `data/raw/creditcard.csv`.

---

## Estrutura do repositório

```
.
├── data/
│   ├── raw/                                  # base original (não versionada)
│   │   └── creditcard.csv
│   └── processed/                            # artefatos gerados (não versionados)
│       ├── outer_folds/                      # folds externos: treino/teste
│       ├── inner_folds/                      # folds internos: treino/validação
│       └── samples/                          # recortes amostrais estratificados
│
├── src/
│   ├── preprocessing/
│   │   ├── cross_validation.py               # geração de folds estratificados
│   │   └── run_fold_generation.py            # entrada: constrói toda a árvore de folds
│   │
│   ├── models/
│   │   ├── grid_search.py                    # busca em grade (Random Forest / XGBoost)
│   │   ├── run_supervised_grid_search.py     # entrada: modelos supervisionados
│   │   └── run_hdbscan_grid_search.py        # entrada: HDBSCAN (grid por DBCV + avaliação externa)
│   │
│   └── utils/
│       ├── elbow_method.py                   # estimativa de eps pelo método do cotovelo
│       └── stratified_sampling.py            # recorte estratificado da base
│
├── results/
│   ├── supervised/                           # saída de run_supervised_grid_search.py
│   └── unsupervised/                         # saída de run_hdbscan_grid_search.py
│       ├── hdbscan_grid_best_summary.csv
│       └── hdbscan_grid_results_by_fold.csv
│
├── docs/monografia/                          # texto do TCC (LaTeX)
│   ├── main.tex
│   ├── refbib.bib
│   └── uflamon.cls
│
├── requirements.txt
└── README.md
```

**Convenções.** Todos os módulos usam `snake_case` em inglês. O prefixo `run_`
identifica os pontos de entrada executáveis; os demais arquivos são bibliotecas
importadas por eles. Os caminhos são relativos à raiz do repositório.

---

## Instalação

```bash
pip install -r requirements.txt
```

A etapa não supervisionada depende de `cuml-cu12` e `dbcv`, que exigem Linux com
GPU NVIDIA (CUDA 12) e não são instaláveis pelo `requirements.txt`. Os comandos
estão documentados no próprio arquivo. Esse experimento foi executado no Google Colab.

---

## Execução

Todos os scripts assumem a **raiz do repositório** como diretório de trabalho.

**1. Geração dos folds** — nested cross-validation estratificada, 5 folds externos 80/20 e divisão interna 80/20:

```bash
python src/preprocessing/run_fold_generation.py
```

Produz `data/processed/outer_folds/` (227.845 treino / 56.961 teste por fold) e
`data/processed/inner_folds/` (182.276 treino / 45.569 validação), com metadados em JSON.

**2. Grid search supervisionado** — Random Forest ou XGBoost, seleção por F1:

```bash
python src/models/run_supervised_grid_search.py
```

O modelo (`"rf"` ou `"xgb"`) e a grade são definidos no bloco `__main__`.
Resultados em `results/supervised/results_by_fold.csv` e `results_summary.csv`.

**3. HDBSCAN** — grid interno otimizado por DBCV e avaliação nos folds externos:

```bash
python src/models/run_hdbscan_grid_search.py
```

Requer GPU. As constantes `INTERNAL_FOLDS_DIR`, `OUTER_FOLDS_DIR` e `RESULTS_DIR`
no bloco `__main__` apontam para o Google Drive, refletindo o ambiente Colab em que
o experimento foi executado; ajuste-as para `data/processed/` e `results/unsupervised/`
ao rodar localmente.

**Utilitários:**

```bash
python src/utils/elbow_method.py         # estimativa de eps
python src/utils/stratified_sampling.py  # recorte estratificado
```

---

## Metodologia

- **Validação:** nested cross-validation estratificada — 5 folds externos (80/20) e, dentro de cada treino externo, uma divisão interna 80/20 para calibração. As partições de seleção de hiperparâmetros nunca coincidem com as de estimativa de desempenho.
- **Pré-processamento:** `Time` padronizado por Z-score (natureza incremental e limitada); `Amount` por *Robust Scaler* (assimetria de 16,98 e curtose de 845 — mediana e IQR evitam distorção pelos valores extremos); `V1`–`V28` mantidos, por já resultarem de PCA e serem mutuamente ortogonais.
- **Otimização:** busca em grade exaustiva no ciclo interno — por F1 nos modelos supervisionados, por DBCV (*Density-Based Clustering Validation*) no HDBSCAN, mantendo este último inteiramente não supervisionado na seleção.
