
usar o F2 ao inves de f1

usar dbcv

plotar graficos da base e clusters

adicionar formulas matematicas

--------------


	"Outra ideia que você pode usar para avaliar é usar o Score de outlier do HDBSCAN chamado GLOSH ("clusterer.outlier_scores_"). Ele dá pontuações para os outliers baseados no quão perto eles estão de uma região densa, assim você pode selecionar que tipo de outlier queria achar como "fraude = outlier_score > threshold"."

--> usar estiatisca como intervalos inter quartis para definir o tresh hold e justificar que estamos ao menos pegando fraudes onde o valor não acompanha as transações convencionais.

Outliers são observações que se desviam significativamente do padrão dos demais dados. Uma ferramenta comum para sua identificação é o Intervalo Interquartil (IQR), calculado pela diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1). Pontos situados além de 1,5× IQR dos quartis são considerados anomalias. O tratamento desses valores pode ocorrer via Winsorização, técnica que limita os valores extremos a um percentil especificado, reduzindo o impacto dos outliers sem descartar os dados (MONTGOMERY, 2024; ABUZAID; ALKRONZ, 2024).

-----------

## HDBSCAN

CORE SG --> que consegue gerar de forma eficiente diversos resultados do HDBSCAN para um conjunto de _min_samples,_ pode ser encontrado no GitHub: [https://github.com/antoniocavalcante/coresg](https://github.com/antoniocavalcante/coresg).

Como você quer que fraudes virem ruído, aumentar min_cluster_size geralmente ajuda.

Então se for testar tente valores entre: 50, 100, 250, 500, 1000, são valores razoáveis para varrer.
O _min_samples_ você acompanha o _min_clsuter_size_, coloque o mesmo valor.

Avalie usando: Precision, Recall, F1 ou Matthews Correlation Coefficient (MCC).

Outra ideia que você pode usar para avaliar é usar o Score de outlier do HDBSCAN chamado GLOSH ("clusterer.outlier_scores_"). Ele dá pontuações para os outliers baseados no quão perto eles estão de uma região densa, assim você pode selecionar que tipo de outlier queria achar como "fraude = outlier_score > threshold".

Caso a dimensionalidade esteja atrapalhando, pode usar redução de dimensionalidade com UMAP antes do HDBSCAN.

----

fazer com a coluna time e sem a coluna time, ver se tem relação relaçao com o atributo classe