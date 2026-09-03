import os
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "data/processed/outer_folds/fold_1_train.csv"
output_dir = "data/processed/samples"

def amostragem(df, coluna_alvo, n_linhas):
    df_recorte, _ = train_test_split(
        df, 
        train_size=n_linhas, 
        stratify=df[coluna_alvo], 
        random_state=42
    )
    return df_recorte

def verifica_diretorio(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

if __name__ == "__main__":
    n_linhas = 16000
    df = pd.read_csv(csv_path)

    df_recortado = amostragem(df, coluna_alvo="Class", n_linhas=n_linhas)

    print("Contagem absoluta:\n", df_recortado["Class"].value_counts())
    proporcao = df_recortado["Class"].value_counts(normalize=True).to_dict()
    print("\nProporção das classes:", proporcao)

    verifica_diretorio(output_dir)
    nome_arquivo = f"amostra_{n_linhas}_train_fold1.csv"
    caminho_final = os.path.join(output_dir, nome_arquivo)
    
    df_recortado.to_csv(caminho_final, index=False)
    print(f"\nArquivo salvo com sucesso em: {caminho_final}")