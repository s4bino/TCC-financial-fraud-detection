import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------- 
#                       Funções auxiliares
# ----------------------------------------------------------------

def scaling_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    cols = X_train.columns
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)

    return X_train_scaled, X_test_scaled

def split_features_target(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> tuple:
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]
    return X_train, y_train, X_test, y_test


def get_optimal_eps(X) -> float:
    dim = X.shape[1]
    min_pts = 2 * dim
    neighbor_index = (2 * dim - 1) - 1
    
    neigh = NearestNeighbors(n_neighbors=min_pts)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)

    sorted_distances = np.sort(distances[:, neighbor_index], axis=0)
    x_indices = np.arange(len(sorted_distances))

    kneedle = KneeLocator(x_indices, sorted_distances, S=1.0, curve="convex", direction="increasing")

    return kneedle.knee_y


if __name__ == "__main__":  
    fold = 1
    df_train = pd.read_csv(f"data/processed/outer_folds/fold_{fold}_train.csv")
    df_test = pd.read_csv(f"data/processed/outer_folds/fold_{fold}_test.csv")
    target = "Class"
    
    X_train, y_train, X_test, y_test = split_features_target(df_train, df_test, target)

    X_train_sc, X_test_sc = scaling_data(X_train, X_test)

    eps_otimo = get_optimal_eps(X_train_sc)
    min_samples_oficial = 2 * X_train_sc.shape[1]

    print(f"Dimensionalidade: {X_train_sc.shape[1]}")
    print(f"Min_samples sugerido: {min_samples_oficial}")
    print(f"EPS sugerido (Sander et al.): {eps_otimo}")
