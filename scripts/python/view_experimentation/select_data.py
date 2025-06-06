import pandas as pd

INDEX_COLS = ["setup", "matrix", "restart_param", "solver"]

def df_sel_setup(df, setup) -> pd.DataFrame:
    return df[df["setup"] == setup]

def df_sel_setup_matrix(df, setup, matrix) -> pd.DataFrame:
    return df[(df["setup"] == setup) & (df["matrix"] == matrix)]

def df_sel_setup_matrix_restart(df, setup, matrix, restart_param) -> pd.DataFrame:
    return df[
        (df["setup"] == setup) &
        (df["matrix"] == matrix) &
        (df["restart_param"] == restart_param)
    ]