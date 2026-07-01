import pandas as pd

def build_metadata(xlsx_path, index_path, output_csv):
    df_meta = pd.read_excel(xlsx_path)
    df_meta.columns = df_meta.columns.str.strip().str.title()

    df_meta = df_meta.rename(columns={
        "Recoding Original Name": "Speaker_ID",
        "Updrs": "UPDRS_total",
        "H/Y": "Hoehn_Yahr",
        "Sex": "Gender",
        "Age": "Age",
        "Time After Diagnosis": "After_Diagnosis"
    })

    df_index = pd.read_csv(index_path)

    df_meta["Speaker_ID"] = df_meta["Speaker_ID"].astype(str).str.strip()
    df_index["Speaker_ID"] = df_index["Speaker_ID"].astype(str).str.strip()

    df = pd.merge(df_index, df_meta, on="Speaker_ID", how="left")

    df = df[df["Label"].isin(["PD", "HC"])]
    df = df.dropna(subset=["Age", "Gender", "Hoehn_Yahr", "After_Diagnosis"])

    df.to_csv(output_csv, index=False)
    return df
