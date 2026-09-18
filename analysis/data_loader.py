import pandas as pd


def load_data(file_path):
    try:
        if file_path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path)

        # Automatically detect CSV delimiter
        df = pd.read_csv(
            file_path,
            sep=None,
            engine="python"
        )

        return df

    except Exception as e:
        raise RuntimeError(f"Could not read dataset: {str(e)}")