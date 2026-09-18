import pandas as pd
import os


class DataLoader:

    @staticmethod
    def load(file_path: str, nrows=None):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".csv":

            df = pd.read_csv(
                file_path,
                nrows=nrows
            )

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(
                file_path,
                nrows=nrows
            )

        else:

            raise Exception("Unsupported file format.")

        return df