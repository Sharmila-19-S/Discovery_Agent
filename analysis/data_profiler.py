import pandas as pd
import json


class DataProfiler:

    @staticmethod
    def profile(df: pd.DataFrame):

        profile = {}

        # =====================================================
        # BASIC INFORMATION
        # =====================================================

        profile["rows"] = int(df.shape[0])
        profile["columns"] = int(df.shape[1])

        profile["column_names"] = list(df.columns)


        # =====================================================
        # DATA TYPES
        # =====================================================

        profile["data_types"] = {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        }


        # =====================================================
        # MISSING VALUES
        # =====================================================

        profile["missing_values"] = {
            col: int(value)
            for col, value in df.isnull().sum().items()
            if value > 0
        }

        # If there are no missing values
        if not profile["missing_values"]:
            profile["missing_values"] = "No missing values"


        # =====================================================
        # DUPLICATES
        # =====================================================

        profile["duplicate_rows"] = int(
            df.duplicated().sum()
        )


        # =====================================================
        # NUMERIC COLUMNS
        # =====================================================

        numeric_columns = list(
            df.select_dtypes(include="number").columns
        )

        profile["numeric_columns"] = numeric_columns


        # =====================================================
        # CATEGORICAL COLUMNS
        # =====================================================

        categorical_columns = list(
            df.select_dtypes(
                exclude=["number", "datetime"]
            ).columns
        )

        profile["categorical_columns"] = categorical_columns


        # =====================================================
        # DATE COLUMNS
        # =====================================================

        date_columns = list(
            df.select_dtypes(
                include=["datetime"]
            ).columns
        )

        profile["date_columns"] = date_columns


        # =====================================================
        # NUMERIC SUMMARY
        # =====================================================
        #
        # Only send useful statistics.
        # Avoid df.describe(include="all").
        # =====================================================

        numeric_statistics = {}

        for col in numeric_columns:

            numeric_statistics[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 3),
                "median": round(float(df[col].median()), 3)
            }

        profile["numeric_statistics"] = numeric_statistics


        # =====================================================
        # CATEGORICAL SUMMARY
        # =====================================================
        #
        # Send number of unique values and only a few examples.
        # =====================================================

        categorical_summary = {}

        for col in categorical_columns:

            categorical_summary[col] = {
                "unique_values": int(df[col].nunique()),
                "examples": [
                    str(x)
                    for x in df[col]
                    .dropna()
                    .unique()[:5]
                ]
            }

        profile["categorical_summary"] = categorical_summary


        # =====================================================
        # SAMPLE ROWS
        # =====================================================
        #
        # Only 3 rows are enough for metadata understanding.
        # =====================================================

        profile["sample_rows"] = (
            df.head(3)
            .astype(str)
            .to_dict(orient="records")
        )


        return profile


    # =========================================================
    # CONVERT PROFILE TO JSON
    # =========================================================

    @staticmethod
    def to_json(profile):

        return json.dumps(
            profile,
            indent=2,
            ensure_ascii=False
        )