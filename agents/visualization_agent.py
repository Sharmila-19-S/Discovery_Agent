import json
from pathlib import Path

import pandas as pd
import numpy as np


class VisualizationAgent:

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # COLUMN DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def detect_columns(df):

        numeric = list(df.select_dtypes(include=np.number).columns)

        datetime_cols = list(
            df.select_dtypes(include=["datetime", "datetimetz"]).columns
        )

        categorical = [
            c for c in df.columns
            if c not in numeric and c not in datetime_cols
        ]

        # Detect date-like object columns
        for col in categorical.copy():

            sample = df[col].dropna().astype(str).head(100)

            if len(sample) == 0:
                continue

            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed"
            )

            if parsed.notna().mean() >= 0.8:
                datetime_cols.append(col)
                categorical.remove(col)

        # Detect time-like columns
        time_cols = []

        for col in categorical.copy():

            sample = df[col].dropna().astype(str).head(100)

            if len(sample) == 0:
                continue

            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="%H:%M:%S"
            )

            if parsed.notna().mean() >= 0.8:
                time_cols.append(col)

        return {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols,
            "time": time_cols
        }

    # ---------------------------------------------------------
    # IDENTIFY USEFUL COLUMNS
    # ---------------------------------------------------------

    @staticmethod
    def is_identifier(df, col):

        name = str(col).lower()

        identifier_words = [
            "id",
            "identifier",
            "code",
            "uuid",
            "transaction_id",
            "customer_id",
            "user_id",
            "order_id"
        ]

        if any(word == name or name.endswith("_" + word)
               for word in identifier_words):
            return True

        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        return unique_ratio > 0.95

    @staticmethod
    def useful_numeric_columns(df, numeric_columns):

        return [
            c for c in numeric_columns
            if not VisualizationAgent.is_identifier(df, c)
        ]

    @staticmethod
    def useful_categorical_columns(df, categorical_columns):

        result = []

        for col in categorical_columns:

            unique_count = df[col].nunique(dropna=True)

            if unique_count <= 50:
                result.append(col)

        return result

    # ---------------------------------------------------------
    # CHART CREATION
    # ---------------------------------------------------------

    def create_visualizations(self, df):

        detected = self.detect_columns(df)

        numeric = self.useful_numeric_columns(
            df,
            detected["numeric"]
        )

        categorical = self.useful_categorical_columns(
            df,
            detected["categorical"]
        )

        datetime_cols = detected["datetime"]
        time_cols = detected["time"]

        visualizations = []

        chart_id = 1

        # -----------------------------------------------------
        # 1. NUMERIC DISTRIBUTIONS
        # -----------------------------------------------------

        for col in numeric[:8]:

            visualizations.append({
                "id": chart_id,
                "type": "histogram",
                "title": f"Distribution of {col}",
                "columns": [col],
                "x": col,
                "purpose": "Understand the distribution of a numeric variable."
            })

            chart_id += 1

            visualizations.append({
                "id": chart_id,
                "type": "boxplot",
                "title": f"Outliers in {col}",
                "columns": [col],
                "y": col,
                "purpose": "Identify spread and potential outliers."
            })

            chart_id += 1

        # -----------------------------------------------------
        # 2. CATEGORICAL + NUMERIC
        # -----------------------------------------------------

        for cat in categorical[:8]:

            for num in numeric[:4]:

                grouped = (
                    df.groupby(cat)[num]
                    .mean()
                    .sort_values(ascending=False)
                )

                if len(grouped) < 2:
                    continue

                visualizations.append({
                    "id": chart_id,
                    "type": "bar",
                    "title": f"Average {num} by {cat}",
                    "columns": [cat, num],
                    "x": cat,
                    "y": num,
                    "aggregation": "mean",
                    "purpose": "Compare a numeric measure across categories."
                })

                chart_id += 1

                break

        # -----------------------------------------------------
        # 3. NUMERIC + NUMERIC
        # -----------------------------------------------------

        if len(numeric) >= 2:

            correlation = df[numeric].corr(
                method="pearson"
            )

            pairs = []

            for i in range(len(numeric)):

                for j in range(i + 1, len(numeric)):

                    c1 = numeric[i]
                    c2 = numeric[j]

                    value = correlation.loc[c1, c2]

                    if pd.notna(value):
                        pairs.append(
                            (abs(value), c1, c2, value)
                        )

            pairs.sort(reverse=True)

            for _, c1, c2, value in pairs[:5]:

                visualizations.append({
                    "id": chart_id,
                    "type": "scatter",
                    "title": f"{c1} vs {c2}",
                    "columns": [c1, c2],
                    "x": c1,
                    "y": c2,
                    "correlation": round(float(value), 4),
                    "purpose": "Explore relationships between numeric variables."
                })

                chart_id += 1

        # -----------------------------------------------------
        # 4. DATETIME + NUMERIC
        # -----------------------------------------------------

        for date_col in datetime_cols:

            if not numeric:
                continue

            for num in numeric[:4]:

                visualizations.append({
                    "id": chart_id,
                    "type": "time_series",
                    "title": f"{num} over {date_col}",
                    "columns": [date_col, num],
                    "x": date_col,
                    "y": num,
                    "aggregation": "sum",
                    "purpose": "Identify trends and changes over time."
                })

                chart_id += 1

        # -----------------------------------------------------
        # 5. TIME + NUMERIC
        # -----------------------------------------------------

        for time_col in time_cols:

            if not numeric:
                continue

            for num in numeric[:3]:

                visualizations.append({
                    "id": chart_id,
                    "type": "time_of_day",
                    "title": f"{num} by {time_col}",
                    "columns": [time_col, num],
                    "x": time_col,
                    "y": num,
                    "aggregation": "sum",
                    "purpose": "Identify patterns during the day."
                })

                chart_id += 1

        # -----------------------------------------------------
        # 6. CATEGORICAL + CATEGORICAL
        # -----------------------------------------------------

        if len(categorical) >= 2:

            for i in range(min(len(categorical), 5)):

                for j in range(i + 1, min(len(categorical), 5)):

                    c1 = categorical[i]
                    c2 = categorical[j]

                    if (
                        df[c1].nunique() <= 15
                        and df[c2].nunique() <= 15
                    ):

                        visualizations.append({
                            "id": chart_id,
                            "type": "category_heatmap",
                            "title": f"{c1} vs {c2}",
                            "columns": [c1, c2],
                            "x": c1,
                            "y": c2,
                            "aggregation": "count",
                            "purpose": "Explore relationships between categorical variables."
                        })

                        chart_id += 1

                        break

        # -----------------------------------------------------
        # 7. CATEGORY FREQUENCY
        # -----------------------------------------------------

        for cat in categorical[:6]:

            if df[cat].nunique() <= 20:

                visualizations.append({
                    "id": chart_id,
                    "type": "category_count",
                    "title": f"Frequency of {cat}",
                    "columns": [cat],
                    "x": cat,
                    "aggregation": "count",
                    "purpose": "Understand category frequency."
                })

                chart_id += 1

        return visualizations

    # ---------------------------------------------------------
    # SAVE MANIFEST
    # ---------------------------------------------------------

    def save_manifest(self, df):

        visualizations = self.create_visualizations(df)

        manifest = {
            "dataset_rows": int(len(df)),
            "dataset_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "detected_columns": self.detect_columns(df),
            "visualization_count": len(visualizations),
            "visualizations": visualizations
        }

        output_file = self.output_dir / "visualizations.json"

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2,
                ensure_ascii=False
            )

        return manifest

    # ---------------------------------------------------------
    # MAIN METHOD
    # ---------------------------------------------------------

    def run(self, df):

        print("Generating dynamic visualizations...")

        manifest = self.save_manifest(df)

        print(
            f"{manifest['visualization_count']} "
            "visualizations generated successfully."
        )

        return manifest