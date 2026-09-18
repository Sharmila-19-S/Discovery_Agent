import json
import os
import re
import pandas as pd


class HypothesisAgent:

    # =========================================================
    # IDENTIFIER DETECTION
    # =========================================================

    @staticmethod
    def _is_identifier(column):

        name = str(column).lower().strip()

        identifier_names = {
            "id",
            "identifier",
            "uuid",
            "code",
            "key",
            "transaction_id",
            "customer_id",
            "user_id",
            "order_id",
            "product_id",
            "store_id",
            "employee_id",
            "account_id"
        }

        if name in identifier_names:
            return True

        if (
            name.endswith("_id")
            or name.endswith("_code")
            or name.endswith("_uuid")
            or name.endswith("_key")
        ):
            return True

        return False

    # =========================================================
    # COLUMN DETECTION FROM DATAFRAME
    # =========================================================

    @staticmethod
    def _detect_from_dataframe(df):

        numeric = []
        categorical = []
        datetime_cols = []
        time_cols = []

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "numeric": [],
                "categorical": [],
                "datetime": [],
                "time": []
            }

        for column in df.columns:

            if HypothesisAgent._is_identifier(column):
                continue

            series = df[column]

            if pd.api.types.is_numeric_dtype(series):

                numeric.append(column)

                continue

            if pd.api.types.is_datetime64_any_dtype(series):

                datetime_cols.append(column)

                continue

            # Try datetime detection for object columns
            if series.dtype == "object":

                sample = (
                    series
                    .dropna()
                    .astype(str)
                    .head(100)
                )

                if len(sample) > 0:

                    try:

                        parsed = pd.to_datetime(
                            sample,
                            errors="coerce",
                            format="mixed"
                        )

                        if (
                            parsed.notna().mean()
                            >= 0.70
                        ):

                            datetime_cols.append(
                                column
                            )

                            continue

                    except Exception:
                        pass

            # Detect time-like columns
            name = str(column).lower()

            if any(
                word in name
                for word in [
                    "time",
                    "hour",
                    "minute",
                    "timestamp"
                ]
            ):

                time_cols.append(column)

                continue

            categorical.append(column)

        return {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols,
            "time": time_cols
        }

    # =========================================================
    # COLUMN DETECTION FROM PROFILE
    # =========================================================

    @staticmethod
    def _detect_from_profile(profile):

        if not isinstance(
            profile,
            dict
        ):

            return {
                "numeric": [],
                "categorical": [],
                "datetime": [],
                "time": []
            }

        numeric = list(
            profile.get(
                "numeric_columns",
                []
            )
            or []
        )

        categorical = list(
            profile.get(
                "categorical_columns",
                []
            )
            or []
        )

        datetime_cols = list(
            profile.get(
                "date_columns",
                []
            )
            or profile.get(
                "datetime_columns",
                []
            )
            or []
        )

        time_cols = list(
            profile.get(
                "time_columns",
                []
            )
            or profile.get(
                "time",
                []
            )
            or profile.get(
                "time_like_columns",
                []
            )
            or []
        )

        numeric = [
            c for c in numeric
            if not HypothesisAgent._is_identifier(c)
        ]

        categorical = [
            c for c in categorical
            if not HypothesisAgent._is_identifier(c)
        ]

        datetime_cols = [
            c for c in datetime_cols
            if not HypothesisAgent._is_identifier(c)
        ]

        time_cols = [
            c for c in time_cols
            if not HypothesisAgent._is_identifier(c)
        ]

        return {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols,
            "time": time_cols
        }

    # =========================================================
    # HUMAN-READABLE COLUMN NAME
    # =========================================================

    @staticmethod
    def _pretty_name(column):

        name = str(column)

        name = re.sub(
            r"[_\-]+",
            " ",
            name
        )

        name = re.sub(
            r"\s+",
            " ",
            name
        ).strip()

        return name.lower()

    # =========================================================
    # BUILD HUMAN-UNDERSTANDABLE HYPOTHESES
    # =========================================================

    @staticmethod
    def _build_hypotheses(
        detected,
        num_hypotheses=5
    ):

        numeric = detected.get(
            "numeric",
            []
        )

        categorical = detected.get(
            "categorical",
            []
        )

        datetime_cols = detected.get(
            "datetime",
            []
        )

        time_cols = detected.get(
            "time",
            []
        )

        hypotheses = []

        # =====================================================
        # HYPOTHESIS 1
        # NUMERIC RELATIONSHIP
        # =====================================================

        if len(numeric) >= 2:

            x = numeric[0]
            y = numeric[1]

            x_name = HypothesisAgent._pretty_name(x)
            y_name = HypothesisAgent._pretty_name(y)

            hypotheses.append({

                "id": 1,

                "question":
                    f"Does {x_name} change when {y_name} changes?",

                "simple_explanation":
                    (
                        f"We want to find out whether "
                        f"{x_name} and {y_name} are related "
                        f"to each other."
                    ),

                "solution":
                    (
                        f"We compare the values of "
                        f"{x_name} and {y_name} using "
                        f"correlation analysis."
                    ),

                "variables": [
                    x,
                    y
                ],

                "analysis_method":
                    "Pearson correlation",

                "expected_insight":
                    (
                        f"Find out whether there is a "
                        f"relationship between "
                        f"{x_name} and {y_name}."
                    )
            })

        # =====================================================
        # HYPOTHESIS 2
        # PREDICTION
        # =====================================================

        if (
            len(numeric) >= 2
            and len(hypotheses) < num_hypotheses
        ):

            x = numeric[0]
            y = numeric[1]

            x_name = HypothesisAgent._pretty_name(x)
            y_name = HypothesisAgent._pretty_name(y)

            hypotheses.append({

                "id":
                    len(hypotheses) + 1,

                "question":
                    f"Can we use {x_name} to predict {y_name}?",

                "simple_explanation":
                    (
                        f"We want to check whether "
                        f"{x_name} can help us estimate "
                        f"the value of {y_name}."
                    ),

                "solution":
                    (
                        f"We use linear regression to "
                        f"measure how much {y_name} changes "
                        f"when {x_name} changes."
                    ),

                "variables": [
                    x,
                    y
                ],

                "analysis_method":
                    "Linear regression",

                "expected_insight":
                    (
                        f"Determine whether {x_name} "
                        f"can help predict {y_name}."
                    )
            })

        # =====================================================
        # HYPOTHESIS 3+
        # CATEGORY COMPARISON
        # =====================================================

        for category in categorical:

            if (
                not numeric
                or len(hypotheses)
                >= num_hypotheses
            ):
                break

            measure = numeric[0]

            category_name = (
                HypothesisAgent._pretty_name(
                    category
                )
            )

            measure_name = (
                HypothesisAgent._pretty_name(
                    measure
                )
            )

            hypotheses.append({

                "id":
                    len(hypotheses) + 1,

                "question":
                    (
                        f"Does the average "
                        f"{measure_name} differ between "
                        f"{category_name} groups?"
                    ),

                "simple_explanation":
                    (
                        f"We want to see whether different "
                        f"{category_name} groups have "
                        f"different average "
                        f"{measure_name}."
                    ),

                "solution":
                    (
                        f"We compare the average "
                        f"{measure_name} across the "
                        f"different {category_name} groups "
                        f"using one-way ANOVA."
                    ),

                "variables": [
                    measure,
                    category
                ],

                "analysis_method":
                    "One-way ANOVA",

                "expected_insight":
                    (
                        f"Find out whether the average "
                        f"{measure_name} changes across "
                        f"{category_name} groups."
                    )
            })

        # =====================================================
        # TIME SERIES
        # =====================================================

        time_candidates = (
            time_cols
            + datetime_cols
        )

        if (
            time_candidates
            and numeric
            and len(hypotheses)
            < num_hypotheses
        ):

            time_column = time_candidates[0]
            measure = numeric[0]

            time_name = (
                HypothesisAgent._pretty_name(
                    time_column
                )
            )

            measure_name = (
                HypothesisAgent._pretty_name(
                    measure
                )
            )

            hypotheses.append({

                "id":
                    len(hypotheses) + 1,

                "question":
                    (
                        f"How does {measure_name} "
                        f"change over {time_name}?"
                    ),

                "simple_explanation":
                    (
                        f"We want to understand whether "
                        f"{measure_name} increases, decreases, "
                        f"or follows a pattern over time."
                    ),

                "solution":
                    (
                        f"We arrange {measure_name} by "
                        f"{time_name} and analyze the trend "
                        f"using time-series analysis."
                    ),

                "variables": [
                    time_column,
                    measure
                ],

                "analysis_method":
                    "Time series analysis",

                "expected_insight":
                    (
                        f"Identify important trends or "
                        f"changes in {measure_name} over "
                        f"{time_name}."
                    )
            })

        # =====================================================
        # CATEGORICAL ASSOCIATION
        # =====================================================

        if (
            len(categorical) >= 2
            and len(hypotheses)
            < num_hypotheses
        ):

            c1 = categorical[0]
            c2 = categorical[1]

            c1_name = (
                HypothesisAgent._pretty_name(
                    c1
                )
            )

            c2_name = (
                HypothesisAgent._pretty_name(
                    c2
                )
            )

            hypotheses.append({

                "id":
                    len(hypotheses) + 1,

                "question":
                    (
                        f"Are {c1_name} and "
                        f"{c2_name} connected?"
                    ),

                "simple_explanation":
                    (
                        f"We want to find out whether "
                        f"the groups in {c1_name} are "
                        f"related to the groups in "
                        f"{c2_name}."
                    ),

                "solution":
                    (
                        f"We compare the frequency of "
                        f"the category combinations "
                        f"using a chi-square test."
                    ),

                "variables": [
                    c1,
                    c2
                ],

                "analysis_method":
                    "Chi-square test",

                "expected_insight":
                    (
                        f"Determine whether {c1_name} "
                        f"and {c2_name} are associated."
                    )
            })

        # =====================================================
        # DESCRIPTIVE FALLBACK
        # =====================================================

        if (
            not hypotheses
            and numeric
        ):

            for measure in numeric:

                if len(hypotheses) >= num_hypotheses:
                    break

                measure_name = (
                    HypothesisAgent._pretty_name(
                        measure
                    )
                )

                hypotheses.append({

                    "id":
                        len(hypotheses) + 1,

                    "question":
                        (
                            f"What does the "
                            f"{measure_name} data look like?"
                        ),

                    "simple_explanation":
                        (
                            f"We want to understand the "
                            f"typical value and spread of "
                            f"{measure_name}."
                        ),

                    "solution":
                        (
                            f"We calculate basic descriptive "
                            f"statistics such as the mean, "
                            f"median, minimum and maximum."
                        ),

                    "variables": [
                        measure
                    ],

                    "analysis_method":
                        "Descriptive statistics",

                    "expected_insight":
                        (
                            f"Summarize the distribution "
                            f"of {measure_name}."
                        )
                })

        return hypotheses[
            :num_hypotheses
        ]

    # =========================================================
    # GENERATE
    # =========================================================

    @staticmethod
    def generate(
        df=None,
        metadata=None,
        profile=None,
        num_hypotheses=5
    ):

        print(
            "Generating dataset-specific hypotheses..."
        )

        # -----------------------------------------------------
        # Prefer actual DataFrame
        # -----------------------------------------------------

        if isinstance(
            df,
            pd.DataFrame
        ):

            detected = (
                HypothesisAgent
                ._detect_from_dataframe(df)
            )

        else:

            detected = (
                HypothesisAgent
                ._detect_from_profile(profile)
            )

        # -----------------------------------------------------
        # Build hypotheses
        # -----------------------------------------------------

        hypotheses = (
            HypothesisAgent
            ._build_hypotheses(
                detected,
                num_hypotheses
            )
        )

        if not hypotheses:

            raise RuntimeError(
                "Unable to generate hypotheses "
                "from the dataset."
            )

        # -----------------------------------------------------
        # Save output
        # -----------------------------------------------------

        output_dir = "outputs"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        output_path = (
            os.path.join(
                output_dir,
                "hypotheses.json"
            )
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "hypotheses":
                        hypotheses
                },
                file,
                indent=4,
                ensure_ascii=False
            )

        # -----------------------------------------------------
        # Save readable text
        # -----------------------------------------------------

        text_path = (
            os.path.join(
                output_dir,
                "hypotheses.txt"
            )
        )

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as file:

            for hypothesis in hypotheses:

                file.write(
                    f"HYPOTHESIS {hypothesis['id']}\n"
                )

                file.write(
                    f"Question: "
                    f"{hypothesis['question']}\n"
                )

                file.write(
                    f"What are we finding?: "
                    f"{hypothesis['simple_explanation']}\n"
                )

                file.write(
                    f"How will we solve it?: "
                    f"{hypothesis['solution']}\n"
                )

                file.write(
                    f"Variables: "
                    f"{', '.join(map(str, hypothesis['variables']))}\n"
                )

                file.write(
                    f"Method: "
                    f"{hypothesis['analysis_method']}\n"
                )

                file.write(
                    f"Expected insight: "
                    f"{hypothesis['expected_insight']}\n"
                )

                file.write(
                    "\n"
                )

        print(
            f"{len(hypotheses)} hypotheses generated successfully."
        )

        return hypotheses