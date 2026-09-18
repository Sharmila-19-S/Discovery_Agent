import re
import pandas as pd


class HypothesisValidator:

    def __init__(self, df=None, profile=None):
        self.df = df
        self.profile = profile

        if df is not None:
            self.columns = list(df.columns)
        else:
            self.columns = []

    # ---------------------------------------------------------
    # COLUMN TYPE DETECTION
    # ---------------------------------------------------------

    def column_type(self, column):

        if self.df is None or column not in self.df.columns:
            return "unknown"

        series = self.df[column]

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Try detecting datetime safely
        if series.dtype == "object":

            sample = (
                series
                .dropna()
                .astype(str)
                .head(50)
            )

            if len(sample) > 0:

                try:
                    parsed = pd.to_datetime(
                        sample,
                        errors="coerce",
                        format="mixed"
                    )

                    if parsed.notna().mean() >= 0.7:
                        return "datetime"

                except Exception:
                    pass

        return "categorical"

    # ---------------------------------------------------------
    # IDENTIFIER DETECTION
    # ---------------------------------------------------------

    def is_identifier(self, column):

        name = str(column).lower()

        identifier_words = [
            "id",
            "identifier",
            "code",
            "key"
        ]

        parts = re.split(
            r"[_\-\s]+",
            name
        )

        if any(
            part in identifier_words
            for part in parts
        ):
            return True

        if (
            self.df is not None
            and column in self.df.columns
        ):

            series = self.df[column].dropna()

            if len(series) > 20:

                uniqueness = (
                    series.nunique()
                    / len(series)
                )

                if uniqueness >= 0.98:
                    return True

        return False

    # ---------------------------------------------------------
    # CHECK WHETHER VARIABLE IS ACTUALLY MENTIONED
    # ---------------------------------------------------------

    def variable_mentioned(
        self,
        variable,
        question
    ):

        if not question:
            return False

        question = str(
            question
        ).lower()

        variable = str(
            variable
        ).lower()

        # Exact column name
        if variable in question:
            return True

        # Check individual meaningful parts
        parts = re.split(
            r"[_\-\s]+",
            variable
        )

        meaningful_parts = [
            part
            for part in parts
            if len(part) >= 3
        ]

        if meaningful_parts:

            matches = sum(
                1
                for part in meaningful_parts
                if re.search(
                    r"\b"
                    + re.escape(part)
                    + r"\b",
                    question
                )
            )

            if matches >= len(
                meaningful_parts
            ):
                return True

        return False

    # ---------------------------------------------------------
    # NORMALIZE ANALYSIS METHOD
    # ---------------------------------------------------------

    def normalize_method(
        self,
        method
    ):

        method = str(
            method or ""
        ).lower().strip()

        if (
            "pearson" in method
            or "correlation" in method
        ):
            return "correlation"

        if (
            "regression" in method
            or "predict" in method
        ):
            return "regression"

        if "anova" in method:
            return "anova"

        if (
            "t-test" in method
            or "t test" in method
            or "ttest" in method
        ):
            return "t-test"

        if "chi" in method:
            return "chi-square"

        if (
            "trend" in method
            or "time series" in method
            or "time-series" in method
        ):
            return "time"

        if (
            "descriptive" in method
            or "summary" in method
        ):
            return "descriptive"

        return method

    # ---------------------------------------------------------
    # VALIDATE SINGLE HYPOTHESIS
    # ---------------------------------------------------------

    def validate_hypothesis(
        self,
        hypothesis
    ):

        if not isinstance(
            hypothesis,
            dict
        ):
            return None

        question = str(
            hypothesis.get(
                "question",
                ""
            )
        ).strip()

        variables = hypothesis.get(
            "variables",
            []
        )

        method = hypothesis.get(
            "analysis_method",
            hypothesis.get(
                "method",
                ""
            )
        )

        # Convert string variable to list
        if isinstance(
            variables,
            str
        ):
            variables = [variables]

        # Basic validation
        if not question:
            return None

        if not variables:
            return None

        # Keep only columns that actually exist
        variables = [
            str(variable)
            for variable in variables
            if str(variable) in self.columns
        ]

        if not variables:
            return None

        # Remove identifiers when other useful variables exist
        if len(variables) > 1:

            useful_variables = [
                variable
                for variable in variables
                if not self.is_identifier(variable)
            ]

            if useful_variables:
                variables = useful_variables

        # -----------------------------------------------------
        # QUESTION ↔ VARIABLE CONSISTENCY
        # -----------------------------------------------------

        mentioned = [
            variable
            for variable in variables
            if self.variable_mentioned(
                variable,
                question
            )
        ]

        # If the question mentions some, but not all,
        # of the selected variables, reject the hypothesis.
        if (
            mentioned
            and len(mentioned) != len(variables)
        ):
            return None

        # -----------------------------------------------------
        # METHOD VALIDATION
        # -----------------------------------------------------

        normalized_method = self.normalize_method(
            method
        )

        types = [
            self.column_type(variable)
            for variable in variables
        ]

        # -----------------------------------------------------
        # CORRELATION
        # -----------------------------------------------------

        if normalized_method == "correlation":

            if len(variables) != 2:
                return None

            if not all(
                column_type == "numeric"
                for column_type in types
            ):
                return None

        # -----------------------------------------------------
        # REGRESSION
        # -----------------------------------------------------

        elif normalized_method == "regression":

            if len(variables) != 2:
                return None

            if not all(
                column_type == "numeric"
                for column_type in types
            ):
                return None

        # -----------------------------------------------------
        # ANOVA
        # -----------------------------------------------------

        elif normalized_method == "anova":

            if len(variables) != 2:
                return None

            if "numeric" not in types:
                return None

            if "categorical" not in types:
                return None

        # -----------------------------------------------------
        # T-TEST
        # -----------------------------------------------------

        elif normalized_method == "t-test":

            if len(variables) != 2:
                return None

            if "numeric" not in types:
                return None

            if "categorical" not in types:
                return None

        # -----------------------------------------------------
        # CHI-SQUARE
        # -----------------------------------------------------

        elif normalized_method == "chi-square":

            if len(variables) != 2:
                return None

            if not all(
                column_type == "categorical"
                for column_type in types
            ):
                return None

        # -----------------------------------------------------
        # TIME SERIES / TREND
        # -----------------------------------------------------

        elif normalized_method == "time":

            if len(variables) != 2:
                return None

            if "numeric" not in types:
                return None

            if "datetime" not in types:
                return None

        # -----------------------------------------------------
        # DESCRIPTIVE
        # -----------------------------------------------------

        elif normalized_method == "descriptive":

            if len(variables) < 1:
                return None

        # -----------------------------------------------------
        # UNKNOWN METHOD
        # -----------------------------------------------------

        else:

            if not variables:
                return None

        # -----------------------------------------------------
        # CLEAN RESULT
        # -----------------------------------------------------

        cleaned = dict(hypothesis)

        cleaned["variables"] = variables

        cleaned["analysis_method"] = method

        cleaned.setdefault(
            "question",
            question
        )

        cleaned.setdefault(
            "expected_insight",
            ""
        )

        return cleaned

    # ---------------------------------------------------------
    # VALIDATE ALL HYPOTHESES
    # ---------------------------------------------------------

    def validate(
        self,
        hypotheses
    ):

        if not isinstance(
            hypotheses,
            list
        ):
            return []

        valid = []

        for hypothesis in hypotheses:

            cleaned = self.validate_hypothesis(
                hypothesis
            )

            if isinstance(
                cleaned,
                dict
            ):
                valid.append(cleaned)

        return valid