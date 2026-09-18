import math
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


class AnalysisEngine:

    def __init__(self, df=None):
        self.df = df

    # ---------------------------------------------------------
    # DATAFRAME SETUP
    # ---------------------------------------------------------

    def set_dataframe(self, df):
        self.df = df

    # ---------------------------------------------------------
    # NUMERIC PAIR
    # ---------------------------------------------------------

    def get_numeric_pair(self, variable_1, variable_2):

        if self.df is None:
            return None

        if (
            variable_1 not in self.df.columns
            or variable_2 not in self.df.columns
        ):
            return None

        data = self.df[
            [variable_1, variable_2]
        ].copy()

        data[variable_1] = pd.to_numeric(
            data[variable_1],
            errors="coerce"
        )

        data[variable_2] = pd.to_numeric(
            data[variable_2],
            errors="coerce"
        )

        data = data.dropna()

        if len(data) < 2:
            return None

        return data

    # ---------------------------------------------------------
    # COLUMN TYPE
    # ---------------------------------------------------------

    def column_type(self, column):

        if self.df is None:
            return "unknown"

        if column not in self.df.columns:
            return "unknown"

        series = self.df[column]

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

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
    # DESCRIPTIVE ANALYSIS
    # ---------------------------------------------------------

    def descriptive_analysis(self, variables):

        result = {
            "analysis_type": "Descriptive Statistics",
            "variables": [],
            "observations": 0
        }

        if self.df is None:
            return result

        for variable in variables:

            if variable not in self.df.columns:
                continue

            series = pd.to_numeric(
                self.df[variable],
                errors="coerce"
            ).dropna()

            if len(series) == 0:
                continue

            result["variables"].append({
                "variable": variable,
                "type": "numeric",
                "count": int(series.count()),
                "mean": round(float(series.mean()), 6),
                "median": round(float(series.median()), 6),
                "minimum": round(float(series.min()), 6),
                "maximum": round(float(series.max()), 6),
                "standard_deviation": round(
                    float(series.std()),
                    6
                )
            })

            result["observations"] = max(
                result["observations"],
                int(series.count())
            )

        return result

    # ---------------------------------------------------------
    # CORRELATION
    # ---------------------------------------------------------

    def correlation_analysis(
        self,
        variable_1,
        variable_2
    ):

        data = self.get_numeric_pair(
            variable_1,
            variable_2
        )

        if data is None:
            raise ValueError(
                "Unable to perform correlation analysis."
            )

        x = data[variable_1]
        y = data[variable_2]

        if x.nunique() < 2 or y.nunique() < 2:
            raise ValueError(
                "Correlation requires variation in both variables."
            )

        if stats is None:
            raise ImportError(
                "scipy is required for statistical analysis."
            )

        correlation, p_value = stats.pearsonr(
            x,
            y
        )

        correlation = float(correlation)
        p_value = float(p_value)

        absolute = abs(correlation)

        if absolute < 0.2:
            strength = "very weak"
        elif absolute < 0.4:
            strength = "weak"
        elif absolute < 0.6:
            strength = "moderate"
        elif absolute < 0.8:
            strength = "strong"
        else:
            strength = "very strong"

        if correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "none"

        return {
            "analysis_type": "Pearson correlation",
            "variable_1": variable_1,
            "variable_2": variable_2,
            "observations": int(len(data)),
            "correlation_coefficient": round(
                correlation,
                6
            ),
            "p_value": round(
                p_value,
                8
            ),
            "significance_level": 0.05,
            "significant": bool(
                p_value < 0.05
            ),
            "strength": strength,
            "direction": direction
        }

    # ---------------------------------------------------------
    # LINEAR REGRESSION
    # ---------------------------------------------------------

    def regression_analysis(
        self,
        predictor,
        target
    ):

        data = self.get_numeric_pair(
            predictor,
            target
        )

        if data is None:
            raise ValueError(
                "Unable to perform regression analysis."
            )

        x = data[predictor].astype(float)
        y = data[target].astype(float)

        if x.nunique() < 2:
            raise ValueError(
                "Regression predictor must contain variation."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Regression target must contain variation."
            )

        if stats is None:
            raise ImportError(
                "scipy is required for regression analysis."
            )

        regression = stats.linregress(
            x,
            y
        )

        slope = float(
            regression.slope
        )

        intercept = float(
            regression.intercept
        )

        r_value = float(
            regression.rvalue
        )

        r_squared = r_value ** 2

        p_value = float(
            regression.pvalue
        )

        standard_error = float(
            regression.stderr
        )

        predictions = (
            intercept
            + slope * x
        )

        residuals = (
            y - predictions
        )

        mae = float(
            np.mean(
                np.abs(residuals)
            )
        )

        mse = float(
            np.mean(
                residuals ** 2
            )
        )

        rmse = math.sqrt(mse)

        if slope > 0:
            direction = "positive"
        elif slope < 0:
            direction = "negative"
        else:
            direction = "none"

        if r_squared < 0.04:
            strength = "very weak"
        elif r_squared < 0.16:
            strength = "weak"
        elif r_squared < 0.36:
            strength = "moderate"
        elif r_squared < 0.64:
            strength = "strong"
        else:
            strength = "very strong"

        equation = (
            f"{target} = "
            f"{intercept:.6f} + "
            f"({slope:.6f} × {predictor})"
        )

        return {
            "analysis_type": "Linear regression",
            "predictor": predictor,
            "target": target,
            "observations": int(len(data)),
            "slope": round(
                slope,
                6
            ),
            "intercept": round(
                intercept,
                6
            ),
            "r_squared": round(
                r_squared,
                6
            ),
            "r_value": round(
                r_value,
                6
            ),
            "p_value": round(
                p_value,
                8
            ),
            "standard_error": round(
                standard_error,
                6
            ),
            "mae": round(
                mae,
                6
            ),
            "mse": round(
                mse,
                6
            ),
            "rmse": round(
                rmse,
                6
            ),
            "significance_level": 0.05,
            "significant": bool(
                p_value < 0.05
            ),
            "direction": direction,
            "strength": strength,
            "equation": equation
        }

    # ---------------------------------------------------------
    # ANOVA
    # ---------------------------------------------------------

    def anova_analysis(
        self,
        numeric_variable,
        categorical_variable
    ):

        if self.df is None:
            raise ValueError(
                "Dataset is not loaded."
            )

        if (
            numeric_variable not in self.df.columns
            or categorical_variable not in self.df.columns
        ):
            raise ValueError(
                "ANOVA variables were not found."
            )

        data = self.df[
            [
                numeric_variable,
                categorical_variable
            ]
        ].copy()

        data[numeric_variable] = pd.to_numeric(
            data[numeric_variable],
            errors="coerce"
        )

        data = data.dropna()

        groups = []

        group_statistics = {}

        for group_name, group_data in data.groupby(
            categorical_variable
        ):

            values = group_data[
                numeric_variable
            ].dropna()

            if len(values) == 0:
                continue

            group_values = values.astype(
                float
            ).tolist()

            groups.append(
                group_values
            )

            group_statistics[
                str(group_name)
            ] = {
                "count": int(len(values)),
                "mean": round(
                    float(values.mean()),
                    6
                ),
                "median": round(
                    float(values.median()),
                    6
                ),
                "minimum": round(
                    float(values.min()),
                    6
                ),
                "maximum": round(
                    float(values.max()),
                    6
                )
            }

        if len(groups) < 2:
            raise ValueError(
                "ANOVA requires at least two groups."
            )

        if stats is None:
            raise ImportError(
                "scipy is required for ANOVA."
            )

        f_statistic, p_value = stats.f_oneway(
            *groups
        )

        means = {
            group: values["mean"]
            for group, values
            in group_statistics.items()
        }

        highest_mean_group = max(
            means,
            key=means.get
        )

        lowest_mean_group = min(
            means,
            key=means.get
        )

        return {
            "analysis_type": "One-way ANOVA",
            "numeric_variable": numeric_variable,
            "categorical_variable": categorical_variable,
            "observations": int(len(data)),
            "number_of_groups": int(len(groups)),
            "group_statistics": group_statistics,
            "f_statistic": round(
                float(f_statistic),
                6
            ),
            "p_value": round(
                float(p_value),
                8
            ),
            "significance_level": 0.05,
            "significant": bool(
                p_value < 0.05
            ),
            "highest_mean_group": str(
                highest_mean_group
            ),
            "lowest_mean_group": str(
                lowest_mean_group
            )
        }

    # ---------------------------------------------------------
    # ANALYZE ONE HYPOTHESIS
    # ---------------------------------------------------------

    def analyze_hypothesis(
        self,
        hypothesis
    ):

        if not isinstance(
            hypothesis,
            dict
        ):
            return None

        hypothesis_id = hypothesis.get(
            "id",
            hypothesis.get(
                "hypothesis_id"
            )
        )

        question = hypothesis.get(
            "question",
            ""
        )

        variables = hypothesis.get(
            "variables",
            []
        )

        method = str(
            hypothesis.get(
                "analysis_method",
                hypothesis.get(
                    "method",
                    ""
                )
            )
        ).lower().strip()

        if isinstance(
            variables,
            str
        ):
            variables = [variables]

        variables = [
            str(variable)
            for variable in variables
        ]

        try:

            # Pearson correlation
            if (
                "pearson" in method
                or "correlation" in method
            ):

                if len(variables) != 2:
                    return None

                result = self.correlation_analysis(
                    variables[0],
                    variables[1]
                )

                conclusion = self.correlation_conclusion(
                    result
                )

            # Linear regression
            elif (
                "regression" in method
                or "predict" in method
            ):

                if len(variables) != 2:
                    return None

                result = self.regression_analysis(
                    predictor=variables[0],
                    target=variables[1]
                )

                conclusion = self.regression_conclusion(
                    result
                )

            # ANOVA
            elif "anova" in method:

                if len(variables) != 2:
                    return None

                numeric_variable = None
                categorical_variable = None

                for variable in variables:

                    variable_type = self.column_type(
                        variable
                    )

                    if variable_type == "numeric":
                        numeric_variable = variable

                    elif variable_type == "categorical":
                        categorical_variable = variable

                if (
                    numeric_variable is None
                    or categorical_variable is None
                ):
                    return None

                result = self.anova_analysis(
                    numeric_variable,
                    categorical_variable
                )

                conclusion = self.anova_conclusion(
                    result
                )

            # Descriptive
            elif (
                "descriptive" in method
                or "summary" in method
            ):

                result = self.descriptive_analysis(
                    variables
                )

                conclusion = (
                    "Descriptive statistics "
                    "were calculated for the "
                    "selected variables."
                )

            else:
                return None

            return {
                "hypothesis_id": hypothesis_id,
                "question": question,
                "variables": variables,
                "method": method,
                "result": result,
                "conclusion": conclusion,
                "expected_insight": hypothesis.get(
                    "expected_insight",
                    ""
                ),
                "status": "success"
            }

        except Exception as error:

            return {
                "hypothesis_id": hypothesis_id,
                "question": question,
                "variables": variables,
                "method": method,
                "result": {},
                "conclusion": (
                    f"Analysis could not be completed: "
                    f"{str(error)}"
                ),
                "expected_insight": hypothesis.get(
                    "expected_insight",
                    ""
                ),
                "status": "error"
            }

    # ---------------------------------------------------------
    # CONCLUSIONS
    # ---------------------------------------------------------

    def correlation_conclusion(
        self,
        result
    ):

        if result["significant"]:

            return (
                f"The Pearson correlation between "
                f"{result['variable_1']} and "
                f"{result['variable_2']} is "
                f"{result['correlation_coefficient']}, "
                f"indicating a "
                f"{result['strength']} "
                f"{result['direction']} relationship. "
                f"The p-value is "
                f"{result['p_value']}, so the "
                f"relationship is statistically "
                f"significant at the 0.05 significance "
                f"level."
            )

        return (
            f"The Pearson correlation between "
            f"{result['variable_1']} and "
            f"{result['variable_2']} is "
            f"{result['correlation_coefficient']}, "
            f"indicating a "
            f"{result['strength']} "
            f"{result['direction']} relationship. "
            f"The p-value is "
            f"{result['p_value']}, so the "
            f"relationship is not statistically "
            f"significant at the 0.05 significance "
            f"level."
        )

    def regression_conclusion(
        self,
        result
    ):

        if result["significant"]:

            return (
                f"Linear regression indicates that "
                f"{result['predictor']} has a "
                f"statistically significant "
                f"{result['direction']} association "
                f"with {result['target']}. "
                f"The estimated slope is "
                f"{result['slope']}, with "
                f"R² = {result['r_squared']}. "
                f"The p-value is "
                f"{result['p_value']}, which is "
                f"below the 0.05 significance level."
            )

        return (
            f"Linear regression indicates that "
            f"{result['predictor']} does not have a "
            f"statistically significant linear "
            f"association with {result['target']}. "
            f"The estimated slope is "
            f"{result['slope']}, with "
            f"R² = {result['r_squared']}. "
            f"The p-value is "
            f"{result['p_value']}, which is "
            f"not below the 0.05 significance level."
        )

    def anova_conclusion(
        self,
        result
    ):

        means = result[
            "group_statistics"
        ]

        highest = result[
            "highest_mean_group"
        ]

        lowest = result[
            "lowest_mean_group"
        ]

        highest_mean = means[
            highest
        ]["mean"]

        lowest_mean = means[
            lowest
        ]["mean"]

        if result["significant"]:

            return (
                f"There is a statistically significant "
                f"difference in average "
                f"{result['numeric_variable']} "
                f"across "
                f"{result['categorical_variable']} "
                f"groups "
                f"(F={result['f_statistic']}, "
                f"p={result['p_value']}). "
                f"The highest observed mean is for "
                f"'{highest}' ({highest_mean}), "
                f"while the lowest is for "
                f"'{lowest}' ({lowest_mean})."
            )

        return (
            f"There is no statistically significant "
            f"difference in average "
            f"{result['numeric_variable']} "
            f"across "
            f"{result['categorical_variable']} "
            f"groups "
            f"(F={result['f_statistic']}, "
            f"p={result['p_value']}). "
            f"The observed group means range from "
            f"{lowest_mean} to "
            f"{highest_mean}."
        )

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    def run(
        self,
        df=None,
        hypotheses=None
    ):

        # Support both:
        # engine.run(df, hypotheses)
        # engine.run(df=df, hypotheses=hypotheses)

        if df is not None:
            self.df = df

        if hypotheses is None:
            hypotheses = []

        if self.df is None:
            raise ValueError(
                "Dataset is not loaded."
            )

        if not isinstance(
            hypotheses,
            list
        ):
            return []

        results = []

        for hypothesis in hypotheses:

            result = self.analyze_hypothesis(
                hypothesis
            )

            if isinstance(
                result,
                dict
            ):
                results.append(result)

        return results

    # ---------------------------------------------------------
    # ALIAS
    # ---------------------------------------------------------

    def analyze(
        self,
        df=None,
        hypotheses=None
    ):

        return self.run(
            df=df,
            hypotheses=hypotheses
        )