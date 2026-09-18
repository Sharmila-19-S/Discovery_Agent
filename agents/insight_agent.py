import json
from pathlib import Path

from models.ollama_client import generate


class InsightAgent:

    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # NORMALIZE ANALYSIS RESULTS
    # =========================================================

    @staticmethod
    def normalize_results(analysis_results):

        if analysis_results is None:
            return []

        if isinstance(analysis_results, str):
            try:
                analysis_results = json.loads(
                    analysis_results
                )
            except json.JSONDecodeError:
                return []

        if isinstance(analysis_results, dict):

            if "results" in analysis_results:
                analysis_results = analysis_results["results"]

            elif "analysis_results" in analysis_results:
                analysis_results = analysis_results[
                    "analysis_results"
                ]

            else:
                analysis_results = [
                    analysis_results
                ]

        if not isinstance(analysis_results, list):
            return []

        return [
            item
            for item in analysis_results
            if isinstance(item, dict)
        ]

    # =========================================================
    # EXTRACT STATISTICAL INFORMATION
    # =========================================================

    @staticmethod
    def extract_statistics(result):

        analysis = result.get("result", {})

        if not isinstance(analysis, dict):
            analysis = {}

        method = str(
            result.get(
                "method",
                result.get(
                    "analysis_method",
                    analysis.get(
                        "analysis_type",
                        ""
                    )
                )
            )
        ).lower()

        p_value = analysis.get("p_value")

        if p_value is None:
            p_value = result.get("p_value")

        try:
            if p_value is not None:
                p_value = float(p_value)
        except (TypeError, ValueError):
            p_value = None

        significance_level = analysis.get(
            "significance_level",
            0.05
        )

        try:
            significance_level = float(
                significance_level
            )
        except (TypeError, ValueError):
            significance_level = 0.05

        significant = analysis.get(
            "significant"
        )

        if significant is None and p_value is not None:
            significant = (
                p_value < significance_level
            )

        if significant is True:
            significance = "Statistically significant"

        elif significant is False:
            significance = "Not statistically significant"

        else:
            significance = "Not available"

        return {
            "method": method,
            "p_value": p_value,
            "significance_level": significance_level,
            "significant": significant,
            "significance": significance,
            "analysis": analysis
        }

    # =========================================================
    # FORMAT P-VALUE
    # =========================================================

    @staticmethod
    def format_p_value(p_value):

        if p_value is None:
            return "not available"

        try:
            p_value = float(p_value)
        except (TypeError, ValueError):
            return "not available"

        if p_value < 0.001:
            return "p < 0.001"

        return f"p = {p_value:.4f}"

    # =========================================================
    # BUILD STATISTICAL CONTEXT
    # =========================================================

    @staticmethod
    def build_context(result):

        statistics = InsightAgent.extract_statistics(
            result
        )

        analysis = statistics["analysis"]
        method = statistics["method"]

        context = {
            "hypothesis_id": result.get(
                "hypothesis_id"
            ),
            "question": result.get(
                "question",
                ""
            ),
            "method": result.get(
                "method",
                ""
            ),
            "conclusion": result.get(
                "conclusion",
                ""
            ),
            "statistics": statistics
        }

        # -----------------------------------------------------
        # CORRELATION
        # -----------------------------------------------------

        if (
            "correlation" in method
            or "pearson" in method
        ):

            context["correlation_coefficient"] = (
                analysis.get(
                    "correlation_coefficient"
                )
            )

            context["strength"] = analysis.get(
                "strength"
            )

            context["direction"] = analysis.get(
                "direction"
            )

        # -----------------------------------------------------
        # REGRESSION
        # -----------------------------------------------------

        elif "regression" in method:

            context["predictor"] = analysis.get(
                "predictor"
            )

            context["target"] = analysis.get(
                "target"
            )

            context["slope"] = analysis.get(
                "slope"
            )

            context["intercept"] = analysis.get(
                "intercept"
            )

            context["r_squared"] = analysis.get(
                "r_squared"
            )

            context["r_value"] = analysis.get(
                "r_value"
            )

            context["rmse"] = analysis.get(
                "rmse"
            )

            context["direction"] = analysis.get(
                "direction"
            )

            context["strength"] = analysis.get(
                "strength"
            )

            context["equation"] = analysis.get(
                "equation"
            )

        # -----------------------------------------------------
        # ANOVA
        # -----------------------------------------------------

        elif "anova" in method:

            context["numeric_variable"] = analysis.get(
                "numeric_variable"
            )

            context["categorical_variable"] = analysis.get(
                "categorical_variable"
            )

            context["number_of_groups"] = analysis.get(
                "number_of_groups"
            )

            context["f_statistic"] = analysis.get(
                "f_statistic"
            )

            context["highest_mean_group"] = analysis.get(
                "highest_mean_group"
            )

            context["lowest_mean_group"] = analysis.get(
                "lowest_mean_group"
            )

            context["group_statistics"] = analysis.get(
                "group_statistics"
            )

        return context

    # =========================================================
    # GENERATE INSIGHTS
    # =========================================================

    def generate_insights(
        self,
        analysis_results
    ):

        results = self.normalize_results(
            analysis_results
        )

        if not results:

            output = {
                "insight_count": 0,
                "insights": []
            }

            self.save_output(output)

            return output

        # =====================================================
        # PREPARE COMPACT RESULTS
        # =====================================================

        compact_results = []

        for index, result in enumerate(
            results,
            start=1
        ):

            compact_results.append(
                self.build_context(result)
            )

        results_text = json.dumps(
            compact_results,
            indent=2,
            ensure_ascii=False,
            default=str
        )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = """
You are an expert data analyst.

You will receive statistical analysis results
from an automated data analysis system.

Generate exactly one useful insight for each
analysis result.

IMPORTANT RULES:

1. Use ONLY the supplied analysis results.
2. Never invent variables.
3. Never invent statistical values.
4. Never change p-values.
5. Never change correlation coefficients.
6. Never change regression statistics.
7. Never change ANOVA statistics.
8. Preserve the direction and strength of relationships.
9. Clearly explain statistical significance.
10. A statistically significant result does NOT automatically mean a strong relationship.
11. For regression, mention R-squared when available.
12. For regression, distinguish statistical significance from predictive strength.
13. For ANOVA, mention whether group means differ significantly.
14. For ANOVA, mention highest and lowest mean groups when supplied.
15. If p < 0.001, write "p < 0.001" rather than "p = 0.0".
16. Keep insights concise and understandable.
17. Do not create recommendations unless directly supported.
18. Do not create visualizations.
19. Return valid JSON only.
20. Generate exactly one insight for each supplied analysis result.

Return exactly:

{
    "insights": [
        {
            "hypothesis_id": 1,
            "title": "Short descriptive title",
            "insight": "Clear explanation of the finding.",
            "status": "success"
        }
    ]
}

For failed analysis:

{
    "hypothesis_id": 1,
    "title": "Analysis unavailable",
    "insight": "Explain the supplied error.",
    "status": "failed"
}

Return ONLY valid JSON.
"""

        # =====================================================
        # USER PROMPT
        # =====================================================

        user_prompt = f"""
Statistical analysis results:

{results_text}

Generate exactly one insight for each supplied result.

Return ONLY valid JSON.
"""

        # =====================================================
        # OLLAMA
        # =====================================================

        response = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )

        if not response:
            raise RuntimeError(
                "Ollama returned an empty insight response."
            )

        response = response.strip()

        # =====================================================
        # REMOVE MARKDOWN FENCES
        # =====================================================

        if response.startswith("```json"):
            response = response[
                len("```json"):
            ]

        elif response.startswith("```"):
            response = response[
                len("```"):
            ]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        # =====================================================
        # PARSE JSON
        # =====================================================

        try:

            parsed = json.loads(
                response
            )

        except json.JSONDecodeError:

            start = response.find("{")
            end = response.rfind("}")

            if (
                start == -1
                or end == -1
            ):
                parsed = {}

            else:

                try:
                    parsed = json.loads(
                        response[
                            start:end + 1
                        ]
                    )

                except json.JSONDecodeError:
                    parsed = {}

        # =====================================================
        # EXTRACT OLLAMA INSIGHTS
        # =====================================================

        if isinstance(parsed, dict):

            insights = parsed.get(
                "insights",
                []
            )

        elif isinstance(parsed, list):

            insights = parsed

        else:

            insights = []

        if not isinstance(insights, list):
            insights = []

        # =====================================================
        # MAP OLLAMA OUTPUT
        # =====================================================

        generated_by_id = {}

        for index, insight in enumerate(
            insights,
            start=1
        ):

            if not isinstance(
                insight,
                dict
            ):
                continue

            hypothesis_id = insight.get(
                "hypothesis_id",
                index
            )

            generated_by_id[
                str(hypothesis_id)
            ] = insight

        # =====================================================
        # FINAL NORMALIZATION
        # =====================================================

        normalized = []

        for index, result in enumerate(
            results,
            start=1
        ):

            hypothesis_id = result.get(
                "hypothesis_id",
                index
            )

            generated = generated_by_id.get(
                str(hypothesis_id),
                {}
            )

            statistics = self.extract_statistics(
                result
            )

            analysis = statistics["analysis"]

            method = statistics["method"]

            p_value = statistics["p_value"]

            significance = statistics[
                "significance"
            ]

            # =================================================
            # TITLE
            # =================================================

            title = generated.get(
                "title",
                ""
            )

            if not title:

                question = result.get(
                    "question",
                    ""
                )

                if question:
                    title = question

                else:
                    title = (
                        f"Insight {index}"
                    )

            # =================================================
            # INSIGHT
            # =================================================

            insight_text = generated.get(
                "insight",
                ""
            )

            if not insight_text:
                insight_text = result.get(
                    "conclusion",
                    ""
                )

            # =================================================
            # REGRESSION QUALITY
            # =================================================

            if "regression" in method:

                predictor = analysis.get(
                    "predictor"
                )

                target = analysis.get(
                    "target"
                )

                slope = analysis.get(
                    "slope"
                )

                r_squared = analysis.get(
                    "r_squared"
                )

                direction = analysis.get(
                    "direction"
                )

                strength = analysis.get(
                    "strength"
                )

                if (
                    predictor is not None
                    and target is not None
                    and r_squared is not None
                ):

                    try:

                        r2_percent = (
                            float(r_squared)
                            * 100
                        )

                        direction_text = (
                            f"{direction} "
                            if direction
                            else ""
                        )

                        strength_text = (
                            f"{strength} "
                            if strength
                            else ""
                        )

                        insight_text = (
                            f"{predictor} has a "
                            f"{strength_text}"
                            f"{direction_text}"
                            f"association with "
                            f"{target}, with a "
                            f"slope of "
                            f"{float(slope):.4f} "
                            f"and "
                            f"{self.format_p_value(p_value)}. "
                            f"The model explains "
                            f"approximately "
                            f"{r2_percent:.2f}% "
                            f"of the variation "
                            f"in {target}, "
                            f"indicating "
                            f"weak predictive "
                            f"power."
                        )

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

            # =================================================
            # CORRELATION
            # =================================================

            elif (
                "correlation" in method
                or "pearson" in method
            ):

                coefficient = analysis.get(
                    "correlation_coefficient"
                )

                strength = analysis.get(
                    "strength"
                )

                direction = analysis.get(
                    "direction"
                )

                variable_1 = analysis.get(
                    "variable_1"
                )

                variable_2 = analysis.get(
                    "variable_2"
                )

                if coefficient is not None:

                    relationship = ""

                    if strength:
                        relationship += (
                            f"{strength} "
                        )

                    if direction:
                        relationship += (
                            f"{direction} "
                        )

                    relationship += (
                        "relationship"
                    )

                    insight_text = (
                        f"There is a "
                        f"{relationship} "
                        f"between "
                        f"{variable_1} and "
                        f"{variable_2}, "
                        f"with a Pearson "
                        f"correlation "
                        f"coefficient of "
                        f"{float(coefficient):.4f} "
                        f"and "
                        f"{self.format_p_value(p_value)}."
                    )

            # =================================================
            # ANOVA
            # =================================================

            elif "anova" in method:

                numeric_variable = analysis.get(
                    "numeric_variable"
                )

                categorical_variable = analysis.get(
                    "categorical_variable"
                )

                highest_group = analysis.get(
                    "highest_mean_group"
                )

                lowest_group = analysis.get(
                    "lowest_mean_group"
                )

                highest_mean = None
                lowest_mean = None

                group_statistics = analysis.get(
                    "group_statistics",
                    {}
                )

                if isinstance(
                    group_statistics,
                    dict
                ):

                    if highest_group in group_statistics:

                        highest_mean = group_statistics[
                            highest_group
                        ].get("mean")

                    if lowest_group in group_statistics:

                        lowest_mean = group_statistics[
                            lowest_group
                        ].get("mean")

                p_text = self.format_p_value(
                    p_value
                )

                if significance == (
                    "Statistically significant"
                ):

                    insight_text = (
                        f"There is a statistically "
                        f"significant difference "
                        f"in average "
                        f"{numeric_variable} "
                        f"across "
                        f"{categorical_variable} "
                        f"groups "
                        f"({p_text})."
                    )

                    if (
                        highest_group is not None
                        and highest_mean is not None
                    ):

                        insight_text += (
                            f" The highest "
                            f"mean is for "
                            f"'{highest_group}' "
                            f"({float(highest_mean):.4f})"
                        )

                    if (
                        lowest_group is not None
                        and lowest_mean is not None
                    ):

                        insight_text += (
                            f" and the lowest "
                            f"is for "
                            f"'{lowest_group}' "
                            f"({float(lowest_mean):.4f})."
                        )

                elif significance == (
                    "Not statistically significant"
                ):

                    insight_text = (
                        f"There is no "
                        f"statistically "
                        f"significant difference "
                        f"in average "
                        f"{numeric_variable} "
                        f"across "
                        f"{categorical_variable} "
                        f"groups "
                        f"({p_text})."
                    )

            # =================================================
            # FINAL OBJECT
            # =================================================

            normalized.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "title": title,
                    "insight": str(
                        insight_text
                    ),
                    "significance": significance,
                    "p_value": p_value,
                    "status": result.get(
                        "status",
                        "success"
                    )
                }
            )

        # =====================================================
        # FINAL OUTPUT
        # =====================================================

        output = {
            "insight_count": len(
                normalized
            ),
            "insights": normalized
        }

        # =====================================================
        # SAVE
        # =====================================================

        self.save_output(
            output
        )

        return output

    # =========================================================
    # SAVE OUTPUT
    # =========================================================

    def save_output(
        self,
        output
    ):

        json_file = (
            self.output_dir
            / "insights.json"
        )

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        text_file = (
            self.output_dir
            / "insights.txt"
        )

        insights = output.get(
            "insights",
            []
        )

        with open(
            text_file,
            "w",
            encoding="utf-8"
        ) as file:

            for index, insight in enumerate(
                insights,
                start=1
            ):

                file.write(
                    f"Insight {index}\n"
                )

                file.write(
                    f"Title: "
                    f"{insight.get('title', '')}\n"
                )

                file.write(
                    f"Insight: "
                    f"{insight.get('insight', '')}\n"
                )

                file.write(
                    f"Significance: "
                    f"{insight.get('significance', '')}\n"
                )

                file.write(
                    f"p-value: "
                    f"{insight.get('p_value', None)}\n"
                )

                file.write(
                    f"Status: "
                    f"{insight.get('status', '')}\n"
                )

                file.write("\n")

    # =========================================================
    # RUN
    # =========================================================

    def run(
        self,
        analysis_results
    ):

        print(
            "Generating insights..."
        )

        output = self.generate_insights(
            analysis_results
        )

        print(
            f"{output.get('insight_count', 0)} "
            "insights generated successfully."
        )

        return output

    # =========================================================
    # STATIC COMPATIBILITY METHOD
    # =========================================================

    @staticmethod
    def generate(
        analysis_results,
        output_dir="outputs"
    ):

        agent = InsightAgent(
            output_dir=output_dir
        )

        return agent.generate_insights(
            analysis_results
        )