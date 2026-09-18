import json
import os


class ExecutionAgent:

    @staticmethod
    def execute(code, df):

        print("\nExecuting generated analysis...")

        # Environment available to generated code
        execution_environment = {
            "df": df
        }

        try:

            exec(
                code,
                execution_environment
            )

            # -------------------------------------------------
            # Collect only meaningful analysis results
            # -------------------------------------------------

            results = {}

            result_names = [
                "result_1",
                "result_2",
                "result_3",
                "result_4",
                "result_5",
                "total_sales"
            ]

            for name in result_names:

                if name in execution_environment:

                    value = execution_environment[name]

                    # Convert pandas objects to strings
                    # so they can be saved safely as JSON
                    if hasattr(value, "to_dict"):

                        try:
                            value = value.to_dict()

                        except Exception:
                            value = str(value)

                    results[name] = value

            # -------------------------------------------------
            # Save results
            # -------------------------------------------------

            os.makedirs(
                "outputs",
                exist_ok=True
            )

            with open(
                "outputs/results.json",
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=4,
                    default=str
                )

            print("\nAnalysis execution completed.")

            return results

        except Exception as e:

            print("\nExecution Agent Error:")
            print(e)

            return {}