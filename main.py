import json
import traceback
from pathlib import Path

import pandas as pd

from analysis.data_profiler import DataProfiler
from analysis.analysis_engine import AnalysisEngine

from agents.metadata_agent import MetadataAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.hypothesis_validator import HypothesisValidator
from agents.insight_agent import InsightAgent
from agents.visualization_agent import VisualizationAgent


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MAX_ANALYSIS_ROWS = 400

SUPPORTED_EXTENSIONS = [
    ".csv",
    ".xlsx",
    ".xls"
]


# ============================================================
# JSON SAVE
# ============================================================

def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    upload_dir = DATA_DIR / "uploads"

    search_dirs = []

    if upload_dir.exists():
        search_dirs.append(upload_dir)

    search_dirs.append(DATA_DIR)

    files = []

    for directory in search_dirs:

        if not directory.exists():
            continue

        for file in directory.iterdir():

            if (
                file.is_file()
                and not file.name.startswith("~$")
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                files.append(file)

    if not files:

        raise FileNotFoundError(
            "No CSV or Excel dataset found in data folder."
        )

    return max(
        files,
        key=lambda file: file.stat().st_mtime
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    print("Reading dataset...")

    if path.suffix.lower() == ".csv":

        df = pd.read_csv(
            path,
            nrows=MAX_ANALYSIS_ROWS
        )

    else:

        df = pd.read_excel(
            path,
            nrows=MAX_ANALYSIS_ROWS
        )

    print(
        f"Working Dataset: "
        f"{len(df)} rows, "
        f"{df.shape[1]} columns"
    )

    return df


# ============================================================
# NORMALIZE HYPOTHESES
# ============================================================

def normalize_hypotheses(raw):

    if isinstance(raw, dict):

        raw = raw.get(
            "hypotheses",
            raw.get(
                "data",
                []
            )
        )

    if isinstance(raw, str):

        try:

            raw = json.loads(raw)

        except Exception:

            return []

        if isinstance(raw, dict):

            raw = raw.get(
                "hypotheses",
                raw.get(
                    "data",
                    []
                )
            )

    if not isinstance(raw, list):

        return []

    return [
        item
        for item in raw
        if isinstance(item, dict)
    ]


# ============================================================
# CLEAN HYPOTHESES
# ============================================================

def clean_hypotheses(hypotheses):

    cleaned = []

    for hypothesis in hypotheses:

        if not isinstance(
            hypothesis,
            dict
        ):
            continue

        question = (
            hypothesis.get("question")
            or hypothesis.get("hypothesis")
            or hypothesis.get("query")
            or ""
        )

        variables = (
            hypothesis.get("variables")
            or hypothesis.get("columns")
            or []
        )

        if isinstance(
            variables,
            str
        ):

            variables = [
                variables
            ]

        method = (
            hypothesis.get("analysis_method")
            or hypothesis.get("method")
            or hypothesis.get("analysis_type")
            or ""
        )

        expected = (
            hypothesis.get("expected_insight")
            or hypothesis.get("expected_result")
            or ""
        )

        simple_explanation = (
            hypothesis.get("simple_explanation")
            or hypothesis.get("what_are_we_finding")
            or ""
        )

        solution = (
            hypothesis.get("solution")
            or hypothesis.get("how_will_we_solve")
            or ""
        )

        cleaned.append(
            {
                **hypothesis,

                "question": str(
                    question
                ),

                "variables": [
                    str(v)
                    for v in variables
                ],

                "analysis_method": str(
                    method
                ),

                "expected_insight": str(
                    expected
                ),

                "simple_explanation": str(
                    simple_explanation
                ),

                "solution": str(
                    solution
                )
            }
        )

    return cleaned


# ============================================================
# METADATA
# ============================================================

def generate_metadata(
    df,
    profile
):

    print("\nGenerating Metadata...")

    agent = MetadataAgent()

    # Current MetadataAgent.generate() expects profile.
    metadata = agent.generate(
        profile
    )

    if isinstance(
        metadata,
        str
    ):

        try:

            metadata = json.loads(
                metadata
            )

        except Exception:

            metadata = {
                "description": metadata
            }

    if not isinstance(
        metadata,
        dict
    ):

        metadata = {
            "description": str(
                metadata
            )
        }

    save_json(
        OUTPUT_DIR / "metadata.json",
        metadata
    )

    with open(
        OUTPUT_DIR / "metadata.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                metadata,
                indent=4,
                ensure_ascii=False,
                default=str
            )
        )

    print(
        "Metadata generated successfully."
    )

    return metadata


# ============================================================
# HYPOTHESES
# ============================================================

def generate_hypotheses(
    df,
    profile,
    metadata
):

    print("\nGenerating Hypotheses...")
    print(
        "Generating dataset-specific hypotheses..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Always pass the REAL dataframe.
    #
    # Do NOT use:
    # agent.generate(profile)
    #
    # because profile is a dictionary.
    # --------------------------------------------------------

    agent = HypothesisAgent()

    raw = agent.generate(
        df=df,
        profile=profile,
        metadata=metadata,
        num_hypotheses=5
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    hypotheses = normalize_hypotheses(
        raw
    )

    hypotheses = clean_hypotheses(
        hypotheses
    )

    print(
        f"{len(hypotheses)} hypotheses generated successfully."
    )

    if not hypotheses:

        raise RuntimeError(
            "No hypotheses were generated."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "Validating hypotheses..."
    )

    validator = HypothesisValidator(
        df=df,
        profile=profile
    )

    validated = validator.validate(
        hypotheses
    )

    if not isinstance(
        validated,
        list
    ):

        raise TypeError(
            "HypothesisValidator.validate() "
            "must return a list of hypothesis dictionaries."
        )

    hypotheses = []

    for hypothesis in validated:

        if not isinstance(
            hypothesis,
            dict
        ):
            continue

        hypotheses.append(
            hypothesis
        )

    # --------------------------------------------------------
    # RENUMBER
    # --------------------------------------------------------

    for index, hypothesis in enumerate(
        hypotheses,
        start=1
    ):

        hypothesis["id"] = index

        hypothesis.setdefault(
            "simple_explanation",
            ""
        )

        hypothesis.setdefault(
            "solution",
            ""
        )

        hypothesis.setdefault(
            "expected_insight",
            ""
        )

    if not hypotheses:

        raise RuntimeError(
            "No valid hypotheses remained after validation."
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    save_json(
        OUTPUT_DIR / "hypotheses.json",
        {
            "hypotheses": hypotheses
        }
    )

    # --------------------------------------------------------
    # SAVE TXT
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "hypotheses.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for hypothesis in hypotheses:

            file.write(
                f"HYPOTHESIS {hypothesis['id']}\n"
            )

            file.write(
                f"Question: "
                f"{hypothesis.get('question', '')}\n"
            )

            file.write(
                f"What are we trying to find?: "
                f"{hypothesis.get('simple_explanation', '')}\n"
            )

            file.write(
                f"How will we solve it?: "
                f"{hypothesis.get('solution', '')}\n"
            )

            file.write(
                f"Variables: "
                f"{', '.join(map(str, hypothesis.get('variables', [])))}\n"
            )

            file.write(
                f"Method: "
                f"{hypothesis.get('analysis_method', '')}\n"
            )

            file.write(
                f"Expected Insight: "
                f"{hypothesis.get('expected_insight', '')}\n"
            )

            file.write(
                "\n"
            )

    print(
        f"{len(hypotheses)} valid hypotheses retained."
    )

    return hypotheses


# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(
    df,
    hypotheses
):

    print("\nRunning Analysis Engine...")

    engine = AnalysisEngine()

    try:

        results = engine.run(
            df,
            hypotheses
        )

    except TypeError:

        results = engine.run(
            df=df,
            hypotheses=hypotheses
        )

    if isinstance(
        results,
        dict
    ):

        results = results.get(
            "results",
            [results]
        )

    if not isinstance(
        results,
        list
    ):

        results = [
            results
        ]

    final_results = []

    for index, result in enumerate(
        results,
        start=1
    ):

        if not isinstance(
            result,
            dict
        ):

            result = {
                "result": str(
                    result
                )
            }

        hypothesis_id = result.get(
            "hypothesis_id",
            index
        )

        result["hypothesis_id"] = (
            hypothesis_id
        )

        # ----------------------------------------------------
        # ATTACH HYPOTHESIS INFORMATION
        # ----------------------------------------------------

        for hypothesis in hypotheses:

            if (
                hypothesis.get("id")
                == hypothesis_id
            ):

                result.setdefault(
                    "question",
                    hypothesis.get(
                        "question",
                        ""
                    )
                )

                result.setdefault(
                    "variables",
                    hypothesis.get(
                        "variables",
                        []
                    )
                )

                result.setdefault(
                    "method",
                    hypothesis.get(
                        "analysis_method",
                        ""
                    )
                )

                result.setdefault(
                    "expected_insight",
                    hypothesis.get(
                        "expected_insight",
                        ""
                    )
                )

                result.setdefault(
                    "simple_explanation",
                    hypothesis.get(
                        "simple_explanation",
                        ""
                    )
                )

                result.setdefault(
                    "solution",
                    hypothesis.get(
                        "solution",
                        ""
                    )
                )

                break

        final_results.append(
            result
        )

    save_json(
        OUTPUT_DIR / "analysis_results.json",
        final_results
    )

    print(
        f"Analysis completed: "
        f"{len(final_results)} results"
    )

    return final_results


# ============================================================
# INSIGHTS
# ============================================================

def generate_insights(
    analysis_results
):

    print("\nGenerating Insights...")

    agent = InsightAgent(
        output_dir=OUTPUT_DIR
    )

    insights = agent.run(
        analysis_results
    )

    if not isinstance(
        insights,
        dict
    ):

        insights = {
            "insight_count": 0,
            "insights": []
        }

    print(
        f"{insights.get('insight_count', 0)} "
        "insights generated."
    )

    return insights


# ============================================================
# VISUALIZATIONS
# ============================================================

def generate_visualizations(
    df
):

    print(
        "\nGenerating Dynamic Visualizations..."
    )

    agent = VisualizationAgent(
        output_dir=OUTPUT_DIR
    )

    visualizations = agent.run(
        df
    )

    if not isinstance(
        visualizations,
        dict
    ):

        visualizations = {
            "visualization_count": 0,
            "visualizations": []
        }

    print(
        f"{visualizations.get('visualization_count', 0)} "
        "visualizations generated."
    )

    return visualizations


# ============================================================
# DATASET OUTPUT
# ============================================================

def save_dataset(
    df
):

    df.to_csv(
        OUTPUT_DIR / "development_dataset.csv",
        index=False
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    df,
    dataset_name
):

    numeric_columns = list(
        df.select_dtypes(
            include="number"
        ).columns
    )

    non_numeric_columns = list(
        df.select_dtypes(
            exclude="number"
        ).columns
    )

    summary = {

        "dataset":
            dataset_name,

        "analysis_mode":
            "Resource-Efficient Sample",

        "rows_used":
            int(len(df)),

        "columns":
            int(df.shape[1]),

        "column_names":
            list(df.columns),

        "numeric_columns":
            numeric_columns,

        "non_numeric_columns":
            non_numeric_columns
    }

    save_json(
        OUTPUT_DIR / "dataset_summary.json",
        summary
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n============================================================"
    )

    print(
        "DISCOVERY AGENT"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\nLoading Dataset...")

    dataset_path = find_dataset()

    print(
        f"Dataset: {dataset_path}"
    )

    df = load_dataset(
        dataset_path
    )

    print(
        f"Analysis Dataset: "
        f"{len(df)} rows, "
        f"{df.shape[1]} columns"
    )

    save_dataset(
        df
    )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    print("\nGenerating Profile...")

    profile = DataProfiler.profile(
        df
    )

    save_json(
        OUTPUT_DIR / "profile.json",
        profile
    )

    print(
        "Profile generated successfully."
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = generate_metadata(
        df,
        profile
    )

    # --------------------------------------------------------
    # HYPOTHESES
    # --------------------------------------------------------

    hypotheses = generate_hypotheses(
        df,
        profile,
        metadata
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    analysis_results = run_analysis(
        df,
        hypotheses
    )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    insights = generate_insights(
        analysis_results
    )

    # --------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------

    visualizations = generate_visualizations(
        df
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    create_dataset_summary(
        df,
        dataset_path.name
    )

    save_json(
        OUTPUT_DIR / "run_info.json",
        {
            "dataset":
                dataset_path.name,

            "rows_used":
                len(df),

            "columns":
                df.shape[1],

            "hypotheses":
                len(hypotheses),

            "analysis_results":
                len(analysis_results),

            "insights":
                insights.get(
                    "insight_count",
                    0
                ),

            "visualizations":
                visualizations.get(
                    "visualization_count",
                    0
                )
        }
    )

    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "DISCOVERY AGENT RUN COMPLETED"
    )

    print(
        "============================================================"
    )

    print(
        f"\nDataset: {dataset_path.name}"
    )

    print(
        f"Rows Used: {len(df)}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print(
        f"Hypotheses: {len(hypotheses)}"
    )

    print(
        f"Analysis Results: {len(analysis_results)}"
    )

    print(
        f"Insights: "
        f"{insights.get('insight_count', 0)}"
    )

    print(
        f"Visualizations: "
        f"{visualizations.get('visualization_count', 0)}"
    )

    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "\n============================================================"
        )

        print(
            "DISCOVERY AGENT ERROR"
        )

        print(
            "============================================================"
        )

        print(
            f"\n{type(error).__name__}: {error}"
        )

        print(
            "\nFull traceback:"
        )

        traceback.print_exc()

        print()