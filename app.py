import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Discovery Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0f16;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #f5f7fa !important;
    }

    .hero {
        padding: 20px 0 30px 0;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #8b98ad;
        font-size: 17px;
        margin-top: 6px;
    }

    .upload-card {
        background: #121925;
        border: 1px solid #263247;
        border-radius: 18px;
        padding: 30px;
        margin: 10px 0 30px 0;
    }

    .upload-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .upload-text {
        color: #9ba8bb;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .dataset-header {
        background: #121925;
        border: 1px solid #263247;
        border-radius: 16px;
        padding: 22px 25px;
        margin-bottom: 25px;
    }

    .dataset-name {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }

    .dataset-meta {
        color: #8e9bae;
        margin-top: 6px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #8e9bae;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .kpi {
        background: #121925;
        border: 1px solid #263247;
        border-radius: 15px;
        padding: 20px;
        min-height: 115px;
    }

    .kpi-label {
        color: #8491a5;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: .5px;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 700;
        margin-top: 8px;
    }

    .hypothesis-card {
        background: #111824;
        border: 1px solid #2a374d;
        border-radius: 18px;
        padding: 26px;
        margin: 25px 0 35px 0;
    }

    .hypothesis-number {
        color: #6ea8fe;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .hypothesis-question {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
        line-height: 1.35;
        margin-bottom: 20px;
    }

    .info-box {
        background: #171f2d;
        border-left: 4px solid #6ea8fe;
        border-radius: 8px;
        padding: 16px 18px;
        margin: 10px 0 20px 0;
        color: #dbe2ec;
        line-height: 1.55;
    }

    .result-box {
        background: #121925;
        border: 1px solid #29364a;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
    }

    .insight-box {
        background: #17251e;
        border-left: 4px solid #54c58a;
        border-radius: 8px;
        padding: 17px 18px;
        color: #dceee3;
        line-height: 1.6;
    }

    .method-tag {
        display: inline-block;
        background: #1b2a42;
        color: #9ec4ff;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin-bottom: 10px;
    }

    .empty-title {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 80px;
    }

    .empty-text {
        text-align: center;
        color: #8d99aa;
        font-size: 16px;
        max-width: 650px;
        margin: 12px auto;
        line-height: 1.6;
    }

    .pipeline {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 30px 0;
    }

    .pipeline-item {
        background: #151d2a;
        border: 1px solid #29364a;
        color: #b9c4d4;
        border-radius: 20px;
        padding: 8px 13px;
        font-size: 12px;
    }

    .footer {
        text-align: center;
        color: #596678;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #202a39;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def load_json(filename, default=None):

    path = OUTPUT_DIR / filename

    if not path.exists():
        return default

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return default


def load_text(filename):

    path = OUTPUT_DIR / filename

    if not path.exists():
        return ""

    try:
        return path.read_text(
            encoding="utf-8"
        )

    except Exception:
        return ""


def clear_outputs():

    for item in OUTPUT_DIR.iterdir():

        if item.is_file():

            try:
                item.unlink()

            except Exception:
                pass


def clear_uploaded_datasets():

    if not UPLOAD_DIR.exists():
        return

    for item in UPLOAD_DIR.iterdir():

        if item.is_file():

            try:
                item.unlink()

            except Exception:
                pass

        elif item.is_dir():

            try:
                shutil.rmtree(item)

            except Exception:
                pass


def save_uploaded_file(uploaded_file):

    clear_uploaded_datasets()
    clear_outputs()

    destination = (
        UPLOAD_DIR
        / uploaded_file.name
    )

    with open(
        destination,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    return destination


def find_uploaded_dataset():

    files = []

    for extension in [
        ".csv",
        ".xlsx",
        ".xls"
    ]:

        files.extend(
            UPLOAD_DIR.glob(
                f"*{extension}"
            )
        )

    if not files:
        return None

    return files[0]


def read_dataset(path):

    if path is None:
        return None

    try:

        if path.suffix.lower() == ".csv":

            return pd.read_csv(
                path
            )

        return pd.read_excel(
            path
        )

    except Exception as error:

        st.error(
            f"Could not read dataset: {error}"
        )

        return None


def clean_text(value):

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "```",
        ""
    )

    return text.strip()


def get_hypotheses():

    data = load_json(
        "hypotheses.json",
        {}
    )

    if isinstance(
        data,
        dict
    ):

        return data.get(
            "hypotheses",
            []
        )

    if isinstance(
        data,
        list
    ):

        return data

    return []


def get_results():

    data = load_json(
        "analysis_results.json",
        []
    )

    if isinstance(
        data,
        dict
    ):

        return data.get(
            "results",
            []
        )

    return data if isinstance(
        data,
        list
    ) else []


def get_insights():

    data = load_json(
        "insights.json",
        {}
    )

    if isinstance(
        data,
        dict
    ):

        return data.get(
            "insights",
            []
        )

    return data if isinstance(
        data,
        list
    ) else []


def get_visualizations():

    data = load_json(
        "visualizations.json",
        {}
    )

    if isinstance(
        data,
        dict
    ):

        return data.get(
            "visualizations",
            []
        )

    return data if isinstance(
        data,
        list
    ) else []


# ============================================================
# VISUALIZATION ENGINE
# ============================================================

def render_visualization(
    df,
    visualization
):

    if not isinstance(
        visualization,
        dict
    ):
        return

    chart_type = str(
        visualization.get(
            "type",
            ""
        )
    ).lower()

    columns = visualization.get(
        "columns",
        []
    )

    x = visualization.get(
        "x"
    )

    y = visualization.get(
        "y"
    )

    if x is None and len(columns) >= 1:
        x = columns[0]

    if y is None and len(columns) >= 2:
        y = columns[1]

    if (
        x not in df.columns
        and x is not None
    ):
        return

    if (
        y not in df.columns
        and y is not None
    ):
        return

    try:

        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

        if chart_type == "histogram":

            fig = px.histogram(
                df,
                x=x,
                title=visualization.get(
                    "title",
                    ""
                ),
                marginal="box"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # BOXPLOT
        # ----------------------------------------------------

        elif chart_type == "boxplot":

            fig = px.box(
                df,
                y=y,
                title=visualization.get(
                    "title",
                    ""
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # BAR
        # ----------------------------------------------------

        elif chart_type == "bar":

            aggregation = (
                visualization.get(
                    "aggregation",
                    "mean"
                )
            )

            grouped = (
                df.groupby(
                    x,
                    dropna=False
                )[y]
            )

            if aggregation == "sum":

                values = grouped.sum()

            elif aggregation == "count":

                values = grouped.count()

            elif aggregation == "median":

                values = grouped.median()

            else:

                values = grouped.mean()

            chart_df = (
                values
                .reset_index()
                .sort_values(
                    y,
                    ascending=False
                )
                .head(20)
            )

            fig = px.bar(
                chart_df,
                x=x,
                y=y,
                title=visualization.get(
                    "title",
                    ""
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # SCATTER
        # ----------------------------------------------------

        elif chart_type == "scatter":

            fig = px.scatter(
                df,
                x=x,
                y=y,
                title=visualization.get(
                    "title",
                    ""
                ),
                trendline=None
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # TIME SERIES
        # ----------------------------------------------------

        elif chart_type == "time_series":

            temp = df[
                [
                    x,
                    y
                ]
            ].copy()

            temp[x] = pd.to_datetime(
                temp[x],
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[
                    x,
                    y
                ]
            )

            if temp.empty:
                return

            aggregation = (
                visualization.get(
                    "aggregation",
                    "sum"
                )
            )

            if aggregation == "mean":

                trend = (
                    temp
                    .groupby(x)[y]
                    .mean()
                    .reset_index()
                )

            elif aggregation == "median":

                trend = (
                    temp
                    .groupby(x)[y]
                    .median()
                    .reset_index()
                )

            else:

                trend = (
                    temp
                    .groupby(x)[y]
                    .sum()
                    .reset_index()
                )

            fig = px.line(
                trend,
                x=x,
                y=y,
                markers=True,
                title=visualization.get(
                    "title",
                    ""
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # TIME-LIKE OBJECT
        # ----------------------------------------------------

        elif chart_type == "time_of_day":

            temp = df[
                [
                    x,
                    y
                ]
            ].copy()

            temp["_time"] = pd.to_datetime(
                temp[x].astype(str),
                errors="coerce"
            )

            temp = temp.dropna(
                subset=[
                    "_time"
                ]
            )

            if temp.empty:
                return

            temp["hour"] = (
                temp["_time"]
                .dt.hour
            )

            trend = (
                temp
                .groupby("hour")[y]
                .sum()
                .reset_index()
            )

            fig = px.line(
                trend,
                x="hour",
                y=y,
                markers=True,
                title=visualization.get(
                    "title",
                    ""
                )
            )

            fig.update_xaxes(
                title="Hour of day"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # CATEGORY COUNT
        # ----------------------------------------------------

        elif chart_type == "category_count":

            counts = (
                df[x]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                x,
                "count"
            ]

            fig = px.bar(
                counts,
                x=x,
                y="count",
                title=visualization.get(
                    "title",
                    ""
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        elif chart_type == "category_heatmap":

            table = pd.crosstab(
                df[x],
                df[y]
            )

            if table.empty:
                return

            fig = px.imshow(
                table,
                text_auto=True,
                aspect="auto",
                title=visualization.get(
                    "title",
                    ""
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    except Exception as error:

        st.warning(
            f"Visualization could not be displayed: {error}"
        )


# ============================================================
# FIND VISUALIZATION FOR HYPOTHESIS
# ============================================================

def find_hypothesis_visualization(
    hypothesis,
    visualizations
):

    variables = hypothesis.get(
        "variables",
        []
    )

    variables = set(
        str(v)
        for v in variables
    )

    method = str(
        hypothesis.get(
            "analysis_method",
            ""
        )
    ).lower()

    # First try exact variable match
    for visualization in visualizations:

        columns = set(
            str(c)
            for c in visualization.get(
                "columns",
                []
            )
        )

        if (
            variables
            and variables.issubset(columns)
        ):

            return visualization

    # Then method-based matching
    if (
        "correlation" in method
        or "regression" in method
    ):

        for visualization in visualizations:

            if visualization.get(
                "type"
            ) == "scatter":

                return visualization

    if (
        "anova" in method
    ):

        for visualization in visualizations:

            if visualization.get(
                "type"
            ) == "bar":

                return visualization

    if (
        "time" in method
        or "trend" in method
    ):

        for visualization in visualizations:

            if visualization.get(
                "type"
            ) == "time_series":

                return visualization

    return None


# ============================================================
# FIND RESULT
# ============================================================

def find_result(
    hypothesis_id,
    results
):

    for result in results:

        if str(
            result.get(
                "hypothesis_id"
            )
        ) == str(
            hypothesis_id
        ):

            return result

    return None


# ============================================================
# FIND INSIGHT
# ============================================================

def find_insight(
    hypothesis_id,
    insights
):

    for insight in insights:

        if str(
            insight.get(
                "hypothesis_id"
            )
        ) == str(
            hypothesis_id
        ):

            return insight

    return None


# ============================================================
# RUN PIPELINE
# ============================================================

def run_discovery_agent():

    progress = st.progress(
        0
    )

    status = st.empty()

    steps = [
        "Loading dataset...",
        "Profiling data...",
        "Generating metadata...",
        "Generating discovery questions...",
        "Running analysis...",
        "Generating insights...",
        "Creating visualizations..."
    ]

    for index, step in enumerate(
        steps[:-1]
    ):

        status.info(
            step
        )

        progress.progress(
            int(
                ((index + 1)
                / len(steps))
                * 90
            )
        )

        # The actual pipeline is executed below.
        # This loop only gives visual feedback.

    status.info(
        "Running Discovery Agent..."
    )

    try:

        process = subprocess.run(
            [
                sys.executable,
                str(
                    BASE_DIR / "main.py"
                )
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True
        )

        progress.progress(
            100
        )

        if process.returncode != 0:

            status.error(
                "Discovery Agent failed."
            )

            with st.expander(
                "View error details"
            ):

                st.code(
                    process.stderr
                    or process.stdout
                )

            return False

        status.success(
            "Discovery completed successfully."
        )

        return True

    except Exception as error:

        status.error(
            f"Could not start Discovery Agent: {error}"
        )

        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔎 Discovery Agent"
    )

    st.caption(
        "Automated data discovery and insight system"
    )

    st.divider()

    st.markdown(
        "### Input Dataset"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        help=(
            "Upload a dataset to start "
            "the Discovery Agent."
        )
    )

    if uploaded_file is not None:

        current_path = (
            UPLOAD_DIR
            / uploaded_file.name
        )

        # Save only when a new file is selected
        if (
            "uploaded_name"
            not in st.session_state
            or st.session_state.uploaded_name
            != uploaded_file.name
        ):

            save_uploaded_file(
                uploaded_file
            )

            st.session_state.uploaded_name = (
                uploaded_file.name
            )

            st.session_state.pipeline_run = False

            st.rerun()

    dataset_path = find_uploaded_dataset()

    if dataset_path:

        st.success(
            f"Dataset ready\n\n{dataset_path.name}"
        )

        run_button = st.button(
            "🚀 Run Discovery Agent",
            use_container_width=True,
            type="primary"
        )

        if run_button:

            success = (
                run_discovery_agent()
            )

            if success:

                st.session_state.pipeline_run = (
                    True
                )

                st.rerun()

    else:

        st.info(
            "Upload a dataset to begin."
        )

    st.divider()

    if dataset_path:

        st.markdown(
            "### Navigation"
        )

        page = st.radio(
            "Navigation",
            [
                "Discovery",
                "Dataset",
                "Metadata",
                "Analysis Details",
                "Generated Code"
            ],
            label_visibility="collapsed"
        )

    else:

        page = "Discovery"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🔎 Discovery Agent
        </div>
        <div class="hero-subtitle">
            Automated data discovery, analysis and insight generation
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EMPTY STATE
# ============================================================

if dataset_path is None:

    st.markdown(
        """
        <div class="upload-card">

            <div class="empty-title">
                Start with your dataset
            </div>

            <div class="empty-text">
                Upload a CSV or Excel file using the input area
                on the left. The Discovery Agent will automatically
                understand the dataset, generate useful questions,
                analyze them and create relevant visualizations.
            </div>

            <div class="pipeline">

                <div class="pipeline-item">
                    Upload
                </div>

                <div class="pipeline-item">
                    Profile
                </div>

                <div class="pipeline-item">
                    Understand
                </div>

                <div class="pipeline-item">
                    Discover Questions
                </div>

                <div class="pipeline-item">
                    Analyze
                </div>

                <div class="pipeline-item">
                    Visualize
                </div>

                <div class="pipeline-item">
                    Insights
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            No dataset loaded • No previous dataset displayed
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# DATASET
# ============================================================

df = read_dataset(
    dataset_path
)

if df is None:

    st.stop()


# ============================================================
# DISCOVERY PAGE
# ============================================================

if page == "Discovery":

    st.markdown(
        f"""
        <div class="dataset-header">

            <div class="dataset-name">
                📄 {dataset_path.name}
            </div>

            <div class="dataset-meta">
                {len(df):,} rows · {len(df.columns)} columns
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    with cols[0]:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Rows</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cols[1]:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{len(df.columns)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cols[2]:

        missing = int(
            df.isna()
            .sum()
            .sum()
        )

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Missing Values</div>
                <div class="kpi-value">{missing:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with cols[3]:

        numeric_count = len(
            df.select_dtypes(
                include="number"
            ).columns
        )

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">Numeric Fields</div>
                <div class="kpi-value">{numeric_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # QUICK DATA VIEW
    # --------------------------------------------------------

    with st.expander(
        "Preview dataset",
        expanded=False
    ):

        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # CHECK PIPELINE OUTPUT
    # --------------------------------------------------------

    hypotheses = get_hypotheses()
    results = get_results()
    insights = get_insights()
    visualizations = get_visualizations()

    if not hypotheses:

        st.markdown(
            """
            <div class="section-title">
                Discovery Questions
            </div>

            <div class="section-subtitle">
                Upload the dataset and click
                <b>Run Discovery Agent</b> to begin.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "No discovery analysis has been run for this dataset yet."
        )

        st.stop()

    # --------------------------------------------------------
    # DISCOVERY FLOW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Discovery Questions
        </div>

        <div class="section-subtitle">
            The agent investigates the dataset one question at a time.
            Each question is followed by its reasoning, solution,
            result, visualization and insight.
        </div>
        """,
        unsafe_allow_html=True
    )

    for index, hypothesis in enumerate(
        hypotheses,
        start=1
    ):

        hypothesis_id = hypothesis.get(
            "id",
            index
        )

        question = clean_text(
            hypothesis.get(
                "question"
            )
        )

        explanation = clean_text(
            hypothesis.get(
                "simple_explanation"
            )
            or hypothesis.get(
                "what_are_we_finding"
            )
        )

        solution = clean_text(
            hypothesis.get(
                "solution"
            )
            or hypothesis.get(
                "how_will_we_solve"
            )
        )

        method = clean_text(
            hypothesis.get(
                "analysis_method"
            )
        )

        result = find_result(
            hypothesis_id,
            results
        )

        insight = find_insight(
            hypothesis_id,
            insights
        )

        visualization = (
            find_hypothesis_visualization(
                hypothesis,
                visualizations
            )
        )

        st.markdown(
            f"""
            <div class="hypothesis-card">

                <div class="hypothesis-number">
                    Discovery Question {index}
                </div>

                <div class="hypothesis-question">
                    {question}
                </div>

                <div class="method-tag">
                    {method}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # WHAT ARE WE FINDING?
        # ----------------------------------------------------

        st.markdown(
            "#### 1. What are we trying to find?"
        )

        st.markdown(
            f"""
            <div class="info-box">
                {explanation}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # HOW WILL WE SOLVE IT?
        # ----------------------------------------------------

        st.markdown(
            "#### 2. How will we solve it?"
        )

        st.markdown(
            f"""
            <div class="info-box">
                {solution}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # ANALYSIS RESULT
        # ----------------------------------------------------

        st.markdown(
            "#### 3. What did the analysis find?"
        )

        if result:

            conclusion = clean_text(
                result.get(
                    "conclusion"
                )
            )

            result_data = result.get(
                "result",
                {}
            )

            if conclusion:

                st.markdown(
                    f"""
                    <div class="result-box">
                        {conclusion}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Display useful result values
            if isinstance(
                result_data,
                dict
            ):

                display_values = []

                important_keys = [
                    "correlation_coefficient",
                    "p_value",
                    "r_squared",
                    "slope",
                    "intercept",
                    "f_statistic",
                    "number_of_groups",
                    "observations"
                ]

                for key in important_keys:

                    if key in result_data:

                        label = (
                            key
                            .replace(
                                "_",
                                " "
                            )
                            .title()
                        )

                        value = result_data[
                            key
                        ]

                        display_values.append(
                            (
                                label,
                                value
                            )
                        )

                if display_values:

                    result_cols = st.columns(
                        min(
                            4,
                            len(display_values)
                        )
                    )

                    for col, item in zip(
                        result_cols,
                        display_values
                    ):

                        with col:

                            st.metric(
                                item[0],
                                str(
                                    item[1]
                                )
                            )

        else:

            st.info(
                "Analysis result is not available."
            )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        st.markdown(
            "#### 4. What does the visualization show?"
        )

        if visualization:

            st.caption(
                visualization.get(
                    "purpose",
                    ""
                )
            )

            render_visualization(
                df,
                visualization
            )

        else:

            st.info(
                "No visualization was generated for this question."
            )

        # ----------------------------------------------------
        # INSIGHT
        # ----------------------------------------------------

        st.markdown(
            "#### 5. What is the main insight?"
        )

        if insight:

            insight_text = clean_text(
                insight.get(
                    "insight"
                )
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    {insight_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            expected = clean_text(
                hypothesis.get(
                    "expected_insight"
                )
            )

            st.markdown(
                f"""
                <div class="insight-box">
                    {expected}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**{dataset_path.name}** — "
        f"{len(df):,} rows × {len(df.columns)} columns"
    )

    search = st.text_input(
        "Search dataset",
        placeholder="Search across all columns..."
    )

    filtered = df.copy()

    if search:

        mask = pd.Series(
            False,
            index=df.index
        )

        for column in df.columns:

            mask = (
                mask
                | df[column]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            )

        filtered = df[
            mask
        ]

    st.dataframe(
        filtered.head(500),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# METADATA PAGE
# ============================================================

elif page == "Metadata":

    st.markdown(
        '<div class="section-title">Dataset Understanding</div>',
        unsafe_allow_html=True
    )

    metadata = load_json(
        "metadata.json",
        {}
    )

    profile = load_json(
        "profile.json",
        {}
    )

    if metadata:

        st.markdown(
            "### Dataset Description"
        )

        description = (
            metadata.get(
                "description"
            )
            or metadata.get(
                "dataset_description"
            )
            or metadata.get(
                "overview"
            )
        )

        if description:

            st.write(
                clean_text(
                    description
                )
            )

        with st.expander(
            "View complete metadata"
        ):

            st.json(
                metadata
            )

    if profile:

        st.markdown(
            "### Data Profile"
        )

        with st.expander(
            "View profile"
        ):

            st.json(
                profile
            )


# ============================================================
# ANALYSIS DETAILS
# ============================================================

elif page == "Analysis Details":

    st.markdown(
        '<div class="section-title">Analysis Details</div>',
        unsafe_allow_html=True
    )

    results = get_results()

    if not results:

        st.info(
            "No analysis results available."
        )

    for result in results:

        st.markdown(
            f"### Question {result.get('hypothesis_id', '')}"
        )

        st.write(
            result.get(
                "question",
                ""
            )
        )

        st.json(
            result.get(
                "result",
                result
            )
        )


# ============================================================
# GENERATED CODE
# ============================================================

elif page == "Generated Code":

    st.markdown(
        '<div class="section-title">Generated Analysis Code</div>',
        unsafe_allow_html=True
    )

    generated = load_text(
        "generated_analysis.py"
    )

    verified = load_text(
        "verified_analysis.py"
    )

    with st.expander(
        "Generated Python",
        expanded=True
    ):

        if generated:

            st.code(
                generated,
                language="python"
            )

        else:

            st.info(
                "Generated code is not available."
            )

    with st.expander(
        "Verified Python"
    ):

        if verified:

            st.code(
                verified,
                language="python"
            )

        else:

            st.info(
                "Verified code is not available."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Discovery Agent · Automated Data Understanding ·
        Question Generation · Analysis · Visualization · Insights
    </div>
    """,
    unsafe_allow_html=True
)