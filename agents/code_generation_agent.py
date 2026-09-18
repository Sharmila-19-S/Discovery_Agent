import json
import os
import re

from models.ollama_client import generate


class CodeGenerationAgent:

    @staticmethod
    def _clean_code(code):
        if not code:
            return ""

        code = code.strip()

        # Remove markdown fences
        code = re.sub(r"^```python\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code)

        # Remove accidental leading explanation
        lines = code.splitlines()

        while lines and not lines[0].strip():
            lines.pop(0)

        # If model added text before actual Python
        python_starts = (
            "import ",
            "from ",
            "results",
            "def ",
            "#",
        )

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith(python_starts):
                lines = lines[i:]
                break

        return "\n".join(lines).strip()

    @staticmethod
    def _validate_structure(hypotheses):
        valid = []

        for index, hypothesis in enumerate(hypotheses, start=1):

            if not isinstance(hypothesis, dict):
                continue

            valid.append({
                "id": hypothesis.get("id", index),
                "question": str(
                    hypothesis.get("question", "")
                ),
                "variables": hypothesis.get(
                    "variables", []
                ),
                "analysis_method": str(
                    hypothesis.get("analysis_method", "")
                ),
                "expected_insight": str(
                    hypothesis.get("expected_insight", "")
                )
            })

        return valid

    @staticmethod
    def _check_safety(code):

        forbidden = [
            "import os",
            "from os ",
            "import subprocess",
            "from subprocess",
            "import shutil",
            "from shutil",
            "import requests",
            "from requests",
            "import urllib",
            "from urllib",
            "os.system",
            "subprocess.",
            "shutil.",
            "requests.",
            "urllib.",
            "socket.",
            "open(",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "globals(",
            "locals(",
        ]

        code_lower = code.lower()

        for pattern in forbidden:
            if pattern.lower() in code_lower:
                return False, pattern

        return True, None

    @staticmethod
    def generate(hypotheses, columns, profile=None):

        print("Generating analysis code...")

        # =====================================================
        # NORMALIZE HYPOTHESES
        # =====================================================

        if isinstance(hypotheses, str):

            try:
                hypotheses = json.loads(hypotheses)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Hypotheses string is not valid JSON."
                ) from exc

        if isinstance(hypotheses, dict):
            hypotheses = hypotheses.get(
                "hypotheses",
                []
            )

        if not isinstance(hypotheses, list):
            raise RuntimeError(
                "Invalid hypotheses format."
            )

        valid_hypotheses = (
            CodeGenerationAgent._validate_structure(
                hypotheses
            )
        )

        if not valid_hypotheses:
            raise RuntimeError(
                "No valid hypotheses available."
            )

        print(
            f"{len(valid_hypotheses)} hypotheses "
            f"received for code generation."
        )

        # =====================================================
        # COLUMNS
        # =====================================================

        available_columns = list(columns)

        columns_text = json.dumps(
            available_columns,
            ensure_ascii=False,
            indent=2
        )

        hypotheses_text = json.dumps(
            valid_hypotheses,
            ensure_ascii=False,
            indent=2
        )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = """
You are an expert Python data analyst.

Generate executable Python code for the dataframe `df`.

The dataframe already exists in memory.

IMPORTANT:

1. Do NOT load any dataset.
2. Do NOT create a dataframe.
3. Use only columns supplied by the user.
4. Never invent column names.
5. Analyze EVERY supplied hypothesis.
6. Handle missing values safely.
7. Use pandas and scipy.stats when statistical testing is required.
8. Do not create charts or visualizations.
9. Do not write files.
10. Do not use network access.
11. Do not use os, subprocess, shutil, requests, urllib, socket,
    eval, exec or __import__.
12. Do not print unnecessary information.
13. The code must work with the existing dataframe named `df`.

The final variable MUST be:

results

`results` must be a Python list.

For every hypothesis, append exactly one dictionary:

{
    "hypothesis_id": ...,
    "question": "...",
    "method": "...",
    "result": ...,
    "conclusion": "..."
}

The `result` value may contain dictionaries,
numbers, strings, lists, or other JSON-compatible values.

The `conclusion` must directly answer the hypothesis.

If a requested statistical method cannot be performed,
use the most appropriate valid alternative.

Return ONLY executable Python code.

Do not use markdown fences.
Do not provide explanations outside the code.
"""

        # =====================================================
        # USER PROMPT
        # =====================================================

        user_prompt = f"""
AVAILABLE DATAFRAME COLUMNS:

{columns_text}

HYPOTHESES:

{hypotheses_text}

Generate Python code that analyzes ALL hypotheses.

The dataframe is already available as:

df

The final output must be stored in:

results
"""

        # =====================================================
        # GENERATE
        # =====================================================

        response = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )

        if not response:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        code = CodeGenerationAgent._clean_code(
            response
        )

        if not code:
            raise RuntimeError(
                "Generated analysis code is empty."
            )

        # =====================================================
        # SAFETY CHECK
        # =====================================================

        safe, pattern = (
            CodeGenerationAgent._check_safety(code)
        )

        if not safe:
            raise RuntimeError(
                f"Unsafe generated code detected: {pattern}"
            )

        # =====================================================
        # SYNTAX CHECK
        # =====================================================

        try:
            compile(
                code,
                "<generated_analysis>",
                "exec"
            )
        except SyntaxError as exc:
            raise RuntimeError(
                "Generated code contains a syntax error: "
                f"{exc}"
            ) from exc

        # =====================================================
        # CHECK RESULTS VARIABLE
        # =====================================================

        if "results" not in code:
            raise RuntimeError(
                "Generated code does not create the required "
                "`results` variable."
            )

        # =====================================================
        # SAVE
        # =====================================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        output_path = (
            "outputs/generated_analysis.py"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(code)

        print(
            "Python analysis code generated successfully."
        )

        return code