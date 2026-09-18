import ast
import os
import re

from models.ollama_client import generate


class CodeVerificationAgent:

    FORBIDDEN_IMPORTS = {
        "os",
        "subprocess",
        "shutil",
        "requests",
        "urllib",
        "socket",
    }

    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
    }

    @staticmethod
    def _clean_code(code):

        if not code:
            return ""

        code = code.strip()

        code = re.sub(
            r"^```python\s*",
            "",
            code,
            flags=re.IGNORECASE
        )

        code = re.sub(
            r"^```\s*",
            "",
            code
        )

        code = re.sub(
            r"\s*```$",
            "",
            code
        )

        return code.strip()

    @staticmethod
    def _check_ast_safety(code):

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        for node in ast.walk(tree):

            # -------------------------------------------------
            # IMPORT CHECK
            # -------------------------------------------------

            if isinstance(node, ast.Import):

                for alias in node.names:

                    root = alias.name.split(".")[0]

                    if root in CodeVerificationAgent.FORBIDDEN_IMPORTS:
                        return False, (
                            f"Forbidden import: {alias.name}"
                        )

            if isinstance(node, ast.ImportFrom):

                if node.module:

                    root = node.module.split(".")[0]

                    if root in CodeVerificationAgent.FORBIDDEN_IMPORTS:
                        return False, (
                            f"Forbidden import: {node.module}"
                        )

            # -------------------------------------------------
            # FUNCTION CALL CHECK
            # -------------------------------------------------

            if isinstance(node, ast.Call):

                if isinstance(node.func, ast.Name):

                    if (
                        node.func.id
                        in CodeVerificationAgent.FORBIDDEN_CALLS
                    ):
                        return False, (
                            f"Forbidden function: "
                            f"{node.func.id}"
                        )

                if isinstance(node.func, ast.Attribute):

                    if node.func.attr in {
                        "system",
                        "popen",
                        "remove",
                        "rmtree",
                        "run",
                        "Popen",
                    }:
                        return False, (
                            f"Forbidden operation: "
                            f"{node.func.attr}"
                        )

        return True, None

    @staticmethod
    def _local_syntax_check(code):

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def verify(code):

        print("Verifying generated analysis code...")

        if not code:
            raise RuntimeError(
                "No code supplied for verification."
            )

        code = CodeVerificationAgent._clean_code(
            code
        )

        # =====================================================
        # FIRST LOCAL CHECK
        # =====================================================

        safe, reason = (
            CodeVerificationAgent._check_ast_safety(code)
        )

        if not safe:
            raise RuntimeError(
                f"Unsafe generated code: {reason}"
            )

        # =====================================================
        # IF ALREADY VALID, KEEP IT
        # =====================================================

        if (
            CodeVerificationAgent._local_syntax_check(code)
            and "results" in code
        ):

            verified_code = code

            print(
                "Generated code passed local verification."
            )

        else:

            # =================================================
            # ASK OLLAMA TO REPAIR
            # =================================================

            system_prompt = """
You are an expert Python software engineer.

Repair the supplied Python analysis code.

Rules:

1. The dataframe already exists as `df`.
2. Do not load any dataset.
3. Do not create a new dataframe.
4. Preserve the original analysis logic.
5. Fix syntax errors.
6. Fix indentation errors.
7. Fix invalid Python syntax.
8. Keep the `results` variable.
9. `results` must remain a list.
10. Do not create visualizations.
11. Do not write files.
12. Do not use network access.
13. Do not use os, subprocess, shutil, requests,
    urllib or socket.
14. Do not use eval, exec, compile or open.
15. Return ONLY executable Python code.
16. Do not use markdown fences.
17. Do not provide explanations.
"""

            user_prompt = f"""
Repair and return the following Python code:

{code}
"""

            response = generate(
                system_prompt,
                user_prompt,
                temperature=0.1
            )

            if not response:
                raise RuntimeError(
                    "Ollama returned empty verified code."
                )

            verified_code = (
                CodeVerificationAgent._clean_code(
                    response
                )
            )

            # =============================================
            # VERIFY REPAIRED CODE
            # =============================================

            safe, reason = (
                CodeVerificationAgent._check_ast_safety(
                    verified_code
                )
            )

            if not safe:
                raise RuntimeError(
                    f"Verified code is unsafe: {reason}"
                )

            if not CodeVerificationAgent._local_syntax_check(
                verified_code
            ):
                raise RuntimeError(
                    "Verified code still contains syntax errors."
                )

            if "results" not in verified_code:
                raise RuntimeError(
                    "Verified code does not contain "
                    "the required `results` variable."
                )

            print(
                "Code repaired and verified successfully."
            )

        # =====================================================
        # SAVE VERIFIED CODE
        # =====================================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        output_path = (
            "outputs/verified_analysis.py"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(verified_code)

        print(
            "Verified analysis code saved successfully."
        )

        return verified_code