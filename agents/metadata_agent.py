import os

from models.ollama_client import generate


class MetadataAgent:

    @staticmethod
    def generate(profile):

        print("Preparing compact metadata prompt...")

        system_prompt = """
You are a professional data analyst.

Generate concise and accurate metadata from the supplied
dataset information.

Rules:
- Use only the information provided.
- Do not invent information.
- Do not generate hypotheses.
- Do not generate Python code.
- Keep the response under 200 words.
- Return clean Markdown.
"""

        user_prompt = f"""
Dataset Information:

Rows: {profile.get("rows")}

Columns: {profile.get("columns")}

Column Names:
{profile.get("column_names")}

Data Types:
{profile.get("data_types")}

Numeric Columns:
{profile.get("numeric_columns")}

Categorical Columns:
{profile.get("categorical_columns")}

Date Columns:
{profile.get("date_columns")}

Missing Values:
{profile.get("missing_values")}

Duplicate Rows:
{profile.get("duplicate_rows")}

Generate metadata with ONLY these sections:

1. Dataset Description
2. Column Descriptions
3. Business Domain
4. Possible Use Cases

Do not generate hypotheses.
Do not generate Python code.
Keep the response under 200 words.
"""

        print("Sending compact metadata request to Ollama...")

        metadata = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )

        if metadata is None:
            raise Exception(
                "Metadata Agent received no response."
            )

        os.makedirs("outputs", exist_ok=True)

        with open(
            "outputs/metadata.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(metadata)

        print("Metadata generated successfully.")

        return metadata