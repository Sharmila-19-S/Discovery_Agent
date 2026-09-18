import os
import requests


# ============================================================
# LOCAL OLLAMA SETTINGS
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b"
)


# ============================================================
# CLOUD OPENAI SETTINGS
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)


# ============================================================
# GENERATE
# ============================================================

def generate(
    system_prompt,
    user_prompt,
    temperature=0.1
):

    # ========================================================
    # CLOUD MODE
    # ========================================================

    if OPENAI_API_KEY:

        try:

            from openai import OpenAI

            client = OpenAI(
                api_key=OPENAI_API_KEY
            )

            response = client.responses.create(
                model=OPENAI_MODEL,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=700
            )

            return response.output_text

        except Exception as e:

            raise RuntimeError(
                f"Cloud LLM error: {str(e)}"
            )

    # ========================================================
    # LOCAL OLLAMA MODE
    # ========================================================

    print("Connecting to Ollama...")
    print("Model:", OLLAMA_MODEL)

    prompt = f"""
<System>
{system_prompt}

<User>
{user_prompt}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 700
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        print("Ollama response received.")

        return data.get(
            "response"
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Ollama request timed out after 300 seconds."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama. Start Ollama first."
        )

    except Exception as e:

        raise RuntimeError(
            f"Ollama error: {str(e)}"
        )