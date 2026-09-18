
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def generate(system_prompt, user_prompt, temperature=0.1):

    print("Connecting to Ollama...")
    print("Model:", MODEL_NAME)

    prompt = f"""
<System>
{system_prompt}

<User>
{user_prompt}
"""

    payload = {
        "model": MODEL_NAME,
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

        return data.get("response")

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