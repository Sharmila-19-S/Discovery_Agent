import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MODEL_NAME, DEVICE, DTYPE


class ModelLoader:
    """
    Singleton Model Loader
    Loads the model only once and reuses it across all agents.
    """

    _model = None
    _tokenizer = None

    @classmethod
    def load(cls):

        if cls._model is None:

            print("=" * 60)
            print("Loading Qwen 2.5-3B-Instruct...")
            print("=" * 60)

            cls._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            cls._model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=DTYPE
            )

            cls._model.to(DEVICE)
            cls._model.eval()

            print("✅ Model Loaded Successfully")
            print("=" * 60)

        return cls._model, cls._tokenizer