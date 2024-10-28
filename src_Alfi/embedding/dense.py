from sentence_transformers import SentenceTransformer

from utility.read_config import get_config_from_path

from functools import lru_cache
import torch

dct_config = get_config_from_path("config.yaml")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = SentenceTransformer(dct_config["PRE_TRAINED_EMB"]["DENSE_MODEL_NAME"], device=device)
if device == 'cuda':
    encoder.half()  # This changes the model precision to FP16
EMB_DIM = encoder.get_sentence_embedding_dimension()

@lru_cache(maxsize=1000)
def compute_dense_vector(query_text: str) -> list[float]:
    """
    Computes a dense vector representation of the given query text.

    Args:
        query_text (str): The input text to convert into a dense vector.

    Returns:
        list[float]: A list representing the dense vector of the input text.
    """
    return encoder.encode(query_text, show_progress_bar=False).tolist()
