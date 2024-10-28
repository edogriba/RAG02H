import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from functools import lru_cache

from utility.read_config import get_config_from_path

dct_config = get_config_from_path("config.yaml")

tokenizer = AutoTokenizer.from_pretrained(
    dct_config["PRE_TRAINED_EMB"]["SPARSE_MODEL_NAME"],
    clean_up_tokenization_spaces=True,
)
# Check if CUDA is available and set the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model, with half-precision only if running on a CUDA-enabled device
model = AutoModelForMaskedLM.from_pretrained(
    dct_config["PRE_TRAINED_EMB"]["SPARSE_MODEL_NAME"]
)

# Apply half precision if on CUDA
if device == "cuda":
    model = model.half()

# Move the model to the selected device
model = model.to(device)

# TODO: this implementation is just a placeholder, to be modified! watch out for the max len param!
def __compute_vector(text) -> tuple[torch.Tensor, dict]:
    """
    Computes a vector from the given text using the model and tokenizer.
    Taken from Qdrant documentation: https://qdrant.tech/articles/sparse-vectors/

    Args:
    text (str): The input text to compute the vector for.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The computed vector.
            - dict: The tokens used for the computation.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    output = model(**tokens)
    """ 
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()
    
    """
    masked_logits = output.logits * tokens.attention_mask.unsqueeze(-1)
    relu_log = torch.log1p(torch.relu(masked_logits))
    max_val, _ = torch.max(relu_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens



@lru_cache(maxsize=1000)
def compute_sparse_vector(query_text: str) -> dict[str, list[float]]:
    """
    Computes a sparse vector representation of the query text.

    Args:
        query_text (str): The text to be converted into a sparse vector.

    Returns:
        dict: A dictionary containing the sparse vector indices and values.
    """
    q_vec, q_tokens = __compute_vector(query_text)
    out = {"indices": q_vec.nonzero().numpy().flatten().tolist()}
    out["values"] = q_vec.detach().numpy()[out["indices"]].tolist()
    return out
"""
Explanation:

    q_vec.nonzero() finds the indices of non-zero values, which are the "sparse" elements representing meaningful features in the text.
    q_vec.detach().numpy() converts q_vec to a NumPy array (detached from the computation graph to prevent unnecessary memory retention).
    out is a dictionary where "indices" stores the positions of non-zero elements and "values" stores the corresponding values.

Result: out, a dictionary containing the sparse vector as {indices: [...], values: [...]}. This format allows efficient storage and retrieval, as it avoids saving large zero-filled arrays.

"""
