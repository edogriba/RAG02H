import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.utils import chunk_text, convert_html_to_markdown

"""
This code snippet is designed to process a given HTML document by converting it
into Markdown format, chunking the text into smaller pieces, generating embeddings
for those chunks using a pre-trained model, and finally saving those embeddings into 
a FAISS index for efficient similarity search. 
"""


def save_chunks_to_faiss(chunks: list[str], index_file: str, nlist: int = 100, batch_size: int = 32) -> None:
    """Saves text chunks to a FAISS index after generating their embeddings.

    Args:
        chunks (List[str]): A list of text chunks to be embedded and indexed.
        index_file (str): The path to the file where the FAISS index will be saved.
        nlist (int): Number of clusters for IVF indexing.
        batch_size (int): Size of each batch for embedding generation.
    """
    # Load a pre-trained transformer model for embedding generation
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a FAISS index
    dimension = model.get_sentence_embedding_dimension()  # Get embedding dimension
    quantizer = faiss.IndexFlatL2(dimension)  # Use L2 distance metric for quantizer
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

    # Generate embeddings in batches
    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        embeddings = model.encode(batch_chunks)

        # Train the index with the first batch only
        if i == 0:
            index.train(embeddings)  # Train on the first batch

        # Add embeddings to the index
        index.add(embeddings)

    # Save the FAISS index to file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

    # Save the original chunks to a pickle file
    with open(index_file + "_pkl", "wb") as file:
        pickle.dump(chunks, file)


if __name__ == "__main__":
    # example of loading one file in faiss index
    project_root = os.getenv("MY_HOME", ".")

    # Set the path of the HTML file
    html_file_path = os.path.join(project_root, "data", "docs", "2401.02900v1.html")

    # Set the path to save FAISS index
    faiss_index_file = os.path.join(
        project_root, "embeddings", "faiss_index", "index.faiss"
    )

    # Convert HTML to markdown
    markdown_text = convert_html_to_markdown(html_file_path)

    # Chunk the Markdown text
    chunks = chunk_text(markdown_text)

    # Save the chunks to FAISS index
    save_chunks_to_faiss(chunks, faiss_index_file)
