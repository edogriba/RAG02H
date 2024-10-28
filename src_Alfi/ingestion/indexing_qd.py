import os
from logging import getLogger

from qdrant_client import models

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from ingestion.utils import chunk_text, convert_html_to_markdown, extract_title, extract_author, extract_publication_date
from ingestion.vdb_wrapper import LoadInVdb

logger = getLogger("ingestion")


def main_indexing(
    loader: LoadInVdb, is_fresh_start: bool, html_folder_path: str, batch_size = 32
) -> None:
    """
    Indexes HTML files by converting them to markdown and adding the resulting chunks to the vector database.

    Args:
        loader (LoadInVdb): The LoadInVdb instance used to load data into the vector database.
        is_fresh_start (bool): Indicates whether to start fresh with a new collection.
        html_folder_path (str): The path to the folder containing HTML files to be indexed.
    """
    loader.setup_collection(is_fresh_start=is_fresh_start)

    for f in os.listdir(html_folder_path):
        html_file_path = os.path.join(html_folder_path, f)
        if not html_file_path.endswith(".html"):
            logger.info(f"Indexing in vect skipped for file: {html_file_path}")
            continue

        # Convert HTML to markdown
        #markdown_text = convert_html_to_markdown(html_file_path)

        # Chunk the Markdown text
        chunks = chunk_text(html_file_path)

        # add the chunks to the vector db

        if len(chunks) > 0:
            logger.info(f"Starting indexing in vect db for: {html_file_path}")

            # Prepare lists for batch processing
            dense_vectors, sparse_vectors, payloads = [], [], []

            for chunk in chunks:
                dense_vectors.append(compute_dense_vector(query_text=chunk))
                sparse_vectors.append(models.SparseVector(**compute_sparse_vector(query_text=chunk)))
                payloads.append({"text": chunk, "source_file": html_file_path, "chunk_size": len(chunk), "title": extract_title(html_file_path),"author": extract_author(html_file_path),"publication_date": extract_publication_date(html_file_path)})

                # If the batch is full, add to the vector database
                if len(dense_vectors) >= batch_size:
                    loader.add_to_collection(dense_vectors=dense_vectors, sparse_vectors=sparse_vectors, payloads=payloads)
                    # Reset the lists for the next batch
                    dense_vectors, sparse_vectors, payloads = [], [], []

            # Index any remaining chunks in the last batch
            if dense_vectors:
                loader.add_to_collection(dense_vectors=dense_vectors, sparse_vectors=sparse_vectors, payloads=payloads)

            logger.info(f"Indexing in vect db ended for: {html_file_path}")
        else:
            logger.info(f"Indexing in vect db skipped (no chunks) for: {html_file_path}")


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    from utility.read_config import get_config_from_path
    logger.setLevel('INFO')

    dct_config = get_config_from_path("config.yaml")
    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    COLLECTION_NAME = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    COLL_FRESH_START = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    loader = LoadInVdb(client=client, coll_name=COLLECTION_NAME)

    main_indexing(
        loader=loader,
        is_fresh_start=COLL_FRESH_START,
        html_folder_path=html_folder_path,
    )
