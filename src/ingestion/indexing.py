import os
import faiss
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from sentence_transformers import SentenceTransformer
import numpy as np


# Function to read the HTML file and convert it to markdown using markdown-it-py
def convert_html_to_markdown(html_file):
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text from the HTML content
    cleaned_html = str(soup)

    # Convert HTML to Markdown using markdown-it-py
    md = MarkdownIt()
    markdown_content = md.render(cleaned_html)

    return markdown_content


# Function to chunk the markdown content
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


# Function to save embeddings to FAISS index
def save_chunks_to_faiss(chunks, index_file):
    # Load a pre-trained transformer model for embedding generation
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each chunk
    embeddings = model.encode(chunks)

    # Create a FAISS index and add embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(np.array(embeddings))

    # Save the FAISS index to file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")


if __name__ == "__main__":
    # Set the path of the HTML file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    html_file_path = os.path.join(project_root, 'data', 'docs', '2303.15936v2.html')

    # Set the path to save FAISS index
    faiss_index_file = os.path.join(project_root, 'embeddings', 'faiss_index', 'index.faiss')

    # Convert HTML to markdown
    markdown_text = convert_html_to_markdown(html_file_path)

    # Chunk the Markdown text
    chunks = chunk_text(markdown_text)

    # Save the chunks to FAISS index
    save_chunks_to_faiss(chunks, faiss_index_file)
