from typing import List
import nltk

nltk.download('punkt')

import markdownify
from bs4 import BeautifulSoup


def convert_html_to_markdown(html_file: str) -> str:
    """Reads an HTML file and converts its content to Markdown format.

    Args:
        html_file (str): The path to the HTML file to be converted.

    Returns:
        str: The converted Markdown content.
    """
    with open(html_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, "html.parser")

    # Convert HTML to Markdown
    markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")

    return markdown_content


def chunk_text(text: str, max_chunk_size: int = 300) -> List[str]:
    """
    ---- PLACEHOLDER VERSION ----
    ---- TO BE MODIFIED ----
    Chunks the input text into smaller segments of specified size.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum number of words per chunk. Default is 300.

    Returns:
        List[str]: A list of text chunks.
    """
    
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Check if adding this sentence would exceed the chunk size
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            # If so, add the current chunk to chunks and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Otherwise, keep adding sentences to the current chunk
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it's non-empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks