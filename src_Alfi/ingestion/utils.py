from typing import List
import markdownify
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text: str) -> str:
    """Removes stopwords from the text.

    Args:
        text (str): The text from which to remove stopwords.

    Returns:
        str: The text without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = " ".join(word for word in words if word.lower() not in stop_words)
    return filtered_text

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

def extract_title(html_file_path: str) -> str:
    """Extracts the title from the HTML file.

    Args:
        html_file_path (str): The path to the HTML file.

    Returns:
        str: The extracted title or an empty string if not found.
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.text.strip()
    return ""

def extract_author(html_file_path: str) -> str:
    """Extracts the author from the HTML file.

    Args:
        html_file_path (str): The path to the HTML file.

    Returns:
        str: The extracted author or an empty string if not found.
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
        
        # Adjust the selector based on your HTML structure
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and 'content' in author_tag.attrs:
            return author_tag['content'].strip()
        
        # Alternatively, you can try to extract from <div> or <span> tags if needed
        authors = soup.find_all('div', class_='author')  # Example class name; adjust accordingly
        if authors:
            return ', '.join([author.text.strip() for author in authors])
    return ""

def extract_publication_date(html_file_path: str) -> str:
    """Extracts the publication date from the HTML file.

    Args:
        html_file_path (str): The path to the HTML file.

    Returns:
        str: The extracted publication date or an empty string if not found.
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
        
        # Adjust the selector based on your HTML structure
        date_tag = soup.find('meta', attrs={'name': 'date'})
        if date_tag and 'content' in date_tag.attrs:
            return date_tag['content'].strip()
        
        # Example for extracting from a <div> or <span> if date is in a specific format
        date_div = soup.find('div', class_='published-date')  # Example class name; adjust accordingly
        if date_div:
            return date_div.text.strip()
    return ""

def chunk_text_original(text: str, chunk_size: int = 300) -> List[str]:
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
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks

def preprocess_chunk(chunk: str) -> str:
    """Lowercases, removes punctuation, and lemmatizes the given chunk."""
    # Lowercase the chunk
    chunk = chunk.lower()

    # Remove punctuation
    chunk = chunk.translate(str.maketrans('', '', string.punctuation))

    # Lemmatize the words in the chunk
    words = chunk.split()
    lemmatized_chunk = " ".join(lemmatizer.lemmatize(word) for word in words)
    
    return lemmatized_chunk

def chunk_text(html_file_path: str, chunk_size: int = 300) -> List[str]:
    """Processes the HTML file by converting it to markdown, removing stopwords, and chunking."""
    # Convert HTML to Markdown
    markdown_text = convert_html_to_markdown(html_file_path)
    
    # Remove stopwords
    text_without_stopwords = remove_stopwords(markdown_text)

    # Chunk the text
    chunks = chunk_text_original(text_without_stopwords, chunk_size)

    # Apply additional preprocessing to each chunk
    processed_chunks = [preprocess_chunk(chunk) for chunk in chunks]

    return processed_chunks
