from typing import List
from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import PyPDF2
import faiss
import numpy as np


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Splits the text into smaller chunks of a given size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index for fast similarity search.

    Args:
        embeddings (np.ndarray): The embeddings matrix to add to the index.

    Returns:
        faiss.IndexFlatL2: The FAISS index object.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_relevant_chunks(
    query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, chunks: List[str], top_k: int = 3
) -> List[str]:
    """
    Retrieves the most relevant chunks of text for a given query.

    Args:
        query (str): The query string.
        model (SentenceTransformer): The model to encode the query.
        index (faiss.IndexFlatL2): The FAISS index to search for similar embeddings.
        chunks (List[str]): The list of text chunks.
        top_k (int): The number of top relevant chunks to retrieve.

    Returns:
        List[str]: The most relevant text chunks.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]


def query_llm_with_context(llm: ChatOllama, context: str, query: str) -> str:
    """
    Queries the language model with a given context and question.

    Args:
        llm (ChatOllama): The language model instance.
        context (str): The context string to provide.
        query (str): The question to ask.

    Returns:
        str: The model's response.
    """
    prompt = f"Context: {context}\n\nQuestion: {query}"
    return llm.invoke(prompt)


# Main logic
if __name__ == "__main__":
    # Load PDF and extract text
    pdf_path = '/content/book.pdf'
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = chunk_text(pdf_text)

    # Load the model and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # Create FAISS index
    faiss_index = create_faiss_index(np.array(embeddings))

    # Define the query
    query = "What are the properties of probability models?"

    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, model, faiss_index, chunks)

    # Combine chunks into context
    context = "\n".join(relevant_chunks)

    # Load language model (mock example)
    llm = ChatOllama()  # Replace with your actual LLM instance

    # Query LLM with context
    answer = query_llm_with_context(llm, context, query)

    print("Answer:", answer)
