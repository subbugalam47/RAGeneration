from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.runnables import RunnablePassthrough
from langchain.output_parsers import StrOutputParser
from typing import List


def load_pdf_document(file_path: str) -> List:
    """
    Load a PDF document from the specified path.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List: A list of document pages.
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_document_into_chunks(documents: List, chunk_size: int = 1000, chunk_overlap: int = 100) -> List:
    """
    Split the document into smaller chunks for processing.

    Args:
        documents (List): A list of document pages to split.
        chunk_size (int): The size of each chunk (default is 1000 characters).
        chunk_overlap (int): The number of overlapping characters between chunks (default is 100).

    Returns:
        List: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def create_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Create an embedding model using Hugging Face's sentence transformer.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        HuggingFaceEmbeddings: An instance of the embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(documents: List, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Create a FAISS vector store for fast document retrieval.

    Args:
        documents (List): A list of document chunks.
        embedding_model (HuggingFaceEmbeddings): The embedding model to use for vectorization.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    return FAISS.from_documents(documents, embedding_model)


def create_prompt_template() -> PromptTemplate:
    """
    Create a prompt template for the language model.

    Returns:
        PromptTemplate: A LangChain prompt template instance.
    """
    prompt_template = """
    Given the context, answer the question.

    Context: {context}
    Question: {question}
    """
    return PromptTemplate(input_variables=["context", "question"], template=prompt_template)


def create_rag_chain(retriever, prompt: PromptTemplate) -> LLMChain:
    """
    Create a Retrieval-Augmented Generation (RAG) chain using the provided retriever and prompt template.

    Args:
        retriever: The retriever to fetch context.
        prompt (PromptTemplate): The prompt template to use.

    Returns:
        LLMChain: A LangChain RAG chain instance.
    """
    return (
        {
            "context": retriever,  # Retrieve context using FAISS retriever
            "question": RunnablePassthrough()  # Pass the question as is
        }
        | prompt  # Apply the prompt template
        | LLMChain()  # Run the LLM
        | StrOutputParser()  # Parse the output
    )


def ask_question(chain: LLMChain, question: str) -> str:
    """
    Ask a question using the RAG chain.

    Args:
        chain (LLMChain): The RAG chain to use.
        question (str): The question to ask.

    Returns:
        str: The response from the model.
    """
    return chain.invoke(question)


def main():
    # Load PDF document
    documents = load_pdf_document("/content/book.pdf")

    # Split documents into smaller chunks
    chunks = split_document_into_chunks(documents)

    # Create embedding model
    model_name = "all-MiniLM-L6-v2"
    embedding_model = create_embedding_model(model_name)

    # Create FAISS vector store
    vector_store = create_vector_store(chunks, embedding_model)
    retriever = vector_store.as_retriever()

    # Create prompt template
    prompt = create_prompt_template()

    # Create RAG chain
    rag_chain = create_rag_chain(retriever, prompt)

    # Ask question and get the response
    question = "summarize the book"
    response = ask_question(rag_chain, question)

    print(response)


if __name__ == "__main__":
    main()
