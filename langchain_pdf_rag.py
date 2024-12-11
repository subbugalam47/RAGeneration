# pip install langchain_community
from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate



# Load PDF document
loader = PyPDFLoader("/content/book.pdf")
documents = loader.load()



# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

model_name = "all-MiniLM-L6-v2"
hf_embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# Use FAISS to create a vector store for fast retrieval
vector_store = FAISS.from_documents(splits, hf_embedding_model)
retriever = vector_store.as_retriever()

# Define your prompt template for LLM
prompt_template = """
Given the context, answer the question.

Context: {context}
Question: {question}
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


# Step 3: Define your RAG chain using LangChain components
rag_chain = (
    {
        "context": retriever,  # Retrieve context using FAISS retriever
        "question": RunnablePassthrough()  # Pass the question as is
    }
    | prompt  # Apply the prompt template
    | llm  # Run the LLM
    | StrOutputParser()  # Parse the output
)

# Step 4: Use the chain
question = "summarize the book"
response = rag_chain.invoke(question)

print(response)
