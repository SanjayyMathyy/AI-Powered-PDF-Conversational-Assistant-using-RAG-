import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain_classic.prompts import PromptTemplate
from loguru import logger


#PDF TEXT EXTRACTION
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    logger.info(f"Extracted text from {len(pdf_docs)} PDFs")
    return text


#  TEXT SPLITTING   
def split_text(text, chunk_size=1500, chunk_overlap=300):
    """
    Splits long text into overlapping chunks.
    Larger chunks preserve more context for LLM reasoning.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", "?", "!", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


#VECTOR STORE CREATION
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    logger.info("Vector store created successfully with MiniLM embeddings")
    return vectorstore


def get_contextual_prompt():
    template = """
You are an intelligent assistant that answers questions based on the provided document context.
Use only the context below to answer the question. If the answer is not found, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


#  RAG CHAIN CREATION 
def build_conversational_rag_chain(vectorstore):
    llm = Ollama(model="mistral") 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    contextual_prompt = get_contextual_prompt()

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": contextual_prompt}
    )

    logger.info("Conversational RAG chain built successfully!")
    return rag_chain
