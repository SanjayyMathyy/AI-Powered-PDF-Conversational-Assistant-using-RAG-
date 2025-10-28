import fitz  # PyMuPDF
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.llms import ollama
from loguru import logger
from PIL import Image
import io

# -------------------- Extract text and images per page -------------------- #
def extract_pdf_pages(pdf_docs):
    """
    Returns a list of pages with text and images:
    [{"text": "...", "images": [PIL.Image, ...]}, ...]
    """
    page_data = []

    for pdf in pdf_docs:
        pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
        logger.info(f"Opened PDF: {pdf.name} with {len(pdf_file)} pages")

        for page_index in range(len(pdf_file)):
            page = pdf_file.load_page(page_index)
            text = page.get_text("text")
            images = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)

            page_data.append({"text": text or "", "images": images})

        pdf_file.close()

    logger.info(f"Extracted {len(page_data)} pages from {len(pdf_docs)} PDFs")
    return page_data


# -------------------- Split text into chunks -------------------- #
def get_text_chunks(page_data):
    all_text = [page["text"] for page in page_data]
    full_text = "\n".join(all_text)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    logger.info(f"Split the text into {len(chunks)} chunks")
    return chunks


# -------------------- Create vector store -------------------- #
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# -------------------- Build history-aware RAG chain -------------------- #
def build_conversational_rag_chain(vectorstore):
    llm = ollama.Ollama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return rag_chain


# -------------------- Retrieve relevant images -------------------- #
def get_relevant_images(user_question, page_data, top_k=2):
    """
    Returns images from the top-k pages most relevant to the user question.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    question_embedding = embeddings.embed([user_question])[0]

    # Compute similarity
    page_scores = []
    for idx, page in enumerate(page_data):
        page_embedding = embeddings.embed([page["text"] or ""])[0]
        # Cosine similarity
        score = sum([q * p for q, p in zip(question_embedding, page_embedding)])
        page_scores.append((score, idx))

    # Get top_k pages
    page_scores.sort(reverse=True)
    relevant_images = []
    for _, idx in page_scores[:top_k]:
        relevant_images.extend(page_data[idx]["images"])

    return relevant_images
