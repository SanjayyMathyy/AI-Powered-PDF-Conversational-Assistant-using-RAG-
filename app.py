import streamlit as st
from rag_process import (
    extract_pdf_pages,
    get_text_chunks,
    get_vector_store,
    build_conversational_rag_chain,
    get_relevant_images
)
from loguru import logger

# -------------------- Handle User Input -------------------- #
def handle_user_input(user_question):
    if st.session_state.rag_chain is None:
        st.warning("⚠️ Upload and process PDFs first!")
        return

    answer = ""
    images_to_show = []

    try:
        # If user asks for image, get relevant images
        if "image" in user_question.lower() or "diagram" in user_question.lower():
            images_to_show = get_relevant_images(
                user_question, st.session_state.page_data, top_k=2
            )
            answer = f"Found {len(images_to_show)} relevant image(s)."
        else:
            response = st.session_state.rag_chain({"question": user_question})
            answer = response.get("answer", "No answer returned.")
    except Exception as e:
        logger.error(f"Error during chain execution: {e}")
        answer = "❌ Error generating answer."

    st.session_state.history.append({"role": "user", "content": user_question})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Save images for display
    if images_to_show:
        st.session_state.pdf_images = images_to_show


# -------------------- Display Chat -------------------- #
def display_chat():
    st.markdown("### 💬 Conversation")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.history:
            color = "#1B309E" if message["role"] == "user" else "#310861"
            align = "right" if message["role"] == "user" else "left"
            st.markdown(
                f"""<div style="background-color:{color}; color:white; padding:10px;
                            border-radius:10px; margin:5px 0; text-align:{align};">
                    <b>{message['role'].capitalize()}:</b> {message['content']}
                </div>""",
                unsafe_allow_html=True,
            )


# -------------------- Main App -------------------- #
def main():
    st.set_page_config(page_title="Chat on PDF", page_icon=":guardsman:", layout="wide")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "page_data" not in st.session_state:
        st.session_state.page_data = []
    if "pdf_images" not in st.session_state:
        st.session_state.pdf_images = []

    st.title("📘 Context-Aware PDF Chatbot")

    col1, col2 = st.columns([2, 1])

    # ---------------- Left column: Chat ---------------- #
    with col1:
        display_chat()
        st.markdown("---")

        with st.form(key="user_input_form", clear_on_submit=False):
            user_question = st.text_area(
                "💭 Type your question here:",
                height=80,
                placeholder="Ask about the text or request an image..."
            )
            send_button = st.form_submit_button("Send")

            if send_button and user_question.strip():
                handle_user_input(user_question.strip())
                st.rerun()

    # ---------------- Right column: PDF Upload ---------------- #
    with col2:
        st.header("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        if st.button("⚙️ Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    st.session_state.page_data = extract_pdf_pages(pdf_docs)
                    text_chunks = get_text_chunks(st.session_state.page_data)
                    vectorstore = get_vector_store(text_chunks)
                    st.session_state.rag_chain = build_conversational_rag_chain(vectorstore)
                    st.success("✅ RAG chain ready! You can now start chatting.")
            else:
                st.warning("Please upload at least one PDF.")

        # Display images if any were retrieved for the question
        if st.session_state.pdf_images:
            st.markdown("### 🖼️ Relevant Images")
            for img in st.session_state.pdf_images:
                st.image(img, use_container_width=True)


if __name__ == "__main__":
    main()
