import streamlit as st
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup
import re
import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

st.set_page_config(page_title="RAGify Docs", layout="wide")

# Title & Description
st.title("RAGify Docs")
st.caption(
    "A Developers Tool — Scrape entire documentation recursively and ask questions using AI")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Documentation Input
with st.sidebar:
    st.header("Documentation Link")
    url_input = st.text_input("Paste the documentation URL:",
                              placeholder="https://docs.langchain.com/oss/python/langchain/overview")

    if st.button("🚀 RAGify This Docs", use_container_width=True):
        if url_input:
            try:
                with st.spinner("🔄 Scraping documentation..."):
                    # Scraper setup
                    def bs4_extractor(html: str) -> str:
                        soup = BeautifulSoup(html, "lxml")
                        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

                    loader = RecursiveUrlLoader(
                        url_input, extractor=bs4_extractor)

                    # Chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200)
                    all_chunks = []

                    progress_bar = st.progress(0)
                    for idx, doc in enumerate(loader.lazy_load()):
                        chunks = text_splitter.split_documents([doc])
                        all_chunks.extend(chunks)
                        progress_bar.progress(
                            min((idx + 1) / 50, 1.0))  # Rough estimate

                    st.success(f"✅ Created {len(all_chunks)} chunks")

                with st.spinner("🧠 Initializing embeddings..."):
                    # Embeddings & Vector Store
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vector_store = InMemoryVectorStore.from_documents(
                        documents=all_chunks, embedding=embeddings)
                    st.success("✅ Vector store ready")

                with st.spinner("⚡ Setting up retrieval chain..."):
                    # Retriever
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 5, "fetch_k": 10, "lambda_mult": 0.5}
                    )

                    # LLM & Prompt
                    llm = ChatGroq(
                        model="openai/gpt-oss-120b", temperature=0.2)
                    prompt = ChatPromptTemplate.from_template(
                        """You are a helpful and factual AI assistant.
                            Use the following retrieved context to answer the user's question.

                            If the answer is not found in the context, reply with:
                            "I'm not sure based on the provided information."

                            <context>
                            {context}
                            </context>

                            Question: {input}
                            """)

                    # Chain
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state.rag_chain = create_retrieval_chain(
                        retriever, document_chain)
                    st.success("✅ RAG System Ready!")
                    st.session_state.messages = []  # Reset chat history

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a documentation URL")

# Main chat interface
if st.session_state.rag_chain:
    st.header("Ask a Question")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                st.caption(f"📍 Sources: {message['sources']}")

    # User input
    user_query = st.chat_input("Ask a question about the documentation...")

    if user_query:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(
                    {"input": user_query})
                answer = response['answer']
                sources = {doc.metadata.get('source')
                           for doc in response['context']}
                sources_str = ", ".join(sources)

                st.markdown(answer)
                st.caption(f"📍 Sources: {sources_str}")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_str
                })
else:
    st.info("👈 Paste a documentation URL in the sidebar and click 'RAGify This Docs' to get started!")
