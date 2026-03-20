"""
RAGify Docs - A Developers Tool
Transform documentation into intelligent, queryable knowledge
"""

import streamlit as st
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

load_dotenv()


def bs4_extractor(html: str) -> str:
    """Extract text from HTML using BeautifulSoup"""
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


# Page configuration
st.set_page_config(
    page_title="RAGify Docs | A Developers Tool",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ RAGify Docs")
st.markdown(
    "**_A Developers Tool_** — Scrape entire documentation recursively and ask questions using AI")
st.markdown("---")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "scraped_url" not in st.session_state:
    st.session_state.scraped_url = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar for URL input
with st.sidebar:
    st.header("📚 Documentation Link")

    website_url = st.text_input(
        "Paste documentation root URL:",
        placeholder="https://docs.example.com",
        help="Enter the main URL (e.g., Get Started, Overview, Welcome page)"
    )

    scrape_button = st.button("🔄 RAGify This Docs", use_container_width=True)


# Main content area
if scrape_button and website_url:
    if st.session_state.scraped_url != website_url or st.session_state.rag_chain is None:
        try:
            with st.spinner("🔄 Scraping website..."):
                # Load documents
                loader = RecursiveUrlLoader(
                    website_url,
                    extractor=bs4_extractor
                )
                web_pages = []
                for doc in loader.lazy_load():
                    web_pages.append(doc)

                st.success(f"✅ Scraped {len(web_pages)} pages")

            # Convert to string
            web_page_content = "\n\n".join(
                [page.page_content for page in web_pages])

            with st.spinner("✂️ Splitting text..."):
                # Text splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=200
                )
                splitted_content = text_splitter.split_text(web_page_content)
                st.success(f"✅ Created {len(splitted_content)} chunks")

            with st.spinner("🧠 Creating embeddings..."):
                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Convert to documents
                list_of_documents = [
                    Document(page_content=text) for text in splitted_content
                ]

                # Vector store
                vector_store = InMemoryVectorStore.from_documents(
                    documents=list_of_documents,
                    embedding=embeddings
                )
                st.success("✅ Vector store created")

            # Retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            # LLM
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.2,
                reasoning_effort=None,
                reasoning_format=None,
                verbose=False
            )

            # Prompt
            prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful and factual AI assistant.
                Use the following retrieved context to answer the user's question.
                If the answer is not found in the context, reply with:
                "I'm not sure based on the provided information." DONT MAKE UP ANSWERS.

                <context>
                {context}
                </context>

                Question: {input}
                """
            )

            # Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)

            # Store in session state
            st.session_state.rag_chain = rag_chain
            st.session_state.scraped_url = website_url
            st.session_state.chat_history = []

        except Exception as e:
            st.error(f"❌ Error during scraping: {str(e)}")


# Chat interface
if st.session_state.rag_chain is not None:
    st.markdown("---")
    st.subheader("💬 Ask Questions")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_question = st.chat_input(
        "Ask a question about the scraped documentation...",
        key="user_input"
    )

    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get AI response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_question})
                ai_answer = response["answer"]

                # Add AI message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_answer
                })

                # Display AI message
                with st.chat_message("assistant"):
                    st.markdown(ai_answer)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

else:
    st.info("👈 Paste a documentation URL in the sidebar and click 'RAGify This Docs' to transform it into an AI-powered knowledge base!")
