import streamlit as st
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# Page configuration
st.set_page_config(
    page_title="RAGify Docs | A Developers Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("RAGify Docs")
st.markdown(
    "**_A Developers Tool_** — Scrape entire documentation recursively and ask questions using AI")
st.markdown("---")

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- 2. SESSION STATE SETUP ---
# We use session_state to keep variables alive when the app re-runs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- 3. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Documentation Link")
    website_url = st.text_input(
        "Paste the documentation URL:", placeholder="https://docs.langchain.com/oss/python/langchain/overview")

    scrape_btn = st.button("RAGify This Docs", type="primary")

# --- 4. CORE FUNCTIONS ---


@st.cache_resource
def get_embeddings():
    # Cached so we don't reload the heavy model on every interaction
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove excessive newlines for cleaner text
    return soup.text.strip()


# --- 5. MAIN LOGIC: SCRAPING ---
if scrape_btn:
    # Initialize resources
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    loader = RecursiveUrlLoader(website_url, extractor=get_bs4_extractor)

    all_chunks = []

    # VISUAL FEEDBACK: We use st.status to show the loop progress
    with st.status("🚀 Scraping and Chunking...", expanded=True) as status:
        st.write("Initializing loader...")

        # THE "BEST" STREAMING METHOD:
        # We iterate manually to update the UI and save memory
        count = 0
        for doc in loader.lazy_load():
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)  # Flattens the list efficiently

            # Update UI every few pages so it doesn't flicker too much
            count += 1
            if count % 1 == 0:
                st.write(
                    f"📄 Processed: {doc.metadata.get('source', 'Unknown Page')}")

        st.write(f"✅ Scraping Complete! Found {len(all_chunks)} chunks.")
        st.write("🧠 Building Vector Index (this may take a moment)...")

        # Create Vector Store
        vector_store = InMemoryVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings
        )

        # Create Retriever (MMR for diversity)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
        )

        # Create LLM & Chain
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

        prompt = ChatPromptTemplate.from_template("""
            You are a helpful and factual AI assistant.
            Use the following retrieved context to answer the user's question.
            If the answer is not found in the context, reply with:
            "I'm not sure based on the provided information."
            
            <context>
            {context}
            </context>
            
            Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # Save everything to session state
        st.session_state.rag_chain = rag_chain
        st.session_state.vector_store = vector_store

        status.update(label="✅ Knowledge Base Ready!",
                      state="complete", expanded=False)

# --- 6. CHAT INTERFACE ---

# Display previous chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_input := st.chat_input("Ask a question about the docs..."):
    if not st.session_state.rag_chain:
        st.error("⚠️ Please build the Knowledge Base first using the sidebar!")
    else:
        # User Message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input})
                answer = response["answer"]

                # Extract Sources
                sources = {doc.metadata.get('source')
                           for doc in response['context']}
                source_text = "\n\n**📍 Sources:**\n" + \
                    "\n".join([f"- {s}" for s in sources])

                final_response = answer + source_text
                st.markdown(final_response)

        # Save AI Message
        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_response})

else:
    st.info("👈 Paste a documentation URL in the sidebar and click 'RAGify This Docs' to transform it into an AI-powered knowledge base!")
