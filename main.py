"""
RAGify Docs - A Developers Tool
Transform documentation into intelligent, queryable knowledge
"""

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# 1. SCRAPER SETUP

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


website_url = input("Enter the website URL to scrape: ")
loader = RecursiveUrlLoader(
    website_url, extractor=bs4_extractor)


# 2. STREAMING & SPLITTING

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)

all_chunks = []

# We use a loop to handle documents one-by-one (Streaming)
for doc in tqdm(loader.lazy_load(), bar_format="[INFO] {n_fmt} Pages Scraped."):

    # split_documents returns a LIST of chunks for this specific page
    chunks = text_splitter.split_documents([doc])

    # We use EXTEND to flatten those chunks into our main list
    all_chunks.extend(chunks)

print(f"[INFO] Created {len(all_chunks)} chunks.")


# 3. EMBEDDINGS & VECTOR STORE

print("[INFO] Embedding model initializing...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pass the list of chunks directly - metadata is preserved automatically!
vector_store = InMemoryVectorStore.from_documents(
    documents=all_chunks,
    embedding=embeddings
)

print("[INFO] Vector store created.")


# 4. RETRIEVER SETUP

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,

        "fetch_k": 10,  # it gives 10 candidates, but only pick the 5 most diverse ones

        "lambda_mult": 0.5  # 1.0 = Pure similarity, 0.0 = Pure diversity
    }
)


# 5. LLM & PROMPT TEMPLATE

llm = ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0.2)

# The chain expects {context} and {input}
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


# 6. THE CHAIN

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)


# 7. FINAL IMPLEMENTATION

print("\n🚀 RAG System Ready! (Type '0' to exit)")
while True:
    query = input("\nYou: ")
    if query == "0":
        break

    # One call handles retrieval, prompt formatting, and LLM generation
    response = rag_chain.invoke({"input": query})

    print(f"\n🧠 AI: {response['answer']}")

    # Optional: show which URLs were used
    sources = {doc.metadata.get('source') for doc in response['context']}
    print(f"📍 Sources: {', '.join(sources)}")
