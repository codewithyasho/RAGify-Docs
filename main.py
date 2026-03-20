"""
A Multi Website Scraper
"""

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
from bs4 import BeautifulSoup
import re


# WEBSITE SCRAPER USING RECURSIVEURLLOADER
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


website_url = input("Enter the website URL to scrape: ")
loader = RecursiveUrlLoader(
    website_url, extractor=bs4_extractor)

web_pages = []
for doc in loader.lazy_load():
    web_pages.append(doc)

# converting list of pages to a single string, useful for text splitting and embedding
web_page_content = "\n\n".join([page.page_content for page in web_pages])


# TEXT SPLITTING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=200)

splitted_content = text_splitter.split_text(web_page_content)


# EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


# converting list of pages to list of langchain documents, useful for creating vector store and retrieval
list_of_documents = [Document(page_content=text) for text in splitted_content]


# CREATING IN-MEMORY VECTOR STORE
vector_store = InMemoryVectorStore.from_documents(
    documents=list_of_documents, embedding=embeddings)


# RETRIEVER
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# LLM
llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud",
    temperature=0.2,
)

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
    """)

# CREATING CHAINS
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# FINAL TESTING
while True:
    user_inp = input("\nAsk a question: ")
    if user_inp.lower() in ["exit", "quit"]:
        break

    response = rag_chain.invoke({"input": user_inp})
    print("\n🧠 AI Answer:")
    print(response["answer"])
