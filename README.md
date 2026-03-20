# RAGify Docs - A Developers Tool

Transform any documentation into an intelligent, queryable knowledge base powered by AI and RAG (Retrieval-Augmented Generation).

## 🎯 The Problem

Developers spend hours scrolling through endless documentation pages, searching for specific information scattered across multiple links and sub-pages. Reading 100+ pages line-by-line to find answers is inefficient and frustrating.

## ✨ The Solution

**RAGify Docs** automatically:

1. **Recursively scrapes** entire documentation structures (all child links, sub-links, and nested pages)
2. **Splits & embeds** the content intelligently
3. **Stores** everything in a vector database
4. **Answers questions** using AI powered by Retrieval-Augmented Generation

Just paste the documentation root URL (welcome page, overview, or get-started page), and ask natural language questions. Say goodbye to endless scrolling!

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (venv, conda, etc.)
- [Ollama](https://ollama.ai/) installed and running locally (for the LLM)
- Internet connection for web scraping

## 🔧 Installation

### 1. Clone or Download the Project

```bash
cd "Ragify Docs"
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Ensure Ollama is Running

The project uses **Ollama with `deepseek-v3.1:671b-cloud` model**. Make sure:

- Ollama is installed on your system
- Ollama service is running
- The model is available locally

To start Ollama:

```bash
ollama serve
```

To pull the model:

```bash
ollama pull deepseek-v3.1:671b-cloud
```

## 🎮 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Enter Documentation URL**
   - Paste the root URL of the documentation (e.g., `https://docs.example.com/getting-started`)
   - This should be the main entry point (Welcome page, Overview, Get Started, etc.)

2. **Click "RAGify This Docs"**
   - The tool will automatically discover and scrape all linked pages
   - You'll see progress updates for each stage:
     - 🔄 Scraping documentation
     - ✂️ Splitting text into chunks
     - 🧠 Creating embeddings

3. **Ask Questions**
   - Once processing is complete, use the chat interface
   - Ask natural language questions about the documentation
   - The AI will search the content and provide accurate answers
   - Chat history is maintained in the sidebar

### Example Queries

- *"How do I authenticate users?"*
- *"What are the available API endpoints?"*
- *"Show me an example of using the database connection"*
- *"What are the system requirements?"*

## 🏗️ Architecture & How It Works

```
Documentation URL
       ↓
   [RecursiveUrlLoader]
       ↓
   Web Pages (HTML)
       ↓
[BeautifulSoup Parser]
       ↓
   Clean Text Content
       ↓
[RecursiveCharacterTextSplitter]
       ↓
   Text Chunks (1200 chars, 200 overlap)
       ↓
[HuggingFace Embeddings]
       ↓
   Vector Embeddings
       ↓
[InMemoryVectorStore]
       ↓
   Searchable Knowledge Base
       ↓
   [User Question] → [Retriever] → [Top 5 Similar Chunks] → [LLM] → [Answer]
```

## 📁 Project Structure

```
Ragify Docs/
├── app.py                      # Main Streamlit application
├── main.py                     # CLI version (command-line interface)
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── README.md                  # This file
├── recursive_loader.ipynb     # Jupyter notebook for testing recursive loading
└── webbase_loader.ipynb       # Jupyter notebook for testing web loading
```

## 👤 Author

Built with ❤️ for developers who want to master new technologies faster.

## 🎓 Learning Resources

- [LangChain Documentation](https://python.langchain.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Ollama Guide](https://ollama.ai)
- [RAG Pattern](https://python.langchain.com/docs/use_cases/question_answering/)
- [Hugging Face Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

**Made by developers, for developers. Happy learning! 🚀**
