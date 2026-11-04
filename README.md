# PDF Knowledge Assistant – RAG Chatbot

An AI-powered PDF question-answering application that allows users to upload PDF documents and ask questions in natural language. Built using Retrieval-Augmented Generation (RAG), the application retrieves relevant document content and generates context-aware answers with source and page references.

## 🚀 Features

- **Multiple PDF Uploads:** Upload up to 3 PDF documents per session.
- **PDF Validation:** Handles empty, corrupted, unreadable, duplicate, and oversized PDF files.
- **Document Processing:** Extracts text, splits it into chunks, and generates vector embeddings.
- **Semantic Search:** Retrieves relevant information using Sentence Transformers and ChromaDB.
- **Hybrid Retrieval:** Combines semantic and keyword-based matching to improve document retrieval.
- **Context-Aware Answers:** Uses Groq LLM API to generate answers based on retrieved PDF content.
- **Question Rewriting:** Handles short questions, typos, and conversational follow-up questions.
- **Source References:** Displays the PDF filename and page number used to generate answers.
- **Session Isolation:** Maintains separate document collections and conversations for different users.
- **Chat History:** Stores and restores conversations using SQLite.
- **Incremental PDF Management:** Add or remove documents without reprocessing the entire collection.
- **Automatic Cleanup:** Removes inactive ChromaDB collections after 24 hours.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Embedding Model | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Conversation Storage | SQLite |
| Deployment | Streamlit Community Cloud |

## 🏗️ Architecture

1. **PDF Upload:** Users upload PDF documents through the Streamlit interface.
2. **Text Extraction:** PyMuPDF extracts textual content from uploaded PDFs.
3. **Chunking:** Extracted text is divided into smaller, manageable chunks.
4. **Embedding Generation:** Sentence Transformers converts chunks into numerical vector representations.
5. **Vector Storage:** ChromaDB stores embeddings along with document metadata.
6. **Question Processing:** User questions are processed and rewritten when necessary.
7. **Hybrid Retrieval:** Relevant chunks are retrieved using semantic and keyword-based search.
8. **Answer Generation:** Retrieved context is passed to the Groq LLM to generate a grounded response.
9. **Source Attribution:** Relevant PDF filenames and page numbers are displayed alongside answers.

## 📂 Project Structure

```text
pdf-rag-chatbot/
│
├── app/
│   ├── chat_history.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── pdf_loader.py
│   ├── rag_pipeline.py
│   └── vector_store.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── streamlit_app.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/srinithp04/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add:

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key from [Groq Console](https://console.groq.com/).

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

## 📋 Application Limits

| Configuration | Limit |
|---|---|
| PDFs per session | 3 |
| Individual PDF size | 10 MB |
| Combined PDF size | 25 MB |
| Inactive collection cleanup | 24 hours |

## 🌐 Live Demo

[PDF Knowledge Assistant](https://ask-your-pdf-rag.streamlit.app/)

## 👨‍💻 Developer

**Srinith P**

- LinkedIn: [linkedin.com/in/srinith-p](https://linkedin.com/in/srinith-p)
- GitHub: [github.com/srinithp04](https://github.com/srinithp04)

---

*Built to explore Retrieval-Augmented Generation, vector search, and practical LLM application development.*
