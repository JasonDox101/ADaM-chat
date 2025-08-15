# ADaM RAG System

An intelligent assistant system for ADaM (Analysis Data Model) clinical data analysis based on Retrieval-Augmented Generation (RAG) technology.

## 🎯 Project Overview

ADaM RAG System is an intelligent Q&A system specifically designed for clinical data analysts and biostatisticians. Built upon the CDISC ADaM Implementation Guide (ADaMIG v1.3), it can answer professional questions about ADaM dataset structures, variable definitions, programming specifications, and more.

### Core Features

- 🤖 **Intelligent Q&A**: Professional ADaM knowledge Q&A based on RAG technology
- 🌐 **Bilingual Support**: Supports both Chinese and English interactive interfaces
- 📚 **Document Retrieval**: Precise document retrieval based on ADaMIG v1.3
- 💬 **Conversation Memory**: Supports context-aware multi-turn conversations
- 🎨 **Modern Interface**: Responsive web interface based on Shiny
- 📊 **Confidence Scoring**: Provides credibility assessment for each answer

## 🏗️ System Architecture
ADaM RAG System
```plaintext
├── Frontend Interface (Shiny Web UI)
├── Question Processor
├── RAG Core System
│   ├── Context Analysis
│   ├── Question Classification
│   ├── Document Retrieval
│   ├── Answer Generation
│   ├── Answer Summarization
│   └── Answer Validation
├── Vector Database (FAISS Vector Store)
└── LLM Client (Qwen LLM Client)
```


## 📋 System Requirements

- Python 3.8+
- 8GB+ RAM (recommended)
- Alibaba Cloud DashScope API Key

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone <repository-url>
cd ADaM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file and configure the API key:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 4. Start the System

```bash
python app.py
```

The system will start at `http://localhost:8000`.

## 📁 Project Structure

```plaintext
Trae

我理解您需要英语版本的README.md文档。我将为您创建一个完整的英文版README.md：

```
# ADaM RAG System

An intelligent assistant system for ADaM 
(Analysis Data Model) clinical data analysis 
based on Retrieval-Augmented Generation (RAG) 
technology.

## 🎯 Project Overview

ADaM RAG System is an intelligent Q&A system 
specifically designed for clinical data analysts 
and biostatisticians. Built upon the CDISC ADaM 
Implementation Guide (ADaMIG v1.3), it can answer 
professional questions about ADaM dataset 
structures, variable definitions, programming 
specifications, and more.

### Core Features

- 🤖 **Intelligent Q&A**: Professional ADaM 
knowledge Q&A based on RAG technology
- 🌐 **Bilingual Support**: Supports both Chinese 
and English interactive interfaces
- 📚 **Document Retrieval**: Precise document 
retrieval based on ADaMIG v1.3
- 💬 **Conversation Memory**: Supports 
context-aware multi-turn conversations
- 🎨 **Modern Interface**: Responsive web 
interface based on Shiny
- 📊 **Confidence Scoring**: Provides credibility 
assessment for each answer

## 🏗️ System Architecture

```
ADaM RAG System
├── Frontend Interface (Shiny Web UI)
├── Question Processor
├── RAG Core System
│   ├── Context Analysis
│   ├── Question Classification
│   ├── Document Retrieval
│   ├── Answer Generation
│   ├── Answer Summarization
│   └── Answer Validation
├── Vector Database (FAISS Vector Store)
└── LLM Client (Qwen LLM Client)

```

## 📋 System Requirements

- Python 3.8+
- 8GB+ RAM (recommended)
- Alibaba Cloud DashScope API Key

## 🚀 Quick Start

### 1. Clone the Project

```bash
git clone <repository-url>
cd ADaM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file and configure the API key:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

### 4. Start the System

```bash
python app.py
```

The system will start at `http://localhost:8000`.

## 📁 Project Structure

```
ADaM/
├── app.py                 # Main application entry
├── requirements.txt       # Project dependencies
├── ADaMIG_v1.3.pdf       # ADaM Implementation Guide document
├── config/
│   └── settings.py       # System configuration
├── rag/
│   └── system.py         # RAG core system
├── handlers/
│   ├── question_processor.py  # Question processor
│   ├── message_handlers.py    # Message handlers
│   ├── ui_components.py       # UI components
│   └── utils.py              # Utility functions
├── llm/
│   └── qwen_client.py    # LLM client
├── vector_store/
│   └── manager.py        # Vector store manager
├── data/
│   └── processor.py      # Data processor
├── graph/
│   └── state.py          # State management
└── static/
└── styles.css        # Style files
```

## 🔧 Configuration

### Main Configuration Items

- `DASHSCOPE_API_KEY`: Alibaba Cloud DashScope API key
- `MODEL_NAME`: LLM model to use (default: qwen3-30b-a3b-thinking-2507)
- `ADAM_PDF_PATH`: ADaM document path
- `VECTOR_STORE_PATH`: Vector database storage path
- `RETRIEVAL_K`: Number of documents to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Similarity threshold (default: 0.7)

## 💡 Usage Guide

### Supported Question Types

1. **Variable-related**: ADaM variable definitions, naming conventions, data types, etc.
2. **Dataset-related**: ADSL, BDS, OCCDS and other dataset structures and requirements
3. **General concepts**: ADaM standards, implementation guides, best practices, etc.

### Example Questions

- "What are the key variables that must be included in ADSL datasets?"
- "What is the difference between AVAL and AVALC in BDS datasets?"
- "How to implement data traceability in ADaM datasets?"
- "Explain the concept of analysis flags in ADaM"

### Language Switching

Click the language toggle button in the upper right corner of the interface to switch between Chinese and English.

## 🛠️ Technology Stack

- **Frontend Framework**: Shiny for Python
- **LLM Service**: Alibaba Cloud DashScope (Qwen Model)
- **Vector Database**: FAISS
- **Document Processing**: LangChain
- **Embedding Model**: Sentence Transformers
- **Workflow Engine**: LangGraph

## 📊 System Features

### RAG Workflow

1. **Context Analysis**: Analyze conversation history and identify relevant context
2. **Question Classification**: Classify questions into variable, dataset, or general types
3. **Document Retrieval**: Retrieve relevant document fragments from vector database
4. **Answer Generation**: Generate detailed answers based on retrieved content
5. **Answer Summarization**: Generate concise answer summaries
6. **Answer Validation**: Evaluate answer quality and confidence

### Intelligent Features

- **Context Awareness**: Supports multi-turn conversations with understanding of question relationships
- **Professional Terminology**: Accurate use of CDISC ADaM professional terminology
- **Source Document References**: Provides document source information for answers
- **Confidence Scoring**: Provides 0-1 confidence scores for each answer

## 🔍 Troubleshooting

### Common Issues

1. **System Initialization Failed**
   - Check if API key is correctly configured
   - Confirm network connection is normal
   - Verify PDF document exists

2. **Vector Database Loading Failed**
   - Delete the `adam_vectorstore` folder and restart the system
   - Check if disk space is sufficient

3. **Poor Answer Quality**
   - Try rephrasing the question
   - Use more specific ADaM terminology
   - Check if the question is within ADaM scope

## 📈 Performance Optimization

- Vector database uses FAISS for efficient similarity search
- Conversation history limited to recent 10 rounds to control memory usage
- Document chunk size optimized to 1000 characters with 200 character overlap
- Retrieval results limited to 5 most relevant documents

**Note**: This system is built based on ADaMIG v1.3 documentation. Please ensure your usage complies with relevant regulatory requirements.