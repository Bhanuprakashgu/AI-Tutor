# AI Tutor

## Project Description

The AI Tutor is an interactive educational assistant designed to provide personalized learning experiences. It leverages advanced AI models to understand user queries, provide comprehensive explanations, and facilitate learning through various interactive features. This project aims to make learning more accessible and engaging by offering an intelligent tutoring system.

## Features

*   **Intelligent Q&A:** Answers user questions on a wide range of topics using a robust knowledge base.
*   **Document Analysis:** Processes uploaded documents (PDF, DOCX, TXT) to extract information and answer questions based on their content.
*   **Interactive Learning:** Supports dynamic conversations and provides explanations in an easy-to-understand format.
*   **Virtual Environment Setup:** Easy to set up and run using a Python 3.11 virtual environment.

## Installation

To set up and run the AI Tutor project, follow these steps:

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/Bhanuprakashgu/AI-Tutor.git
cd ai_tutor
```

### 2. Create a Python 3.11 Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects. Ensure you have Python 3.11 installed on your system. If not, you can download it from the official Python website or use a version manager like `pyenv` or `conda`.

Once Python 3.11 is available, create and activate the virtual environment:

**On macOS/Linux:**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python3.11 -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Database Initialization

The project uses an SQLite database to track documents, study sessions, and quizzes. Initialize the database by running:

```bash
python run.py
# This will create the necessary database file (ai_tutor.db) and tables.
```

## Usage

To start the AI Tutor application, ensure your virtual environment is activated and run the `run.py` file:

```bash
python run.py
```

Once the server is running, open your web browser and navigate to `http://127.0.0.1:5000` (or the address displayed in your console) to access the AI Tutor interface.

## Project Structure

```
ai_tutor_enhanced_mascot_and_mute/
├── app.py                 # Main Flask application file
├── requirements.txt       # Python dependencies
├── run.py                 # Entry point for running the application
├── templates/             # HTML templates for the web interface
│   └── index.html         # Main HTML page
├── ai_tutor.db            # SQLite database file (generated after first run)
└── chroma_db/             # Directory for ChromaDB vector store (generated after first run)
```

## Dependencies

The core dependencies are listed in `requirements.txt` and include:

*   `Flask`: Web framework for building the application.
*   `Flask-CORS`: Handles Cross-Origin Resource Sharing.
*   `requests`: Used for making HTTP requests (e.g., to external APIs).
*   `langchain`, `langchain-community`, `langchain-huggingface`: For building AI applications with LLMs.
*   `chromadb`: Vector database for storing and retrieving document embeddings.
*   `sentence-transformers`: For generating sentence embeddings.
*   `SpeechRecognition`, `pyttsx3`, `pyaudio`: For speech-to-text and text-to-speech functionalities.
*   `PyPDF2`, `python-docx`: For parsing PDF and DOCX documents.
*   `python-dotenv`: For managing environment variables.
*   `numpy`, `pandas`: For numerical operations and data manipulation.
*   `Werkzeug`: WSGI utility library for Python.

## Contributing

We welcome contributions to the AI Tutor project! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

---

**Enjoy your enhanced learning experience with the AI Tutor! 🎓✨**

