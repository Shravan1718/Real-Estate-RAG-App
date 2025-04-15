# 🏠 Real Estate RAG App

A Gen AI-powered tool that helps Real Estate Research Analysts automate the process of extracting insights from real estate news articles using Retrieval-Augmented Generation (RAG).

---

## Problem Statement

Real estate analysts often need to stay updated with the latest market trends by reading articles from trusted news websites such as *The Times of India*, *The Hindu*, and *Deccan Herald*. However, the current process is manual and inefficient:

- Analysts have to visit each website individually
- Read through lengthy articles
- Manually extract relevant insights
- Create summaries and reports for investment decisions

This repetitive and time-consuming task reduces productivity and delays actionable insights.

---

## 💡 Solution: Real Estate Research Tool using RAG

To overcome these inefficiencies, this project leverages Retrieval-Augmented Generation (RAG) to automate the article analysis workflow.

---

## Key Features

- Accepts multiple article URLs from supported sources
- Extracts and processes article content from the web
- Splits and stores articles in a vector database
- Enables question-answering via natural language prompts
- Returns answers sourced directly from the ingested articles

---

## Tech Stack

- **Python**
- **Streamlit** – For building the web app interface
- **LangChain** – Core framework for chaining LLM tasks
- **ChatGroq (Groq API)** – High-performance LLM for fast inference
- **HuggingFace Embeddings** – For generating document embeddings
- **Chroma (langchain_chroma)** – Lightweight vector database for semantic search
- **UnstructuredURLLoader** – To scrape and extract article content from URLs
- **RecursiveCharacterTextSplitter** – For chunking text efficiently

---

## About the Author

I’m a Real Estate Research Analyst at a company that manages commercial and residential properties. Investors can participate in our portfolio similar to how they invest in stocks. My role involves:

- Tracking real estate trends and market news
- Analyzing financial and market data
- Creating reports for investment decisions

---

## Why Not Just Use ChatGPT?

- ChatGPT lacks memory of past articles  
- Context window limitations prevent full article ingestion
- Manual copy-pasting of article content is tedious  

This tool solves all those issues by letting you *ask questions directly about multiple articles*, without switching tabs or manually summarizing content.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

Feel free to fork the project, submit improvements, or use it as a base for your own vertical-specific RAG tools.  
Happy researching!
