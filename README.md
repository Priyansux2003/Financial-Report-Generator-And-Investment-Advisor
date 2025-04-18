# Financial-Report-Generator-And-Investment-Advisor

Welcome to the Financial-Report-Generator-And-Investment-Advisor—a comprehensive, user-friendly platform that leverages the power of Generative AI (GenAI) and Natural Language Processing (NLP) to help you analyze financial reports, extract insights from financial news, and receive personalized investment advice. This project is designed for both beginners and advanced users, making financial data analysis accessible and insightful for everyone.

Project Overview
This repository consists of three main modules:

1. Financial Report Generator:
Fetch and preprocess real-time financial data for stocks and economic indicators.

2. Financial News QA Bot:
Ask questions about any financial news article and get instant, AI-generated answers with sources.

3. Investment Advisor:
Upload your financial data and receive personalized investment suggestions powered by a state-of-the-art language model.

Key Features
No Prior AI Knowledge Needed:
Simple interfaces and clear instructions make it easy for anyone to use.

Advanced GenAI Integration:
Utilizes large language models (LLMs) like Google FLAN-T5 and Zephyr 7B for deep understanding and natural language responses.

Real-Time Data Processing:
Fetches live financial data and news for up-to-date analysis.

Customizable and Extensible:
Built with modular Python code and Streamlit for easy customization and deployment.

How It Works
1. Financial Report Generator
Fetches financial data (e.g., stock prices, earnings announcements) from APIs like Financial Modeling Prep.

Preprocesses data for analysis, converting dates and cleaning columns.

Exports data to CSV for further use.

Technical Highlights:

Uses pandas for data manipulation.

API integration for real-time data.

Data cleaning and preprocessing functions.

2. Financial News QA Bot
Paste any financial news article URL.

Ask questions in plain English about the article.

Get instant answers with references to the source text.

Technical Highlights:

Uses langchain for document loading, text splitting, and vector search.

Embeddings generated with sentence-transformers/all-MiniLM-L6-v2.

LLM-powered QA using Google FLAN-T5.

Built with Streamlit for interactive UI.

3. Investment Advisor
Upload your financial data (CSV format).

Ask investment-related questions (e.g., "What are my best options for retirement savings?").

Receive personalized advice generated by the Zephyr 7B language model.

Technical Highlights:

Data ingestion and cleaning with pandas.

Embedding and vector search with Chroma and langchain.

Contextual response generation using Zephyr 7B (HuggingFace).

Device-aware model loading (CPU/GPU).

Streamlit interface for easy interaction.

Getting Started
Prerequisites
To run this project, ensure you have:

Python 3.9+

Installed the following Python libraries:

Streamlit

HuggingFace Transformers

LangChain

Pandas

FAISS or Chroma

An API key from Financial Modeling Prep.

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/Financial-Report-Generator-And-Investment-Advisor.git
cd Financial-Report-Generator-And-Investment-Advisor
Install dependencies:

bash
pip install -r requirements.txt
Set up your API keys in the code where indicated.

Usage
1. Financial Report Generator
Run the script to fetch, preprocess, and export financial data:

bash
python financial_report_generator.py
2. Financial News QA Bot
Launch the app using Streamlit:

bash
streamlit run news_qa_app.py
Paste a news article URL into the input field, ask your question, and get AI-driven insights.

3. Investment Advisor
Launch the app using Streamlit:

bash
streamlit run investment_advisor_app.py
Upload your CSV file containing financial preferences or objectives, enter your query, and receive personalized investment advice.

Technical Concepts Explained
For those familiar with Generative AI or looking to understand its technical aspects:

GenAI (Generative AI):
AI models that can generate human-like text, answer questions, and provide recommendations based on context.

LLM (Large Language Model):
Advanced AI models trained on vast amounts of text data to understand and generate language (e.g., FLAN-T5, Zephyr 7B).

Embeddings:
Numerical representations of text that allow the AI to understand semantic meaning and perform similarity searches efficiently.

Vector Database:
Stores embeddings for fast retrieval of relevant information based on user queries.

Retrieval-Augmented Generation (RAG):
Combines information retrieval with text generation to provide accurate, context-aware answers by grounding responses in retrieved documents.

Example Use Cases
For Beginners:
"What does this financial report mean for my investments?"

"Summarize this news article for me."

For Advanced Users:
"How does the earnings announcement impact the stock's valuation?"

"Compare investment avenues based on user-uploaded data."

Contributing
Contributions are welcome! Please open issues or submit pull requests to improve functionality or add new features.
