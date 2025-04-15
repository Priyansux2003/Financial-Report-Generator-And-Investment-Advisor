import pandas as pd
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set device for model (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Zephyr 7B model
@st.cache_resource
def load_zephyr_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map=device
    )
    return model, tokenizer

zephyr_model, zephyr_tokenizer = load_zephyr_model()

# Function to generate responses using Zephyr 7B with retrieved context
def generate_response(prompt, retrieved_docs):
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}\n\nProvide a detailed investment suggestion."

    inputs = zephyr_tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = zephyr_model.generate(
            **inputs, max_length=512, temperature=0.7, pad_token_id=zephyr_tokenizer.eos_token_id
        )

    return zephyr_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    try:
        data = pd.read_csv("Finance_data.csv")

        # Rename and clean columns
        data = data.rename(columns={
            'age': 'Age',
            'gender': 'Gender',
            'Investment_Avenues': 'Investment_Avenue',
            'Factor': 'Factors_Considered',
            'Objective': 'Savings_Objectives',
            'What are your savings objectives?': 'Savings_Objectives',
            'Invest_Monitor': 'Investment_Monitoring'
        })

        # Drop unnecessary columns
        data = data.drop(columns=[col for col in ['Purpose', 'Stock_Marktet'] if col in data.columns])

        # Clean Expect column
        if 'Expect' in data.columns:
            data['Expect'] = data['Expect'].astype(str).str.replace('%', '', regex=True)

        return data.to_dict(orient='records')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Convert data to prompt-response format
def prepare_prompt_response(data_fin):
    prompt_response_data = []
    for entry in data_fin:
        try:
            expected_returns = entry.get('Expect', 'N/A')

            prompt = (
                f"I'm a {entry.get('Age', 'N/A')}-year-old {entry.get('Gender', 'N/A')} looking to invest in "
                f"{entry.get('Investment_Avenue', 'N/A')} for {entry.get('Savings_Objectives', 'N/A')} "
                f"over the next {entry.get('Duration', 'N/A')}. What are my options?"
            )

            response = (
                f"Based on your preferences, here are your investment options:\n"
                f"- Mutual Funds: {entry.get('Mutual_Funds', 'N/A')}\n"
                f"- Equity Market: {entry.get('Equity_Market', 'N/A')}\n"
                f"- Debentures: {entry.get('Debentures', 'N/A')}\n"
                f"- Government Bonds: {entry.get('Government_Bonds', 'N/A')}\n"
                f"- Fixed Deposits: {entry.get('Fixed_Deposits', 'N/A')}\n"
                f"- PPF: {entry.get('PPF', 'N/A')}\n"
                f"- Gold: {entry.get('Gold', 'N/A')}\n"
                f"Factors considered: {entry.get('Factors_Considered', 'N/A')}\n"
                f"Objective: {entry.get('Savings_Objectives', 'N/A')}\n"
                f"Expected returns: {expected_returns}%\n"
                f"Investment monitoring: {entry.get('Investment_Monitoring', 'N/A')}\n"
                f"Reasons for choices:\n"
                f"- Equity: {entry.get('Reason_Equity', 'N/A')}\n"
                f"- Mutual Funds: {entry.get('Reason_Mutual', 'N/A')}\n"
                f"- Bonds: {entry.get('Reason_Bonds', 'N/A')}\n"
                f"- Fixed Deposits: {entry.get('Reason_FD', 'N/A')}\n"
                f"Source of information: {entry.get('Source', 'N/A')}\n"
            )
            prompt_response_data.append({"prompt": prompt, "response": response})
        except KeyError as e:
            st.error(f"Missing key in data: {e}")

    return prompt_response_data

# Initialize HuggingFace Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process Data into VectorDB
@st.cache_resource
def create_vectordb(data_fin):
    prompt_response_data = prepare_prompt_response(data_fin)
    documents = [Document(page_content=f"Prompt: {entry['prompt']}\nResponse: {entry['response']}") for entry in prompt_response_data]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    persist_directory = 'docs/chroma/'
    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(documents=texts, embedding=get_embeddings(), persist_directory=persist_directory)

    return vectordb

# Streamlit UI
st.title("💰 AI Financial Advisor with Zephyr 7B")

st.sidebar.header("Upload Financial Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data_fin = load_data(uploaded_file)

    if data_fin:
        vectordb_fin = create_vectordb(data_fin)
        retriever_fin = vectordb_fin.as_retriever(search_kwargs={"k": 5})

        st.subheader("Investment Query")
        user_query = st.text_area("Enter your financial query:")

        if st.button("Get Investment Advice"):
            if user_query:
                with st.spinner("Fetching advice..."):
                    retrieved_docs = retriever_fin.get_relevant_documents(user_query)
                    if retrieved_docs:
                        result = generate_response(user_query, retrieved_docs)
                        st.subheader("Investment Advice:")
                        st.write(result)
                    else:
                        st.error("No relevant documents found. Try rephrasing your query.")
            else:
                st.warning("Please enter a query to get investment advice.")
    else:
        st.error("Failed to load data. Please check your CSV file format.")
else:
    st.info("Upload a CSV file to start.")
