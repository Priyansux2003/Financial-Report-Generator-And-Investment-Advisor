import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

st.set_page_config(page_title="News QA Bot", layout="wide")
st.title("🧠 Ask Questions from a Financial News Article")

# Sidebar input for URL and question
url = st.text_input("🔗 Paste the article URL below", "")
query = st.text_input("❓ Ask a question about the article", "")

run_button = st.button("🚀 Run")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = load_llm()
embedding_model = get_embeddings()

def load_and_index_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()
    if not docs:
        return None, None, "❌ Failed to load content from the URL."
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return chunks, vectorstore, None

if run_button and url and query:
    with st.spinner("🔄 Processing the article..."):
        chunks, vectorstore, error = load_and_index_url(url)
        if error:
            st.error(error)
        else:
            retriever = vectorstore.as_retriever()
            docs_retrieved = retriever.get_relevant_documents(query)

            st.success(f"✅ Retrieved {len(docs_retrieved)} relevant chunks.")
            with st.expander("📄 Preview First Retrieved Chunk"):
                if docs_retrieved:
                    st.write(docs_retrieved[0].page_content[:1000])

            try:
                qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
                result = qa_chain({"question": query}, return_only_outputs=True)

                st.subheader("📌 Answer")
                st.write(result.get("answer", "[No answer found]"))

                st.subheader("📎 Sources")
                st.write(result.get("sources", "[No sources found]"))
            except Exception as e:
                st.error(f"❌ Error: {e}")
elif run_button:
    st.warning("⚠ Please enter both a URL and a question.")
