import streamlit as st
import google.generativeai as genai
import os

# LangChain specific imports for RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="RAG HR Policy Q&A Bot", page_icon="📚")
st.title("📚 RAG-Powered HR Policy Q&A Assistant")
st.write("Ask questions about specific HR policies based on provided documents.")

# --- API Key Configuration (Securely from Streamlit Secrets) ---
try:
    gemini_api_key = st.secrets["gemini_api_key"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("Error: Gemini API key not found in Streamlit secrets. "
             "Please ensure 'gemini_api_key' is set in your app's secrets.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during API key configuration: {e}")
    st.stop()

# --- RAG Setup: Load, Split, Embed, Store ---

# Persistent storage for ChromaDB (so it doesn't rebuild every time)
# Note: For public web hosting, this might be more complex. For Streamlit Cloud,
# it might rebuild on every refresh. For a persistent solution, consider a
# dedicated vector DB or a more complex caching strategy.
# For now, let's assume it rebuilds or stores to /tmp/chroma_db for the session.
CHROMA_DB_PATH = "chroma_db" # Or "/tmp/chroma_db" if issues with permissions

@st.cache_resource # Cache this to avoid rebuilding the vector store on every rerun
def setup_rag():
    # 1. Load Documents
    documents = []
    policy_dir = "hr_policies"
    if not os.path.exists(policy_dir):
        st.error(f"Error: The '{policy_dir}' directory was not found. Please create it and add your HR policy files.")
        st.stop()

    for file_name in os.listdir(policy_dir):
        file_path = os.path.join(policy_dir, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        # Add more loaders for other file types if needed (e.g., CSVLoader, Docx2txtLoader)

    if not documents:
        st.warning(f"No documents found in the '{policy_dir}' directory. Please add some policy files (e.g., .txt, .pdf).")
        st.stop()

    # 2. Split Documents
    # Use RecursiveCharacterTextSplitter for more intelligent splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. Create Embeddings
    # Using GoogleGenerativeAIEmbeddings for consistency with Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # This is a common embedding model

    # 4. Build a Vector Store (ChromaDB)
    # This will create/load the vector store from the specified path
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    # Persist the database to disk (important for subsequent runs)
    vectordb.persist()
    return vectordb

# --- Initialize RAG components (this runs only once thanks to @st.cache_resource) ---
try:
    vectordb = setup_rag()
    # 5. Setup the Retriever and LLM for RAG Chain
    # Configure the LLM that will answer questions
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2) # Lower temperature for factual answers

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' means put all retrieved docs into the prompt
        retriever=vectordb.as_retriever()
    )
except Exception as e:
    st.error(f"Error setting up RAG system: {e}")
    st.info("Please ensure your 'hr_policies' directory exists with valid documents and all libraries are installed.")
    st.stop()


# --- Streamlit Interaction ---
user_question = st.text_input("Ask a question about HR policies:", placeholder="e.g., How many sick days do we get?")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Searching and generating answer..."):
            try:
                # Get the answer from the RAG chain
                response = qa_chain.invoke(user_question)
                st.write(response["result"]) # The answer is in the "result" key

            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
                st.info("Please check the 'View app logs' for more details.")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("Developed by [Your Name] for AI Strategist Portfolio (RAG Enabled)")
