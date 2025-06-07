# Fix for sqlite3 version on Streamlit Cloud
# This needs to be at the very top of the file, before any other imports that might
# implicitly load sqlite3 (e.g., chromadb which is used by langchain_community.vectorstores).
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']

import streamlit as st
import os
import shutil # Used for clearing the persist directory if needed

# LangChain specific imports for RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader # For loading different document types
# IMPORTANT: For RetrievalQA with chain_type="stuff", use PromptTemplate directly, not ChatPromptTemplate.
# Updated import: Import PromptTemplate from langchain_core.prompts for consistency
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, PromptTemplate


# --- Configuration ---
# Get Google API key from Streamlit secrets.
# This is crucial for authentication with Google Generative AI services.
try:
    google_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. "
             "Please add it to your Streamlit Cloud app secrets via "
             "Settings > Secrets, or check your .streamlit/secrets.toml locally.")
    st.stop() # Stop the app execution if the API key is not available.

# Set the GOOGLE_API_KEY environment variable for LangChain and Google Generative AI clients.
os.environ["GOOGLE_API_KEY"] = google_api_key

# --- Global/Cached Variables ---
# Use st.session_state to store expensive, long-lived objects like the RAG chain
# and the vector store. This prevents them from being re-initialized on every Streamlit rerun,
# which greatly improves performance and avoids timeouts.
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

# Define the directory where ChromaDB will persist its data (embeddings, metadata).
# This directory will be created if it doesn't exist.
PERSIST_DIRECTORY = "./chroma_db"
# Define the directory containing your HR policy documents.
# Ensure this directory exists in your GitHub repository alongside your app file.
HR_POLICIES_DIR = "hr_policies"

# --- Functions for RAG Setup ---

@st.cache_resource(show_spinner=False)
def get_gemini_llm():
    """
    Initializes and returns the Google Gemini LLM (Generative Language Model).
    Uses caching to ensure the LLM is only initialized once per app session.
    """
    return ChatGoogleGenerativeAI(model="gemini-pro")

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    Initializes and returns the Google Generative AI Embeddings model.
    Uses caching to ensure the embedding model is only initialized once per app session.
    """
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_documents(directory):
    """
    Loads all supported documents from a specified directory.
    Currently supports .txt and .pdf files.
    """
    documents = []
    # Iterate through all files in the given directory.
    if not os.path.exists(directory):
        st.error(f"Document directory '{directory}' not found. Please create it and add documents.")
        return []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename) # Construct the full file path.
        if os.path.isfile(filepath): # Ensure it's a file, not a subdirectory.
            try:
                # Load text files.
                if filename.lower().endswith(".txt"):
                    loader = TextLoader(filepath)
                    documents.extend(loader.load())
                # Load PDF files.
                elif filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load())
                else:
                    # Warn about unsupported file types.
                    st.warning(f"Skipping unsupported file type: {filename}. Only .txt and .pdf are supported.")
            except Exception as e:
                # Catch and display errors during document loading.
                st.error(f"Error loading document {filename}: {e}")
    return documents

def setup_rag():
    """
    Sets up the RAG system, implementing persistence for the Chroma vector store.
    It first attempts to load an existing vector store from disk.
    If no existing store is found or if loading fails, it creates a new one
    by loading and processing documents, then persists it to disk.
    """
    llm = get_gemini_llm() # Get the cached LLM instance.
    embeddings = get_embedding_model() # Get the cached embedding model instance.

    # Check if the vector store persistence directory exists and contains data.
    # This check prevents re-embedding documents on every app rerun after the first successful build.
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.info("Loading existing vector store from disk... This is much faster!")
        try:
            # Attempt to load the vector store from the persisted directory.
            vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            st.session_state['vectorstore'] = vectorstore # Store in session state for future use.
            st.success("Vector store loaded successfully!")
        except Exception as e:
            # If loading fails (e.g., corrupted data, version mismatch), log error and try to re-embed.
            st.error(f"Error loading existing vector store: {e}. Attempting to rebuild and re-embed.")
            # Clean up the old directory before rebuilding.
            if os.path.exists(PERSIST_DIRECTORY):
                st.info(f"Removing corrupted persistence directory: {PERSIST_DIRECTORY}")
                shutil.rmtree(PERSIST_DIRECTORY)
            # Proceed to create and persist a new vector store.
            vectorstore = create_and_persist_vectorstore(llm, embeddings)
    else:
        # If no existing vector store is found, create a new one from scratch.
        st.info("No existing vector store found. Creating and persisting new vector store...")
        vectorstore = create_and_persist_vectorstore(llm, embeddings)

    # Define a custom prompt template for the RetrievalQA chain.
    # This guides the LLM on how to behave and use the provided context.
    # For RetrievalQA with chain_type="stuff", a simple PromptTemplate is expected.
    prompt_template_string = """
    You are an AI-powered HR Assistant for Google. Your task is to provide answers based ONLY on the provided HR policy context.
    If the user asks a question, answer it concisely and directly from the context.
    If the context does not contain enough information to answer the question, politely state that you cannot answer from the provided documents.
    Do not make up information. Maintain a professional and helpful tone.

    Context:
    {context}

    Question: {question}
    """
    # Using PromptTemplate as required for RetrievalQA with "stuff" chain type.
    custom_prompt = PromptTemplate(
        template=prompt_template_string,
        input_variables=["context", "question"]
    )

    # Create the RAG chain using LangChain's RetrievalQA.
    # The retriever component fetches relevant document chunks.
    # 'chain_type="stuff"' means all retrieved documents are combined into one prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant chunks.
        chain_type="stuff",
        prompt=custom_prompt, # Apply the custom prompt.
        return_source_documents=True # This allows displaying source documents if desired.
    )
    return qa_chain

def create_and_persist_vectorstore(llm, embeddings):
    """
    Loads documents, splits them into manageable chunks, creates embeddings for each chunk,
    builds a Chroma vector store, and then persists this store to disk.
    This function is called only when the vector store needs to be built from scratch.
    """
    with st.spinner("Loading and processing documents... This may take a moment on first run."):
        # Load documents from the HR policies directory.
        documents = load_documents(HR_POLICIES_DIR)

        if not documents:
            st.error(f"No documents found in the '{HR_POLICIES_DIR}' directory. "
                     "Please add some HR policy files (e.g., .txt, .pdf) to this folder "
                     "and push them to GitHub.")
            st.stop() # Stop the app if no documents are available to prevent further errors.

        # Initialize text splitter to break large documents into smaller, manageable chunks.
        # This is important because LLMs have context window limits.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddings and build the Chroma vector store.
        # This is the most resource-intensive step and is why persistence is crucial.
        st.info(f"Creating embeddings for {len(splits)} document chunks...")
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY # Specify the directory for persistence.
            )
            vectorstore.persist() # Explicitly save the collection to disk.
            st.session_state['vectorstore'] = vectorstore # Store in session state.
            st.success("Vector store created and persisted successfully!")
            return vectorstore
        except Exception as e:
            # Handle errors during embedding creation (e.g., API timeouts).
            st.error(f"Error embedding content: {e}. "
                     "This might be due to too many/large documents, an invalid API key, or network issues. "
                     "Please check your `hr_policies` folder content and Streamlit secrets.")
            st.stop() # Stop the app if embedding fails.

# --- Streamlit UI ---

# Configure the Streamlit page's title and icon.
st.set_page_config(page_title="AI-Powered HR Policy Assistant", page_icon="🤖")

# Apply custom CSS for better aesthetics and a consistent look.
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6; /* Light gray background */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px; /* Limit content width for better readability */
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        border: 1px solid #ccc;
        padding: 0.75rem;
    }
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #4CAF50; /* Google Green-like */
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .chat-bubble {
        background-color: #e0f2f7; /* Light blue for assistant */
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
        align-self: flex-start; /* Align assistant bubbles to the left */
    }
    .chat-bubble.user {
        background-color: #d1e7dd; /* Light green for user */
        align-self: flex-end; /* Align user bubbles to the right */
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 AI-Powered HR Policy Assistant")
st.write("Ask questions about your HR policies, and I'll provide answers based on the documents you've uploaded.")
st.markdown("---") # Add a separator

# Initialize chat history in session state if it doesn't exist.
# This keeps the conversation persistent across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun.
# Each message is displayed in a chat bubble style.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RAG Setup (cached to run only once or on config change) ---
# This block attempts to set up the RAG chain.
# If an error occurs during setup, it's caught and displayed, and the app stops.
try:
    if st.session_state['qa_chain'] is None:
        st.session_state['qa_chain'] = setup_rag()
except Exception as e:
    st.error(f"Error during RAG system setup: {e}")
    st.stop() # Stop the app if RAG setup fails critically.

# Chat input for the user to ask questions.
if prompt := st.chat_input("Ask a question about HR policies..."):
    # Add user message to chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in the chat interface.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the RAG chain.
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder for streaming or dynamic content.
        full_response = "" # Accumulate the full response.
        try:
            # Invoke the cached RAG chain with the user's query.
            response = st.session_state['qa_chain'].invoke({"query": prompt})
            # Extract the main answer from the RAG chain's response.
            answer = response.get("result", "I couldn't find an answer based on the provided documents.")
            full_response = answer

            # Optionally, display source documents for transparency.
            source_docs = response.get("source_documents", [])
            if source_docs:
                full_response += "\n\n**Sources:**\n"
                # Iterate through source documents and append a preview of their content.
                for i, doc in enumerate(source_docs):
                    # Limit content preview to avoid cluttering the UI.
                    content_preview = doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                    # Use os.path.basename to get just the filename.
                    full_response += f"- Document: `{os.path.basename(doc.metadata.get('source', 'Unknown'))}`\n  Content: `{content_preview}`\n"

        except Exception as e:
            # Catch any other general exceptions during RAG system interaction.
            full_response = (f"An error occurred while getting a response from the RAG system: {e}. "
                             "Please try again later. Check your API key and document content.")
            st.error(full_response)

        # Update the message placeholder with the full accumulated response.
        message_placeholder.markdown(full_response)

    # Add the assistant's response to the chat history.
    st.session_state.messages.append({"role": "assistant", "content": full_response})
