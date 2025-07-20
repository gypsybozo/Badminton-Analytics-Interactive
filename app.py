# app.py
import streamlit as st
import os
import pandas as pd
import time
import warnings
from dotenv import load_dotenv
import traceback

# --- Core LangChain Components ---
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- LLM Integration (Google Generative AI / AI Studio) ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Embedding Model Integration ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- Vector Store Integration ---
VECTOR_STORE_TYPE = "FAISS" # Or "CHROMA" - Make sure this matches rag_query.py if reusing index

if VECTOR_STORE_TYPE == "FAISS":
    from langchain_community.vectorstores import FAISS
elif VECTOR_STORE_TYPE == "CHROMA":
    from langchain_community.vectorstores import Chroma
else:
    st.error("Invalid VECTOR_STORE_TYPE configured.")
    st.stop() # Stop execution if invalid config

# --- Document Loaders ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# --- Suppress Warnings (Optional) ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='google') # Suppress google warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module='langchain_google_genai')

# ===============================================================
#                       CONFIGURATION
# ===============================================================
load_dotenv() # Load environment variables from .env file

# --- API Keys ---
# Use st.secrets for deployment, os.getenv for local with .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     try: # Try getting from streamlit secrets if deployed
#         GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
#     except:
#          st.error("ERROR: GOOGLE_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
#          st.stop()
if not GOOGLE_API_KEY: # Check again after trying secrets
     st.error("ERROR: GOOGLE_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
     st.stop()


# --- File Paths ---
# !!! UPDATE THIS PATH to your specific CSV file !!!
CSV_DATA_PATH = "/Users/kriti.bharadwaj03/Badminton_Analysis/knowledge_base/badminton_shot_dataset_20250505_020417.csv" # <--- IMPORTANT: SET YOUR CSV FILE NAME
KNOWLEDGE_BASE_DIR = "knowledge_base" # <--- Ensure this directory exists

# --- Model Configuration ---
LLM_MODEL = "gemini-2.5-flash-preview-05-20" # Use Flash for potentially faster/cheaper responses in GUI
# Or: LLM_MODEL = "gemini-pro"
# Or: LLM_MODEL = "gemini-1.5-pro-latest" # If available and needed
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

# --- Vector Store & Chunking ---
if VECTOR_STORE_TYPE == "FAISS":
    VECTORSTORE_PATH = "vectorstore_faiss_genai"
elif VECTOR_STORE_TYPE == "CHROMA":
    VECTORSTORE_PATH = "vectorstore_chroma_genai"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 70
RETRIEVER_K = 6 # How many chunks to retrieve

# ===============================================================
#                  HELPER & CACHED FUNCTIONS
# ===============================================================

# --- Helper Function (Copied from rag_query.py) ---
def format_rally_for_embedding(rally_df):
    """ Converts rows for a single rally into a descriptive text chunk. """
    if rally_df.empty:
        return None
    try:
        # Ensure required columns exist, provide defaults if missing
        rally_id = rally_df['rally_id'].iloc[0] if 'rally_id' in rally_df.columns else 'UnknownRally'
        # Changed "Analysis for" to "Details of" for potentially better LLM parsing
        text = f"Details of Rally {rally_id}:\n"

        for _, row in rally_df.iterrows():
            player = row.get('player_who_hit', 'UnknownPlayer')
            shot_num = row.get('shot_num', 'UnknownShotNum')

            # --- THIS IS THE KEY FIX ---
            # Use the 'stroke_hand' column to get the stroke information.
            # Provide a clear default if the column is missing or the value is null.
            stroke_played = row.get('stroke_hand', 'an Undetermined Stroke')
            if pd.isna(stroke_played) or not stroke_played: # Check for NaN or empty string
                stroke_played = 'an Undetermined Stroke'
            # --- END OF KEY FIX ---

            # Optionally include other reliable information if it helps context
            confirmation_method = row.get('confirmation_method', '')
            # hitting_posture = row.get('hitting_posture', '') # If you add this later

            player_pos_key = f'player{player}_coords' if player in [1, '1', 2, '2'] else None
            player_pos = row.get(player_pos_key) if player_pos_key else None

            text += f"  - Shot {shot_num}: Player {player} played {stroke_played}." # Use the corrected variable

            if player_pos:
                text += f" Player position was approximately {player_pos}."
            # if confirmation_method: # You might want to include this for debugging or advanced queries
            #     text += f" (Confirmed via {confirmation_method})."
            text += "\n"

        # Add rally outcome later if available
        # rally_outcome = row.get('rally_outcome', None)
        # if rally_outcome:
        #     text += f"Outcome of Rally {rally_id}: {rally_outcome}\n"
        return text
    except Exception as e:
        current_rally_id_for_error = 'UnknownRally'
        if not rally_df.empty and 'rally_id' in rally_df.columns:
            try:
                current_rally_id_for_error = rally_df['rally_id'].iloc[0]
            except:
                pass
        print(f"Error formatting rally {current_rally_id_for_error}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return None

# --- Cached Functions for Loading Models and Data ---

@st.cache_resource # Cache embedding model loading
def load_embeddings(model_name):
    print(f"[Cache] Loading embedding model: {model_name}", flush=True)
    try:
        # Force CPU usage for embeddings if GPU causes issues in deployment/local
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model '{model_name}': {e}")
        st.stop()

@st.cache_resource # Cache LLM loading
def load_llm(model_name, api_key):
    print(f"[Cache] Initializing LLM: {model_name}", flush=True)
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        # Optional test invoke
        # llm.invoke("test")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM '{model_name}': {e}")
        st.stop()

@st.cache_resource # Cache vector store loading/building
def load_or_build_vector_store(_embeddings): # Pass embeddings as arg for caching dependency
    vectorstore = None
    if os.path.exists(VECTORSTORE_PATH):
        print(f"\n[Cache] Loading existing vector store from {VECTORSTORE_PATH}...", flush=True)
        load_start = time.time()
        try:
            if VECTOR_STORE_TYPE == "FAISS":
                vectorstore = FAISS.load_local(VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)
            elif VECTOR_STORE_TYPE == "CHROMA":
                vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=_embeddings)
            load_end = time.time()
            print(f"[Cache] Vector store loaded successfully. Duration: {load_end - load_start:.2f}s", flush=True)
            return vectorstore # Return loaded store
        except Exception as e:
            print(f"[Cache] Error loading vector store: {e}. Will rebuild.", flush=True)
            vectorstore = None # Force rebuild

    # --- Build if not loaded ---
    print("\n[Cache] Building new vector store...", flush=True)
    build_start = time.time()
    all_docs = []
    # a) Load Knowledge Base
    print("  [Cache] Loading Knowledge Base...", flush=True)
    if os.path.exists(KNOWLEDGE_BASE_DIR) and os.path.isdir(KNOWLEDGE_BASE_DIR):
        try:
            kb_loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader)
            kb_docs = kb_loader.load()
            for doc in kb_docs: doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "kb_unknown"))
            print(f"    [Cache] Loaded {len(kb_docs)} KB documents.", flush=True)
            all_docs.extend(kb_docs)
        except Exception as e: print(f"    [Cache] Error loading knowledge base: {e}", flush=True)
    else: print(f"    [Cache] Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Skipping.", flush=True)

    # b) Load Tactical Data
    print(f"  [Cache] Loading Tactical Data from {CSV_DATA_PATH}...", flush=True)
    if os.path.exists(CSV_DATA_PATH):
        try:
            df = pd.read_csv(CSV_DATA_PATH)
            if not df.empty and 'rally_id' in df.columns:
                 df['rally_id'] = df['rally_id'].ffill().fillna(-1).astype(int)
                 grouped = df[df['rally_id'] != -1].groupby('rally_id')
                 print(f"    [Cache] Processing {len(grouped)} rallies...", flush=True)
                 for name, group in grouped:
                     rally_text = format_rally_for_embedding(group)
                     if rally_text: all_docs.append(Document(page_content=rally_text, metadata={"source": f"Rally_{name}"}))
                 print(f"    [Cache] Formatted {len(grouped)} rallies into Documents.", flush=True)
            else: print("    [Cache] CSV empty or missing 'rally_id'.", flush=True)
        except Exception as e: print(f"    [Cache] Error loading/processing CSV: {e}", flush=True)
    else: print(f"    [Cache] CSV file '{CSV_DATA_PATH}' not found.", flush=True)

    # c) Split
    if not all_docs: print("ERROR: No documents loaded. Cannot build vector store."); st.stop()
    print(f"  [Cache] Splitting {len(all_docs)} documents...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"    [Cache] Split into {len(split_docs)} chunks.", flush=True)

    # d) Create Index
    print(f"  [Cache] Creating {VECTOR_STORE_TYPE} index...", flush=True)
    index_start_time = time.time()
    try:
        if VECTOR_STORE_TYPE == "FAISS":
            vectorstore = FAISS.from_documents(split_docs, _embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
        elif VECTOR_STORE_TYPE == "CHROMA":
             vectorstore = Chroma.from_documents(documents=split_docs, embedding=_embeddings, persist_directory=VECTORSTORE_PATH)
        index_end_time = time.time()
        print(f"  [Cache] Index created/saved to {VECTORSTORE_PATH}. Duration: {index_end_time - index_start_time:.2f}s", flush=True)
    except Exception as e: print(f"ERROR creating/saving index: {e}\n{traceback.format_exc()}"); st.stop()
    build_end = time.time()
    print(f"[Cache] Vector store build complete. Duration: {build_end - build_start:.2f}s", flush=True)
    return vectorstore


@st.cache_resource # Cache QA chain setup
def setup_qa_chain(_llm, _vectorstore):
    print("[Cache] Setting up RetrievalQA chain...", flush=True)
    if _vectorstore is None:
        st.error("Vector store is not available. Cannot setup QA chain.")
        st.stop()
    try:
        retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        template = """You are an expert badminton tactical analyst reviewing match data. Use ONLY the following pieces of retrieved context (which includes rules, tactical principles, and specific rally data) to answer the question concisely and factually. Do not add information not present in the context. If the context does not contain the answer, clearly state that the provided data does not have the information. Reference specific rally or shot numbers from the context if they support your answer.

Context:
---------
{context}
---------

Question: {question}

Concise Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        print("[Cache] RetrievalQA chain ready.", flush=True)
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA chain: {e}")
        st.stop()

# ===============================================================
#                       STREAMLIT APP UI
# ===============================================================

st.set_page_config(page_title="Badminton RAG Analysis", layout="wide")
st.title("🏸 Badminton Tactical Analysis Chatbot")
st.caption(f"Powered by LangChain and Google Gemini ({LLM_MODEL})")

# --- Load resources using cached functions ---
try:
    embeddings = load_embeddings(EMBEDDING_MODEL)
    llm = load_llm(LLM_MODEL, GOOGLE_API_KEY)
    vector_store = load_or_build_vector_store(embeddings) # Pass embeddings
    qa_chain = setup_qa_chain(llm, vector_store) # Pass llm and vector_store
except Exception as e:
     st.error(f"A critical error occurred during initialization: {e}")
     st.stop()


# --- User Input Form ---
with st.form("query_form"):
    user_query = st.text_input("Ask a question about the match tactics:", placeholder="e.g., Why did player 1 lose points on their backhand?")
    submitted = st.form_submit_button("Ask")

# --- Process Query and Display Results ---
if submitted and user_query:
    st.markdown("---") # Separator
    start_query_time = time.time()
    with st.spinner("Thinking... Retrieving data and generating answer..."):
        try:
            # Invoke the QA chain
            result = qa_chain.invoke({"query": user_query})
            answer = result.get("result", "Error: No answer found.")
            source_docs = result.get("source_documents", [])

            end_query_time = time.time()
            query_duration = end_query_time - start_query_time

            # Display Answer
            st.subheader("Answer:")
            st.markdown(answer) # Use markdown for better formatting
            st.caption(f"Query processed in {query_duration:.2f} seconds.")

            # Optional: Display Sources
            if source_docs:
                with st.expander("View Retrieved Context Chunks"):
                    st.markdown("**Context used to generate the answer:**")
                    for i, doc in enumerate(source_docs):
                        source = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Chunk {i+1} [Source: {source}]**")
                        st.text(doc.page_content) # Display full chunk content
                        st.markdown("---")


        except Exception as e:
            st.error(f"An error occurred processing your query: {e}")
            st.exception(e) # Show full traceback in the app for debugging

elif submitted and not user_query:
    st.warning("Please enter a question.")