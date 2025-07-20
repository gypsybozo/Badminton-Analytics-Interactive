# rag_query.py
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
# Choose one: FAISS or Chroma
VECTOR_STORE_TYPE = "FAISS" # Or "CHROMA"

if VECTOR_STORE_TYPE == "FAISS":
    from langchain_community.vectorstores import FAISS
elif VECTOR_STORE_TYPE == "CHROMA":
    from langchain_community.vectorstores import Chroma
else:
    raise ValueError("Invalid VECTOR_STORE_TYPE configured.")

# --- Document Loaders ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# --- Suppress Warnings (Optional) ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch') # Suppress specific torch warnings if noisy

# ===============================================================
#                       CONFIGURATION
# ===============================================================
load_dotenv() # Load environment variables from .env file

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables. Get one from Google AI Studio (https://aistudio.google.com/app/apikey).")
    exit()

# --- File Paths ---
# !!! UPDATE THIS PATH to your specific CSV file !!!
CSV_DATA_PATH = "/Users/kriti.bharadwaj03/Badminton_Analysis/knowledge_base/badminton_shot_dataset_20250505_020417.csv" # <--- IMPORTANT: SET YOUR CSV FILE NAME
KNOWLEDGE_BASE_DIR = "knowledge_base" # <--- Ensure this directory exists with .txt files

# --- Model Configuration ---
# Use model names compatible with the Google Generative AI SDK (AI Studio API Key)
# Check https://ai.google.dev/models/gemini for available models via API Key
# LLM_MODEL = "gemini-pro" # Good baseline, generally available
# LLM_MODEL = "gemini-1.5-flash-latest" # Faster, potentially less capable than Pro
LLM_MODEL = "gemini-2.5-pro-exp-03-25" # Use the latest Pro 1.5 model available via API key
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" # Good for Q&A retrieval

# --- Vector Store & Chunking ---
if VECTOR_STORE_TYPE == "FAISS":
    VECTORSTORE_PATH = "vectorstore_faiss_genai" # Directory to save FAISS index
elif VECTOR_STORE_TYPE == "CHROMA":
    VECTORSTORE_PATH = "vectorstore_chroma_genai" # Directory for Chroma persistent data

CHUNK_SIZE = 700    # Experiment with chunk size based on context length needs
CHUNK_OVERLAP = 70  # Overlap helps maintain context between chunks

# --- RAG Configuration ---
RETRIEVER_K = 6 # How many relevant chunks to retrieve for context

# ===============================================================
#                       HELPER FUNCTIONS
# ===============================================================

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

# ===============================================================
#                       MAIN EXECUTION
# ===============================================================

# --- 1. Initialize Models ---
print("Initializing models (Google Generative AI & HuggingFace Embeddings)...", flush=True)
try:
    # Embedding Model (runs locally)
    # Consider 'mps' for device if on Apple Silicon Mac with PyTorch nightlies supporting it
    # Adjust batch_size based on your RAM/CPU
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} # Normalize for better cosine similarity
    )
    print(f"  Embedding model '{EMBEDDING_MODEL}' loaded.", flush=True)

    # LLM (Google Generative AI / AI Studio)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1, # Lower temperature for more factual, less creative answers
        convert_system_message_to_human=True # Helps with compatibility
        # safety_settings=... # Consider configuring safety settings if needed
    )
    print(f"  LLM '{LLM_MODEL}' initialized via Google Generative AI.", flush=True)
    # Optional: Simple test invoke
    # print("    Testing LLM connection...")
    # llm.invoke("Test prompt")
    # print("    LLM connection successful.")

except ImportError as e:
     print(f"ERROR: Missing required libraries. Install them:\npip install langchain-google-genai google-generativeai langchain-huggingface sentence-transformers\nOriginal error: {e}")
     exit()
except Exception as e:
    print(f"Error initializing models: {e}\n{traceback.format_exc()}")
    exit()
print("Models initialized successfully.", flush=True)


# --- 2. Load or Build Vector Store ---
vectorstore = None
vectorstore_exists = os.path.exists(VECTORSTORE_PATH)

if vectorstore_exists:
    print(f"\nLoading existing {VECTOR_STORE_TYPE} vector store from {VECTORSTORE_PATH}...", flush=True)
    load_start = time.time()
    try:
        if VECTOR_STORE_TYPE == "FAISS":
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        elif VECTOR_STORE_TYPE == "CHROMA":
            vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
        load_end = time.time()
        print(f"Vector store loaded successfully. Duration: {load_end - load_start:.2f}s", flush=True)
    except Exception as e:
        print(f"Error loading vector store: {e}. Will rebuild.", flush=True)
        vectorstore = None
        vectorstore_exists = False

if not vectorstore_exists:
    print("\nBuilding new vector store...", flush=True)
    build_start = time.time()
    all_docs = []

    # a) Load Knowledge Base
    print("  Loading Knowledge Base...", flush=True)
    if os.path.exists(KNOWLEDGE_BASE_DIR) and os.path.isdir(KNOWLEDGE_BASE_DIR):
        try:
            kb_loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
            kb_docs = kb_loader.load()
            # Add source metadata more clearly
            for doc in kb_docs:
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "kb_unknown"))
            print(f"    Loaded {len(kb_docs)} documents from Knowledge Base.", flush=True)
            all_docs.extend(kb_docs)
        except Exception as e: print(f"    Error loading knowledge base: {e}", flush=True)
    else: print(f"    Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found or not a directory. Skipping.", flush=True)

    # b) Load and Process Tactical Data (CSV)
    print(f"  Loading Tactical Data from {CSV_DATA_PATH}...", flush=True)
    if os.path.exists(CSV_DATA_PATH):
        try:
            df = pd.read_csv(CSV_DATA_PATH)
            print(f"    Loaded {len(df)} shots from CSV.", flush=True)
            rally_texts = []
            if not df.empty and 'rally_id' in df.columns:
                 df['rally_id'] = df['rally_id'].ffill().fillna(-1).astype(int)
                 grouped = df[df['rally_id'] != -1].groupby('rally_id')
                 print(f"    Processing {len(grouped)} rallies...", flush=True)
                 for name, group in grouped:
                     rally_text = format_rally_for_embedding(group)
                     if rally_text:
                         # Create Document with metadata pointing to the rally
                         rally_doc = Document(
                             page_content=rally_text,
                             metadata={"source": f"Rally_{name}"} # Use actual rally ID in metadata
                         )
                         all_docs.append(rally_doc)
                 print(f"    Formatted {len(grouped)} rallies into Documents.", flush=True)
            else: print("    CSV empty or missing 'rally_id'. Skipping tactical data processing.", flush=True)
        except Exception as e: print(f"    Error loading or processing CSV {CSV_DATA_PATH}: {e}", flush=True)
    else: print(f"    CSV file '{CSV_DATA_PATH}' not found. Skipping tactical data.", flush=True)

    # c) Split Documents
    if not all_docs: print("ERROR: No documents loaded. Cannot build vector store."); exit()
    print(f"\n  Splitting {len(all_docs)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"    Split into {len(split_docs)} chunks.", flush=True)

    # d) Create Vector Store and Save
    print(f"  Creating {VECTOR_STORE_TYPE} index from chunks...", flush=True)
    index_start_time = time.time()
    try:
        if VECTOR_STORE_TYPE == "FAISS":
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
        elif VECTOR_STORE_TYPE == "CHROMA":
             vectorstore = Chroma.from_documents(
                 documents=split_docs,
                 embedding=embeddings,
                 persist_directory=VECTORSTORE_PATH
             )
             # Chroma persists automatically during creation with persist_directory
        index_end_time = time.time()
        print(f"  Index created and saved to {VECTORSTORE_PATH}. Duration: {index_end_time - index_start_time:.2f}s", flush=True)
    except Exception as e: print(f"ERROR creating/saving index: {e}\n{traceback.format_exc()}"); exit()
    build_end = time.time()
    print(f"Vector store build complete. Duration: {build_end - build_start:.2f}s", flush=True)

# --- 3. Setup QA Chain ---
if vectorstore is None: print("ERROR: Vector store initialization failed."); exit()

print("\nSetting up RetrievalQA chain...", flush=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

template = """You are an expert badminton tactical analyst reviewing match data. Use ONLY the following pieces of retrieved context (which includes rules, tactical principles, and specific rally data) to answer the question concisely and factually. Do not add information not present in the context. If the context does not contain the answer, clearly state that the provided data does not have the information. Reference specific rally or shot numbers from the context if they support your answer.

Context:
---------
{context}
---------

Question: {question}

Concise Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Suitable for moderate context sizes
    retriever=retriever,
    return_source_documents=True, # Set to True to see retrieved chunks
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("RetrievalQA chain ready.", flush=True)

# --- 4. Query Loop ---
print("\n--- Badminton Tactical Query Interface ---")
print(f"(Using LLM: {LLM_MODEL}, Embeddings: {EMBEDDING_MODEL}, Vector Store: {VECTOR_STORE_TYPE})")
print("Enter your questions about the match analysis (e.g., 'Why did player 1 lose points on their backhand?', 'Show rallies where player 2 made an error after a drop shot', 'Was the center court controlled effectively?'). Type 'quit' to exit.")

while True:
    try:
        user_query = input("\n> ")
        if user_query.strip().lower() == 'quit':
            break
        if not user_query.strip():
            continue

        print("  Processing query...", flush=True)
        start_query_time = time.time()

        # Invoke the chain
        result = qa_chain.invoke({"query": user_query})
        end_query_time = time.time()

        print("\nAnswer:", flush=True)
        print(result["result"], flush=True)
        print(f"\n(Query duration: {end_query_time - start_query_time:.2f}s)", flush=True)

        # --- Optional: Display retrieved source documents ---
        if result.get("source_documents"):
             print("\n--- Retrieved Context Chunks (for debugging): ---", flush=True)
             for i, doc in enumerate(result["source_documents"]):
                 source = doc.metadata.get("source", "Unknown")
                 print(f"  Chunk {i+1} [Source: {source}]")
                 # Print first few lines of the chunk
                 content_preview = "\n".join(doc.page_content.splitlines()[:3])
                 print(f"    '{content_preview}...'\n")
        # -----------------------------------------------------

    except Exception as e:
        print(f"\nAn error occurred during query processing: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

print("\nExiting analysis session.")