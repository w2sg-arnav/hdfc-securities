import streamlit as st
from pathlib import Path
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from together import Together
import PIL.Image
import google.generativeai as genai
from pdf2image import convert_from_path
import time
from dotenv import load_dotenv
from pypdf import PdfReader
import tempfile
import uuid

load_dotenv()

# --- Environment Variables ---
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_BASE_INDEX_NAME = os.getenv("PINECONE_BASE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- Helper Functions ---
def load_prompt(prompt_file: Path) -> str:
    """Loads the prompt from the given text file."""
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file}. Using default prompt.")
        return ""
    except Exception as e:
        print(f"Error loading prompt from {prompt_file}: {e}. Using default prompt.")
        return ""


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Extracts text from a PDF file, page by page, using pypdf.
    If pypdf extraction fails or extracts less than 20 words, it falls back to Gemini Vision API.
    """
    try:
        reader = PdfReader(pdf_path)
        page_text = {}
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()
            page_text[page_number + 1] = text

        total_words = sum(len(text.split()) for text in page_text.values())
        if total_words >= 20:
            return page_text
        else:
            print(
                f"pypdf extracted less than 20 words ({total_words}). Falling back to Gemini Vision API."
            )
    except Exception as e:
        print(
            f"An error occurred during PDF extraction with pypdf: {e}. Falling back to Gemini Vision API."
        )
        # Fallback to Gemini Vision API
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            images = convert_from_path(pdf_path)
            gemini_page_text = {}
            pdf_name = os.path.splitext(os.path.basename(str(pdf_path)))[0]
            output_dir = "GeminiVisionResult"
            os.makedirs(output_dir, exist_ok=True)

            prompt_file_path = Path("prompt.txt")
            prompt = load_prompt(prompt_file_path)

            if not images:
                raise FileNotFoundError(f"Could not convert the PDF to images")

            for i, img in enumerate(images):
                page_number = i + 1
                output_file_path = os.path.join(
                    output_dir, f"{pdf_name}_{page_number}.txt"
                )

                try:
                    response = model.generate_content(
                        [prompt, img], generation_config={"max_output_tokens": 4096}
                    )
                    response.resolve()
                    gemini_page_text[page_number] = response.text
                    print(f"Gemini processed page {page_number}")
                except Exception as page_err:
                    print(f"Error processing page {page_number} with Gemini: {page_err}")
                    gemini_page_text[
                        page_number
                    ] = f"Error: An error occurred during Gemini processing of page {page_number}: {page_err}"
            return gemini_page_text
        except FileNotFoundError as e:
            print(f"Error: Could not find file: {e}")
            return {}
        except Exception as e:
            print(f"Error during Gemini Vision API processing: {e}")
            return {}


def semantic_chunking(text_dict: dict, chunk_size: int = 500, chunk_overlap: int = 50):
    """Chunks the extracted text semantically."""
    all_text = "\n".join(text_dict.values())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_text(all_text)
    return chunks


def embed_and_upsert_to_pinecone(chunks: list[str], index):
    """Embeds text chunks and upserts them to Pinecone."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        ids = [f"chunk-{i}-{j}" for j in range(len(batch_chunks))]
        embeds = embeddings.embed_documents(batch_chunks)
        metadata = [{"text": text} for text in batch_chunks]
        to_upsert = list(zip(ids, embeds, metadata))
        index.upsert(vectors=to_upsert)
    print(f"Upserted {len(chunks)} chunks to Pinecone.")


def generate_response(
    query: str, context: str, model_name: str = "meta-llama/Llama-3-8b-chat-hf"
):
    """Generates a response using Together AI's API."""
    if not TOGETHER_AI_API_KEY:
        raise ValueError("TOGETHER_AI_API_KEY environment variable not set.")

    client = Together(api_key=TOGETHER_AI_API_KEY)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content


def query_pinecone(query: str, index, top_k: int = 15):
    """Queries Pinecone for relevant chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector, top_k=top_k, include_values=False, include_metadata=True
    )
    context = "\n\n".join([match.metadata["text"] for match in results.matches])
    return context


# --- RAG Chatbot Class ---
class RAGChatbot:
    def __init__(self, existing_index_name):
        if not PINECONE_API_KEY:
            raise ValueError("Pinecone API key must be set.")
        self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        # Generate a unique index name for each run
        self.pinecone_index_name = f"{existing_index_name}-{str(uuid.uuid4())[:8]}"
        print(f"Creating a new Pinecone index: '{self.pinecone_index_name}'...")

        try:
            spec = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
            self.pc.create_index(name=self.pinecone_index_name, dimension=768, spec=spec)
            self.index = self.pc.Index(self.pinecone_index_name)
            print(f"Successfully created and connected to Pinecone index: '{self.pinecone_index_name}'")
        except Exception as create_e:
           raise Exception(f"Error creating Pinecone index: '{self.pinecone_index_name}': {create_e}")


    def ingest_pdfs(self, pdf_paths: list[Path]):
        """Ingests a list of PDFs, chunks them, and uploads to Pinecone."""
        for pdf_path in pdf_paths:
            print(f"Processing PDF: {pdf_path}")
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text:
                chunks = semantic_chunking(extracted_text)
                embed_and_upsert_to_pinecone(chunks, self.index)
            else:
                print(f"No text extracted from the PDF: {pdf_path}")

    def query(self, query: str):
       """Queries the chatbot with a user's question."""
       context = query_pinecone(query, self.index)
       if not context:
           return "No relevant information found in the document."
       response = generate_response(query, context)
       return response


# --- Streamlit App ---
def main():
    st.title("RAG Chatbot with Pinecone")

    # Initialize or load the existing Pinecone index
    existing_index_name = PINECONE_BASE_INDEX_NAME
    try:
        chatbot = RAGChatbot(existing_index_name)
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        pdf_paths = []
        with st.spinner("Ingesting uploaded PDFs..."):
            try:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        pdf_paths.append(Path(tmp_file.name))
                chatbot.ingest_pdfs(pdf_paths)
                for file_path in pdf_paths:
                    os.unlink(file_path)
                st.success("PDFs ingested successfully!")
            except Exception as e:
                st.error(f"Error during PDF ingestion: {e}")

    # Input area for the query
    query = st.text_input("Enter your query:", "")

    # Query the bot and display the response
    if query:
        with st.spinner("Generating response..."):
            try:
                response = chatbot.query(query)
                st.write("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error processing the query: {e}")


if __name__ == "__main__":
    main()