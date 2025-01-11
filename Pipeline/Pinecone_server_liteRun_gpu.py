import fitz
import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from together import Together
import PIL.Image
import os
import google.generativeai as genai
from pdf2image import convert_from_path
import time

# Ensure you have these environment variables set
TOGETHER_AI_API_KEY = "64880c44ef37384040dc253c954ed2f190c0e4702c3e80745e5eb78221f47376"  
PINECONE_API_KEY = "pcsk_2aEGcj_7cwy95qcT59b57wGLdNgNquJdiTiBJXNU27UiEob5cisrASpM99fcBHPeHwxp4U"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_BASE_INDEX_NAME = "rag-chatbot-index"  
GOOGLE_API_KEY = "AIzaSyBe7hdWbsCf6kQmyoMAUXbOlr7p8v1Tjhk"


model_kwargs = {'device': 'cuda:0'}
model_name = "sentence-transformers/all-mpnet-base-v2"

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
    Extracts text from a PDF file, page by page, using fitz.
    If fitz extraction fails or extracts less than 20 words, it falls back to Gemini Vision API.

    Args:
      pdf_path: Path to the input PDF file.

    Returns:
      A dictionary where keys are page numbers (starting from 1) and values are the
      extracted text from that page.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        page_text = {}
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text()
            page_text[page_number + 1] = text
        pdf_document.close()

        total_words = sum(len(text.split()) for text in page_text.values())
        if total_words >= 20:
            return page_text
        else:
            print(f"fitz extracted less than 20 words ({total_words}). Falling back to Gemini Vision API.")
    except Exception as e:
        print(f"An error occurred during PDF extraction with fitz: {e}. Falling back to Gemini Vision API.")

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
            output_file_path = os.path.join(output_dir, f"{pdf_name}_{page_number}.txt")

            try:
                response = model.generate_content([prompt, img], generation_config={"max_output_tokens": 4096})
                response.resolve()
                gemini_page_text[page_number] = response.text
                print(f"Gemini processed page {page_number}")
            except Exception as page_err:
                print(f"Error processing page {page_number} with Gemini: {page_err}")
                gemini_page_text[page_number] = f"Error: An error occurred during Gemini processing of page {page_number}: {page_err}"
        return gemini_page_text

    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
        return {}
    except Exception as e:
        print(f"Error during Gemini Vision API processing: {e}")
        return {}

def semantic_chunking(text_dict: dict, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Chunks the extracted text semantically.

    Args:
      text_dict: Dictionary of page-wise extracted text.
      chunk_size: Maximum size of each chunk.
      chunk_overlap: Number of overlapping characters between chunks.

    Returns:
      A list of text chunks.
    """
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
    """
    Embeds the text chunks using HuggingFace embeddings and upserts them to Pinecone.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        ids = [f"chunk-{i}-{j}" for j in range(len(batch_chunks))]
        embeds = embeddings.embed_documents(batch_chunks)
        metadata = [{"text": text} for text in batch_chunks]
        to_upsert = list(zip(ids, embeds, metadata))
        index.upsert(vectors=to_upsert)
    print(f"Upserted {len(chunks)} chunks to Pinecone.")

def generate_response(query: str, context: str, model_name: str = "meta-llama/Llama-3-8b-chat-hf"):
    """
    Generates a response using Together AI's API with the `together` library.

    Args:
      query: The user's question.
      context: Retrieved relevant text from Pinecone.
      model_name: The name of the Together AI model to use.

    Returns:
      The LLM's response.
    """
    if not TOGETHER_AI_API_KEY:
        raise ValueError("TOGETHER_AI_API_KEY environment variable not set.")

    client = Together(api_key=TOGETHER_AI_API_KEY)

    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7
    )

    return response.choices[0].message.content

def query_pinecone(query: str, index, top_k: int = 5):
    """
    Queries Pinecone for relevant chunks.

    Args:
      query: The user's question.
      index: The Pinecone index to query.
      top_k: Number of relevant chunks to retrieve.

    Returns:
      A string containing the concatenated text of the top_k chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    query_vector = embeddings.embed_query(query)

    results = index.query(vector=query_vector, top_k=top_k, include_values=False, include_metadata=True)
    context = "\n\n".join([match.metadata["text"] for match in results.matches])
    return context

class RAGChatbot:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("Pinecone API key must be set.")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.pinecone_index_name = f"{PINECONE_BASE_INDEX_NAME}-{int(time.time())}"
        print(f"Creating Pinecone index '{self.pinecone_index_name}'...")
        try:
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=768,  # Dimension of all-mpnet-base-v2 embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )
            self.index = self.pc.Index(self.pinecone_index_name)
        except Exception as e:
            raise Exception(f"Error creating Pinecone index: {e}")

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
    

pdf_files = [
    Path('Concord Enviro Systems Limited_RHP.pdf'),
    Path('DAM Capital Advisors Limited_RHP.pdf'),
    Path('Ventive Hospitality Limited_RHP.pdf'),
    Path('NewMalayalam Steel Limited_RHP.pdf'),
]

chatbot = RAGChatbot()

chatbot.ingest_pdfs(pdf_files)
print("Pinecone setup and ready for querying.")