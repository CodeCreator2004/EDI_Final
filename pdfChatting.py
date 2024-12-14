from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
from tqdm import tqdm
import google.generativeai as genai
import spacy
import re
from groq import Groq

# Initialize the Groq client with API key from environment variable
client = Groq(
    api_key="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",  # Ensure the API key is set in your environment
)


def generate_response(query, matching_text, model="gemma2-9b-it"):
    print("Inside generate")
    """
    Generate a response using Groq's Generative AI model.

    Parameters:
        query (str): User's query input.
        matching_text (str): Context text to assist the AI in generating a relevant response.
        model (str): The Groq model to use for generation. Default is 'llama3-8b-8192'.

    Returns:
        str: The AI-generated response.
    """
    try:
        # Create a chat completion using query and context
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant providing detailed responses."},
                {"role": "user", "content": f"{query}\n\nContext:\n{matching_text}"},
            ],
            model=model,
        )
        # Return the generated content
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


def process_pdf(pdf_path, dbName):
    """Extract text, create embeddings, and store them in ChromaDB."""
    print("Processing PDF...")

    # Step 1: Extract Text
    def remove_escape(text: str) -> str:
        return text.replace("\n", " ").strip()

    reader = PdfReader(pdf_path)
    pdf_info = []
    for page_number in tqdm(range(len(reader.pages)), desc="Extracting pages"):
        page = reader.pages[page_number]
        text = page.extract_text()
        if text:
            text = remove_escape(text)
            pdf_info.append({"page_number": page_number, "text": text})

    # Step 2: Sentence Splitting
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    for item in tqdm(pdf_info, desc="Splitting sentences"):
        sentences = list(nlp(item["text"]).sents)
        item["sentences"] = [str(s) for s in sentences]

    # Step 3: Chunking
    chunk_size = 10
    split_list = lambda lst, n=chunk_size: [lst[i:i + n] for i in range(0, len(lst), n)]

    for item in pdf_info:
        sentences = item.get("sentences", [])
        chunks = split_list(sentences)
        item.update({"sentence_chunks": chunks})

    # Step 4: Create Embeddings
    model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    for item in tqdm(pdf_info, desc="Creating embeddings"):
        if "sentence_chunks" in item:
            joined_chunks = [" ".join(chunk) for chunk in item["sentence_chunks"]]
            embeddings = [model.encode(chunk) for chunk in joined_chunks]
            item["embeddings"] = embeddings

    # Step 5: Store in ChromaDB
    print("Storing data in ChromaDB...")
    client = chromadb.Client()
    collection_name = dbName
    collection = client.get_or_create_collection(name=collection_name)

    ids, embeddings = [], []
    for i, item in enumerate(pdf_info):
        for j, embedding in enumerate(item["embeddings"]):
            ids.append(f"{i}-{j}")  # Unique ID for each chunk
            embeddings.append(embedding.tolist())

    collection.add(ids=ids, embeddings=embeddings)

    print("PDF processing complete.")
    # print(pdf_info)
    return collection, pdf_info


def handle_query(collection, model, pdf_info, query):
    print("Enter in as")
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection)
    """Answer queries using stored embeddings."""
    query_embedding = model.encode(query).tolist()
    print("Enter in as")
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    matching_id = results["ids"][0][0]
    page_num, chunk_num = map(int, matching_id.split("-"))
    matching_text = " ".join(pdf_info[page_num]["sentence_chunks"][chunk_num])
    print(f"\nBest matching content on page number : {page_num + 1}")

    print(f"\nBest match from PDF:\n{matching_text}")

    # Generate AI response
    # genai.configure(api_key="AIzaSyDyHlnRMzW5SrLUiUWqH90GXxWnJRhDw7Q")
    # gen_model = genai.GenerativeModel("gemini-1.5-flash")
    # response = gen_model.generate_content([query, matching_text])
    # print("\nGenerative AI Response:")
    # print(response.text)
    response = generate_response(query, matching_text)
    print(response)
    response+=f"\n**Page No.{page_num+1}**"
    return response, (page_num + 1)

# pdf_path = input("Enter the path of the PDF: ").strip()
#
# while not os.path.isfile(pdf_path):
#         print("Invalid path. Please try again.")
#         pdf_path = input("Enter the path of the PDF: ").strip()
# collection, pdf_info = process_pdf(pdf_path,"MyDB")
# model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
#
# while True:
#         query = input("\nEnter your query (type 'exit' to stop): ").strip()
#         if query.lower() == 'exit':
#             print("Exiting the program. Goodbye!")
#             break
#         handle_query("pdf_embeddings1", model, pdf_info, query)
#

# while not os.path.isfile(pdf_path):
#     print("Invalid path. Please try again.")
#     pdf_path = input("Enter the path of the PDF: ").strip()
# collection, pdf_info = process_pdf(pdf_path)
# model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
#
# while True:
#     query = input("\nEnter your query (type 'exit' to stop): ").strip()
#     if query.lower() == 'exit':
#         print("Exiting the program. Goodbye!")
#         break
#     handle_query("pdf_embeddings1", model, pdf_info, query)


from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
from tqdm import tqdm
import google.generativeai as genai
import spacy
import re


# def process_pdf(pdf_path):
#     """Extract text, create embeddings, and store them in ChromaDB."""
#     print("Processing PDF...")
#
#     # Step 1: Extract Text
#     def remove_escape(text: str) -> str:
#         return text.replace("\n", " ").strip()
#
#     reader = PdfReader(pdf_path)
#     pdf_info = []
#     for page_number in tqdm(range(len(reader.pages)), desc="Extracting pages"):
#         page = reader.pages[page_number]
#         text = page.extract_text()
#         if text:
#             text = remove_escape(text)
#             pdf_info.append({"page_number": page_number, "text": text})
#
#     # Step 2: Sentence Splitting
#     nlp = spacy.blank("en")
#     nlp.add_pipe("sentencizer")
#
#     for item in tqdm(pdf_info, desc="Splitting sentences"):
#         sentences = list(nlp(item["text"]).sents)
#         item["sentences"] = [str(s) for s in sentences]
#
#     # Step 3: Chunking
#     chunk_size = 10
#     split_list = lambda lst, n=chunk_size: [lst[i:i + n] for i in range(0, len(lst), n)]
#
#     for item in pdf_info:
#         sentences = item.get("sentences", [])
#         chunks = split_list(sentences)
#         item.update({"sentence_chunks": chunks})
#
#     # Step 4: Create Embeddings
#     model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
#     for item in tqdm(pdf_info, desc="Creating embeddings"):
#         if "sentence_chunks" in item:
#             joined_chunks = [" ".join(chunk) for chunk in item["sentence_chunks"]]
#             embeddings = [model.encode(chunk) for chunk in joined_chunks]
#             item["embeddings"] = embeddings
#
#     # Step 5: Store in ChromaDB
#     print("Storing data in ChromaDB...")
#     client = chromadb.Client()
#     collection_name = "pdf_embeddings1"
#     collection = client.create_collection(name=collection_name)
#
#     ids, embeddings = [], []
#     for i, item in enumerate(pdf_info):
#         for j, embedding in enumerate(item["embeddings"]):
#             ids.append(f"{i}-{j}")  # Unique ID for each chunk
#             embeddings.append(embedding.tolist())
#
#     collection.add(ids=ids, embeddings=embeddings)
#
#     print("PDF processing complete.")
#     return collection, pdf_info

#
# def handle_query(collection, model, pdf_info, query):
#     """Answer queries using stored embeddings."""
#     query_embedding = model.encode(query).tolist()
#     results = collection.query(query_embeddings=[query_embedding], n_results=1)
#     matching_id = results["ids"][0][0]
#     page_num, chunk_num = map(int, matching_id.split("-"))
#     matching_text = " ".join(pdf_info[page_num]["sentence_chunks"][chunk_num])
#
#     print(f"\nBest match from PDF:\n{matching_text}")
#
#     # Generate AI response
#     genai.configure(api_key="AIzaSyDyHlnRMzW5SrLUiUWqH90GXxWnJRhDw7Q")
#     gen_model = genai.GenerativeModel("gemini-1.5-flash")
#     response = gen_model.generate_content([query, matching_text])
#     print("\nGenerative AI Response:")
#     print(response.text)


# Main Program
# if __name__ == "_main_":
#     pdf_path = input("Enter the path of the PDF: ").strip()
#     while not os.path.isfile(pdf_path):
#         print("Invalid path. Please try again.")
#         pdf_path = input("Enter the path of the PDF: ").strip()
#
#     collection, pdf_info = process_pdf(pdf_path)
#     model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
#
#     while True:
#         query = input("\nEnter your query (type 'exit' to stop): ").strip()
#         if query.lower() == 'exit':
#             print("Exiting the program. Goodbye!")
#             break
#         handle_query(collection, model, pdf_info, query)