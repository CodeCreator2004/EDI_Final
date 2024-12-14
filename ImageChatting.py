from PIL import Image
from io import BytesIO
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Generative AI API key
genai.configure(api_key="AIzaSyCIMXACFt5_JJZ01DdWfS3RgmsLxGtbBc4")  # Replace with your API key

# Configure ChromaDB client
client = chromadb.Client()

# Load the embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Upload Image and Generate Embeddings
def upload_image_embedding(image_url, collection_name):
    # Fetch the image from the URL
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
    else:
        raise ValueError("Failed to fetch image from URL")

    # Describe the image using Generative AI
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = "Describe the given image, identifying objects, people, places, and text present in it."
    response = model.generate_content([prompt,img])
    description = response.text
    print(f"Generated Description: {description}")

    # Embed the description
    embeddings = embedding_model.encode(description, convert_to_tensor=False)

    # Store the embedding in ChromaDB
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        ids=[image_url],  # Use image URL as unique ID
        embeddings=[embeddings.tolist()],
        metadatas=[{"description": description}]
    )
    print(f"Embedding stored in collection '{collection_name}'")
    return collection

# Ask Query Based on Image Embedding
def ask_query_image(collection_name, query):
    # Retrieve collection
    collection = client.get_or_create_collection(name=collection_name)

    # Generate embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)

    # Query the ChromaDB collection
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    if not results["ids"]:
        return "No relevant information found for the query."

    # Retrieve the matching description
    matching_metadata = results["metadatas"][0][0]["description"]
    print(f"Matched Description: {matching_metadata}")

    # Generate answer using Generative AI
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Query: {query}\nContext: {matching_metadata}"
    response = model.generate_content([prompt])
    return response.text

# Main Function
if __name__ == "__main__":
    # Example image URL
    image_url = "https://i0.wp.com/www.mendmotor.com/wp-content/uploads/2023/12/Car-Parts-Diagram-with-name.webp?resize=840%2C473&ssl=1"
    collection_name = "myDB"

    # Upload image and create embeddings
    try:
        upload_image_embedding(image_url, collection_name)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # Query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        try:
            answer = ask_query_image(collection_name, query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
