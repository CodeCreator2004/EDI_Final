from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from pdfChatting import generate_response


# Assuming the necessary import statements for Google Generative AI and others are included
# Importing modules like spacy, re, etc., as in the original code

# Function to process the PDF and create embeddings (as given)
def process_pdf(pdf_path, collection_name):
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file at {pdf_path} was not found.")
        return None

    try:
        # Example logic for processing the PDF
        reader = PdfReader(pdf_path)
        collection = []  # Replace with actual collection initialization logic
        pdf_info = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pdf_info.append({"page_number": i, "text": text})
                # Add code to populate `collection` with embeddings

        # Ensure collection has been populated (add actual logic here)
        collection = "mock_collection"  # Replace with real collection object
        return collection, pdf_info
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None



# Modify the handle_query function to return true/false based on the correctness of retrieval
def handle_query(collection, model, pdf_info, query, ground_truth_page):
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection)

    # Encode the query and search for the best match
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    # Check if results were returned
    if not results["ids"] or not results["ids"][0]:
        print(f"No results found for query: '{query}'")
        return None, None, False  # Return with `None` values and mark as incorrect

    # Extract the matching ID
    matching_id = results["ids"][0][0]
    page_num, chunk_num = map(int, matching_id.split("-"))
    matching_text = " ".join(pdf_info[page_num]["sentence_chunks"][chunk_num])

    # Check if the retrieved page is correct
    is_correct = (page_num + 1 == ground_truth_page)

    # Generate a response using the matching text
    response = generate_response(query, matching_text)
    response += f"\n**Page No.{page_num + 1}**"

    return response, (page_num + 1), is_correct



# Function to evaluate metrics and plot graphs
def evaluate_model(queries, ground_truth_pages, collection, model, pdf_info):
    correct_retrievals = []

    # Run each query and check if the retrieval is correct
    for i, (query, true_page) in enumerate(zip(queries, ground_truth_pages)):
        _, _, is_correct = handle_query(collection, model, pdf_info, query, true_page)
        correct_retrievals.append(is_correct)
        print(f"Query {i + 1}/{len(queries)}: {'Correct' if is_correct else 'Incorrect'}")

    # Calculate metrics
    y_true = [1] * len(ground_truth_pages)  # All true labels are 1 (correct retrieval)
    y_pred = [1 if is_correct else 0 for is_correct in correct_retrievals]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    scores = [precision, recall, f1, accuracy]
    plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightpink'])
    plt.ylim(0, 1)
    plt.title("Model Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.show()


# Sample test setup
if __name__ == "__main__":
    pdf_path = "O:\\mainframe\\Security-and-Privacy.pdf"
    collection_name = "IBMZ_Security_Privacy"

    # Process the PDF and create a collection in ChromaDB
    result = process_pdf(pdf_path, collection_name)
    if result is not None:
        collection, pdf_info = result
    else:
        raise RuntimeError("Failed to process the PDF. Please ensure the PDF path is correct and the function is implemented properly.")

    # Load the model
    model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

    # Define the test queries and their expected ground truth page numbers
    test_queries = [
        "What is the difference between data security and data privacy?",
        "Explain how IBM RACF manages access control in z/OS.",
        "What are the components involved in real-time monitoring and alerting for IBM Z?",
        "Describe how SAF works within the z/OS system.",
        "How was IBM mainframe security managed before public internet access?",
        "What is the role of pervasive encryption in z/OS components?",
        "Explain the analogy of an Italian sub sandwich in the context of security layers.",
        "What is the process by which a user gains or is denied access in z/OS?",
        "How does multi-factor authentication enhance security in IBM Z?",
        "What are the best practices for ensuring data privacy in z/OS?"
    ]
    ground_truth_pages = [1, 2, 4, 3, 5, 6, 1, 4, 3, 2]  # Adjust based on actual PDF content

    # Evaluate the model's performance
    evaluate_model(test_queries, ground_truth_pages, collection_name, model, pdf_info)
