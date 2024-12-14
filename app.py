from flask import Flask, render_template, request, jsonify, session
import chromadb
from pdfChatting import process_pdf
import os
from pdfChatting import handle_query
from sentence_transformers import SentenceTransformer
import numpy as np
import base64
from PyPDF2 import PdfReader
from tqdm import tqdm
from videoChatting import process_video, get_most_similar_frame, ask_groq_api
import cv2

import re

# Video
# Global variables to store video processing data
info = {}
frames_with_text = {}
frames_with_embeddings = {}
frames_with_timestamps = {}


def convert_to_html(input_text):
    html_output = ""
    # Replace special markers with appropriate HTML tags
    lines = input_text.split('\n')
    for line in lines:
        line = line.strip()

        if line.startswith('**') and line.endswith('**'):
            # Convert to <h2>
            html_output += f"<h2>{line.strip('**')}</h2>\n"
        elif line.startswith('* ') and line.endswith(':'):
            # Convert to <h3>
            html_output += f"<h3>{line.strip('* ')}</h3>\n"
        elif line.startswith('* '):
            # Convert to list item
            html_output += f"<li>{line.strip('* ')}</li>\n"
        elif line == '':
            continue
        else:
            # Convert to paragraph for normal text
            html_output += f"<p>{line}</p>\n"

    # html_output += f"<marked><h2>Page.No {page}</h2></marked>"

    return html_output


app = Flask(__name__)


# markdown_converter
def convert_to_markdown(response):
    # Start with the basic markdown structure
    markdown_response = "# Topics List\n\n"
    markdown_response += "This response provides detailed explanations for each topic you've listed, along with additional context and examples.\n\n"

    # Iterate through the response and convert each section to markdown
    for section in response:
        # Add the section title
        markdown_response += f"## {section['title']}\n\n"

        for subsection in section['subsections']:
            # Add subsection title (e.g., Basics, Comparison, etc.)
            markdown_response += f"### {subsection['title']}:\n\n"

            for point in subsection['points']:
                # Add points as bullet points
                markdown_response += f"- **{point['label']}:** {point['description']}\n"

            markdown_response += "\n"  # Add a new line after each subsection

    return markdown_response


client = chromadb.Client()
collection = client.get_or_create_collection("MyDB")

MyFile = ""
# Configure the directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def remove_escape(text: str) -> str:
    return text.replace("\n", " ").strip()


def getPdfInfo(pdf_path):
    reader = PdfReader(pdf_path)
    pdf_info = []
    for page_number in tqdm(range(len(reader.pages)), desc="Extracting pages"):
        page = reader.pages[page_number]
        text = page.extract_text()
        if text:
            text = remove_escape(text)
            pdf_info.append({"page_number": page_number, "text": text})
    return pdf_info


# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def landing_page():
    return render_template('landing.html')


@app.route('/getstarted')
def dashboard():
    return render_template('dashboard.html')


@app.route('/pdfChat')
def pdfChat():
    return render_template('pdfChat.html')

@app.route('/videoChat')
def videoChat():
    return render_template('videoChat.html')


@app.route('/uploadPdf', methods=['POST'])
def upload_pdf():
    if 'pdfFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['pdfFile']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Secure the filename
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the file
    file.save(filepath)
    pdf_Path = "uploads/" + filename
    MyFile = filename
    collection, pdf = process_pdf(pdf_Path, "MyDB")
    pdf_info = pdf

    return jsonify({'message': 'File uploaded successfully', 'file_path': filepath}), 200


@app.route('/getResponsePdf', methods=['POST'])
def get_response_pdf():
    model = SentenceTransformer(
        'Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    data = request.get_json()
    query = data.get('query')
    # Ensure this path is correct
    pdf_Path = "uploads/Unit1_mainframe.pdf"

    try:
        # Process the query
        # Ensure this returns data properly
        collection, pdf_info = process_pdf(pdf_Path, "MyDB")
        print(pdf_info)  # Log PDF info to see if it's processed correctly

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        response_tuple = handle_query("MyDB", model, pdf_info, query)
        # Extract the main response part (adjust the index as needed)
        response = response_tuple[0]
        print(response)

        response = convert_to_html(response)
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Error in get_response_pdf: {e}")  # Log the error message
        # Send the error in the response
        return jsonify({'error': str(e)}), 500


# Video Chat

@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['videoFile']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Secure the filename
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the file
    file.save(filepath)
    video_path = "uploads/" + filename

    # Process the video
    global info, frames_with_text, frames_with_embeddings, frames_with_timestamps
    info = filename
    frames_with_text, frames_with_embeddings, frames_with_timestamps = process_video(video_path)

    if frames_with_text is None:
        return jsonify({'error': 'Video processing failed'}), 500

    return jsonify({'message': 'File uploaded successfully', 'file_path': filepath}), 200

@app.route('/getResponseVid', methods=['POST'])
def get_response_Vid():
    data = request.get_json()
    query = data.get('query', '')
    # Ensure this path is correct

    try:
        # Process the query
        # Ensure this returns data properly


        if not query:
            return jsonify({'error': 'No query provided'}), 400

        relevant_frame, relevant_timestamp = get_most_similar_frame(query, frames_with_text, frames_with_timestamps,frames_with_embeddings)
        response_tuple1 = ""
        if relevant_frame:
            response_tuple1 = ask_groq_api(query, frames_with_text)

        # Extract the main response part (adjust the index as needed)
        response = response_tuple1[0]
        print(response)

        response = convert_to_html(response)
        return jsonify({'response': response}), 200
    except Exception as e:
        print(f"Error in get_response_pdf: {e}")  # Log the error message
        # Send the error in the response
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
