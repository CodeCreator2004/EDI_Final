import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import assemblyai as aai
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize Groq API client
client = Groq(
    api_key="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"
)

# Initialize models and settings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
aai.settings.api_key = "c66511096b124182b1c7a932738790bb"


def upload_video():
    """
    Function to upload a video file.
    Returns the file path of the uploaded video.
    """
    video_path = input("Enter the path of the video file to process: ")
    if not os.path.exists(video_path):
        print("Error: File does not exist. Please provide a valid path.")
        return None
    return video_path


def process_video(video_path):
    """
    Processes the uploaded video to extract frames, transcribe audio,
    and generate text embeddings for each frame.
    """
    frame_interval = 20  # Time interval in seconds to extract frames
    frame_dir = "frames"  # Directory to store frames

    # Ensure the frame directory exists
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Extract audio from video
    clip = VideoFileClip(video_path)
    duration = clip.duration
    audio_path = "audio.wav"
    clip.audio.write_audiofile(audio_path)

    # Transcribe audio using AssemblyAI
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed: {transcript.error}")
        return None

    # Map the transcription into 20-second segments
    segments = []
    words = transcript.words
    current_segment = []
    current_start_time = 0

    for word in words:
        if word.end >= current_start_time + frame_interval * 1000:
            segments.append(' '.join(w.text for w in current_segment))
            current_segment = []
            current_start_time += frame_interval * 1000
        current_segment.append(word)
    if current_segment:
        segments.append(' '.join(w.text for w in current_segment))

    # Extract frames and associate with text
    frames_with_text = {}
    frames_with_timestamp = {}
    for i, t in enumerate(range(0, int(duration), frame_interval)):
        frame = clip.get_frame(t)
        frame_filename = os.path.join(frame_dir, f'frame_{i}.jpg')
        cv2.imwrite(frame_filename, frame)  # Save frame
        frames_with_text[frame_filename] = segments[i] if i < len(segments) else ""
        frames_with_timestamp[frame_filename] = t  # Store timestamp for the frame

    # Create embeddings for each frame's text
    frames_with_embeddings = {}
    for frame, text in frames_with_text.items():
        if text.strip():  # Ensure there's text to process
            embedding = embedding_model.encode(text)
            frames_with_embeddings[frame] = embedding

    return frames_with_text, frames_with_embeddings, frames_with_timestamp


# Function to ask a query to the Groq API
def ask_groq_api(query, frames_with_text):
    """
    This function sends the query and associated text data to the Groq API
    and returns the generated answer.
    """
    # Combine all transcribed text into a single data string
    video_data = " ".join(frames_with_text.values())

    # Send the query and video data to the Groq API
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Query: {query} Data: {video_data}",
        }],
        model="gemma2-9b-it",
    )

    # Access the content of the response properly
    if response.choices:
        answer = response.choices[0].message.content  # Corrected line
    else:
        answer = "No answer found."

    return answer


# Function to get the most similar frame based on the query
def get_most_similar_frame(query, frames_with_text, frames_with_timestamp,frames_with_embeddings):
    """
    This function finds the frame with the most similar text to the query.
    It returns the frame and its timestamp.
    """
    # Create embedding for the query text
    query_embedding = embedding_model.encode(query)

    # Find the frame with the highest similarity to the query
    best_similarity = -1
    relevant_frame = None
    relevant_timestamp = None
    for frame, text in frames_with_text.items():
        frame_embedding = frames_with_embeddings.get(frame)
        if frame_embedding is not None:
            similarity = np.dot(query_embedding, frame_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(frame_embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                relevant_frame = frame
                relevant_timestamp = frames_with_timestamp.get(frame)

    return relevant_frame, relevant_timestamp


# # Main logic
# video_path = upload_video()
# if video_path:
#     frames_with_text, frames_with_embeddings, frames_with_timestamp = process_video(video_path)
#
#     # Ask a query about the video
#     query = input("Enter your query about the video: ")
#
#     # Get the most relevant frame and timestamp for the query
#     relevant_frame, relevant_timestamp = get_most_similar_frame(query, frames_with_text, frames_with_timestamp)
#
#     if relevant_frame:
#         # Load and display the relevant frame
#         img = cv2.imread(relevant_frame)
#         if img is not None:
#             resize_width = 800
#             height, width = img.shape[:2]
#             resize_height = int((resize_width / width) * height)
#             resized_img = cv2.resize(img, (resize_width, resize_height))
#             cv2.imshow(f"Relevant Frame - Timestamp {relevant_timestamp}s", resized_img)
#             print(f"Timestamp of this frame: {relevant_timestamp}s")
#             print(f"Answer from Groq API: {ask_groq_api(query, frames_with_text)}")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#     else:
#         print("No relevant frame found.")