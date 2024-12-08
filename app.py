import streamlit as st
import cv2
import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from yt_dlp import YoutubeDL
import warnings

# Load the environment variables
load_dotenv()
# Retrieve the token from the .env file
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=token)


warnings.filterwarnings("ignore")

# Function to build the base network for feature extraction
def build_base_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return x

# Function to save frames as a video
def save_frames_as_video(frames, output_path, fps=30):
    if not frames:
        raise ValueError("No frames provided to save as video.")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

# Function to find similar sections
def find_similar_sections(reference_section, video_frames, section_length=5, step=1, top_n=5, exclude_range=None):
    def preprocess_frame(frame):
        frame_preprocessed = cv2.resize(frame, (224, 224)) / 255.0
        frame_preprocessed = frame_preprocessed[np.newaxis, ...]
        return frame_preprocessed

    def compute_section_similarity(ref_frames, candidate_frames):
        similarities = []
        for ref_frame, candidate_frame in zip(ref_frames, candidate_frames):
            ref_frame_processed = preprocess_frame(ref_frame)
            candidate_frame_processed = preprocess_frame(candidate_frame)
            similarity = siamese_model.predict([ref_frame_processed, candidate_frame_processed])[0][0]
            similarities.append(similarity)
        return np.mean(similarities)

    reference_frames = reference_section
    similarities = []
    for i in range(0, len(video_frames) - section_length + 1, step):
        if exclude_range and (exclude_range[0] <= i <= exclude_range[1]):
            continue
        candidate_frames = video_frames[i:i + section_length]
        similarity = compute_section_similarity(reference_frames, candidate_frames)
        similarities.append((i, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Function to get YouTube video stream URL
def get_youtube_stream_url(video_id):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
        return info['url']

# Function to extract frames from a video stream
def extract_frames_from_stream(video_url, interval=1):
    cap = cv2.VideoCapture(video_url)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (int(frame_rate) * interval) == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# Model setup
try:
    input_shape = (224, 224, 3)

    # Create inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    tower = tf.keras.models.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu')
    ])

    embedding_a = tower(input_a)
    embedding_b = tower(input_b)
    distance = Lambda(lambda x: tf.math.abs(x[0] - x[1]))([embedding_a, embedding_b])
    output = Dense(1, activation='sigmoid')(distance)
    siamese_model = Model(inputs=[input_a, input_b], outputs=output)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model_path = hf_hub_download(repo_id="samanthajmichael/siamese_model.h5", filename="siamese_model.h5")
    siamese_model.load_weights(model_path)
    st.success("Model loaded successfully!")

    st.title("Match Cutting with YouTube")

    video_id = st.text_input("Enter YouTube Video ID (must be under 5 minutes):")
    frames = []

    if video_id:
        st.write("Processing YouTube video on the fly...")
        stream_url = get_youtube_stream_url(video_id)
        frames = extract_frames_from_stream(stream_url, interval=1)
        st.write(f"Extracted {len(frames)} frames from the video.")

    if frames:
        st.write("Select a range of frames to use as the reference:")
        start_frame_index = st.slider("Start Frame Index:", min_value=0, max_value=len(frames) - 1, value=0)
        end_frame_index = st.slider("End Frame Index:", min_value=start_frame_index, max_value=len(frames) - 1, value=start_frame_index + 5)
        selected_frames = frames[start_frame_index:end_frame_index + 1]
        st.write(f"Selected Frames: {start_frame_index} to {end_frame_index}")
        # st.image(selected_frames[0], caption="Selected Section Start Frame", use_container_width=True)
        # if st.button("Play Selected Frames as Video"):
        selected_video_path = "selected_frames.mp4"
        save_frames_as_video(selected_frames, selected_video_path, fps=10)
        st.video(selected_video_path)

        if st.button("Find Similar Sections"):
            st.write("Finding similar flowing sections...")
            exclude_range = (start_frame_index, end_frame_index)
            top_sections = find_similar_sections(selected_frames, frames, len(selected_frames), step=1, top_n=5, exclude_range=exclude_range)

            for rank, (start_idx, similarity) in enumerate(top_sections, 1):
                similar_section = frames[start_idx:start_idx + len(selected_frames)]
                st.write(f"Match {rank} - Similarity Score: {similarity:.2f}, Frames {start_idx} to {start_idx + len(selected_frames) - 1}")
                # st.image(similar_section[0], caption=f"Section Start Frame (Match {rank})", use_container_width=True)
                video_path = f"similar_section_{rank}.mp4"
                save_frames_as_video(similar_section, video_path, fps=10)
                # st.write(f"Similar Section {rank} - Frames {start_idx} to {start_idx + len(selected_frames) - 1}")
                st.video(video_path)


except Exception as e:
    st.error(f"Error loading model: {str(e)}")