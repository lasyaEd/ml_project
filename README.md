# 🎥 Match Cut Detection with Siamese Neural Networks

This repository provides tools to train a **Siamese Neural Network** for detecting **match cuts** in video frames and a **Streamlit application** for finding visually similar sections within YouTube videos.

---

## ✨ Features

### 📦 **1. Data Preparation**

This section describes how data is prepared for training the Siamese Neural Network to detect match cuts.

#### 🖼️ **Video Frame Extraction**
- Uses the `yt_dlp` library to fetch the highest quality streamable video URL from YouTube.
- Frames are extracted at specified intervals using OpenCV:
  - 📁 Saved in an organized folder structure (`frames/`).
  - 🎬 Scene changes are detected based on pixel intensity differences, and representative frames are saved in the `scenes/` folder.
- Ensures efficient and relevant data extraction for further analysis.

#### 📉 **Frame Downsampling**
- Reduces redundancy and computational load:
  - Selects every `N`th frame using the `downsample_frames` function.
  - Balances data size while retaining key visual elements.

#### 🌊 **Optical Flow Calculation**
- Computes dense optical flow between consecutive frames using the Farneback method:
  - 🌟 Calculates motion magnitude and direction to represent motion.
  - 🖼️ Saves visualized optical flow images in a `flow_images/` folder.

#### 📊 **Feature Extraction**
- Extracts optical flow features:
  - Computes motion intensity (mean magnitude) from grayscale flow images.
  - Features are used for similarity comparison between frames.

#### 🏷️ **Label Generation**
- Labels frame pairs based on motion intensity differences:
  - Applies a threshold to the difference in magnitudes.
  - Pairs with differences below the threshold are labeled as `1` (match), and others as `0` (no match).
- Stores labeled data in a CSV file (`flow_image_labeled_data.csv`) with columns `Flow_Image1`, `Flow_Image2`, and `Label`.

#### 🔍 **Visualization**
- Visualizes labeled pairs:
  - Displays selected pairs of flow images with their labels.
  - Validates the preprocessing and labeling process.

---

### 🧠 **2. Notebook: Training a Siamese Neural Network**

This notebook trains and evaluates a Siamese network for match cut detection.

#### 🔑 **Key Steps**
- **Load and preprocess data**: Reads image pairs and labels from `flow_image_labeled_data.csv`, resizes them to `224x224 pixels`, and normalizes.
- **Split data**: Divides the dataset into training and testing sets.
- **Build and train model**: Implements a Siamese architecture with shared convolutional layers and a distance-based binary classification head.
- **Evaluate performance**: Reports test accuracy and loss.
- **Visualize results**: Displays sample image pairs with similarity scores and plots training metrics.

> 🛠️ **Reproducibility**: Install required libraries (`pandas`, `scikit-learn`, `TensorFlow/Keras`, `OpenCV`, `matplotlib`) and run the notebook. The workflow assumes a Google Colab environment but is adaptable to other setups.

---

### 🌐 **3. Streamlit Application: Match Cutting on YouTube Videos**

This app allows users to detect visually similar sections in YouTube videos.

#### ⚙️ **Process**
- **Pre-trained model**: Downloads a Siamese network for image similarity via `huggingface_hub`.
- **Frame extraction**: Downloads a YouTube video, extracts frames, and stores them.
- **Reference selection**: Allows users to select a range of frames as the reference section.
- **Find similar sections**: Identifies sections with high similarity to the reference using the pre-trained model.
- **Display results**: Shows reference and similar sections as video clips with similarity scores.

---

## 🎯 Use Cases

- 🎬 **Film Editing**: Create seamless scene transitions.
- 📈 **Pattern Analysis**: Detect recurring visual motifs.
- 🔍 **Content Exploration**: Locate visually related sections for deeper analysis.

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/match-cut-detection.git
   cd match-cut-detection
