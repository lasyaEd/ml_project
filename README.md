# Match Cut Detection with Siamese Neural Networks

This repository provides tools to train a **Siamese Neural Network** for detecting **match cuts** in video frames and a **Streamlit application** for finding visually similar sections within YouTube videos.

---

## Features

### **1. Notebook: Training a Siamese Neural Network**
This notebook trains and evaluates a Siamese network for match cut detection, including the following steps:

- **Load and preprocess data**: Reads image pairs and labels from `flow_image_labeled_data.csv`, resizes to `224x224 pixels`, and normalizes.
- **Split data**: Divides the dataset into training and testing sets.
- **Build and train model**: Implements a Siamese architecture with shared convolutional layers and a distance-based binary classification head.
- **Evaluate performance**: Reports test accuracy and loss.
- **Visualize results**: Displays sample image pairs with similarity scores and plots training metrics.

> **Reproducibility**: Install required libraries (`pandas`, `scikit-learn`, `TensorFlow/Keras`, `OpenCV`, `matplotlib`) and run the notebook. The workflow assumes a Google Colab environment but is adaptable to other setups.

---

### **2. Streamlit Application: Match Cutting on YouTube Videos**
This app allows users to detect visually similar sections in YouTube videos. The process includes:

- **Pre-trained model**: Downloads a Siamese network for image similarity via `huggingface_hub`.
- **Frame extraction**: Downloads a YouTube video, extracts frames, and stores them.
- **Reference selection**: Users select a range of frames as the reference section.
- **Find similar sections**: Identifies sections with high similarity to the reference using the pre-trained model.
- **Display results**: Shows reference and similar sections as video clips with similarity scores.

---

## Use Cases

- **Film Editing**: Create seamless scene transitions.
- **Pattern Analysis**: Detect recurring visual motifs.
- **Content Exploration**: Locate visually related sections for deeper analysis.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/match-cut-detection.git
   cd match-cut-detection
