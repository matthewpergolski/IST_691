import streamlit as st
import pandas as pd
import numpy as np
import pickle
from bird_song_dataset import BirdSongDataset, DataPaths
import librosa
import plotly.express as px
from streamlit_tensorboard import st_tensorboard
from tensorboard import program
import json
import os

# Define the project root directory
PROJECT_ROOT = os.getcwd()

# Function to start TensorBoard
def start_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', '6008'])
    url = tb.launch()
    return url

################################## Overview Page ##################################

st.set_page_config(layout="wide")

# Get dynamic paths
data_paths = DataPaths()
paths = data_paths.get_paths()

# Instantiate dataset class
bird_dataset = BirdSongDataset(csv_file=paths['csv_file_path'], root_dir=paths['wav_files_dir'])

# Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ("Overview", "Exploratory Data Analysis (EDA)", "Model Training", "Model Inference"))

if choice == "Overview":
    st.title("IST 691 | PyTorch Bird Audio Classifier 🐦")
    
    st.write("""
    ## Overview 🌟
    
    This application is designed to explore and analyze a dataset of bird songs from 5 different species:
    
    - Bewick's Wren
    - Northern Cardinal
    - American Robin
    - Song Sparrow
    - Northern Mockingbird
    
    The dataset was sourced from the [Kaggle](https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set/data) website, which is a community-driven platform for all things Data Science!
    
    The primary goal of this application is to develop and evaluate deep learning models for classifying bird species based on their audio recordings. By leveraging the power of PyTorch and audio processing techniques, I aim to create a system that can accurately identify bird species from short audio clips.
    
    The dataset consists of bird song recordings, where each audio file is 3 seconds long and includes a portion of the target bird's song. The recordings have been preprocessed to ensure consistent sampling rates and audio channels.
    
    Through this application, the viewer can:
    
    1. Explore the dataset and visualize the class distribution and sample audio recordings.
    2. Inspect deep learning model trainings using PyTorch for bird audio classification.
    3. Evaluate the performance of trained models using metrics such as test loss, accuracy, and confusion matrices.
    4. Experiment with different model architectures and training configurations to improve classification performance.
    
    This app was built for the IST 691 course at Syracuse University for the Master of Science in Applied Data Science program.
    """)


################################## EDA Page ##################################

if choice == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    
    # Display class distribution
    class_dist_fig = bird_dataset.plot_class_distribution()
    st.plotly_chart(class_dist_fig, use_container_width=True)

    # Display samples
    sample_data = bird_dataset.display_samples_streamlit_deployed(data_paths, num_samples=5)
    for data in sample_data:
        #st.markdown(data['label_info'])
        st.plotly_chart(data['fig'], use_container_width=True)
        st.audio(data['audio'], format='audio/wav')


################################## Model Training ##################################

if choice == "Model Training":
    st.title("Model Training")
    
    # Specify the logdir for TensorBoard and start it
    # logdir = f"{paths['runs_dir']}"
    # st_tensorboard(logdir=logdir, port=6008, width=1500)

    if 'tensorboard_url' not in st.session_state:
        logdir = f"{paths['runs_dir']}"
        st.session_state.tensorboard_url = start_tensorboard(logdir)
    
    st.components.v1.iframe(st.session_state.tensorboard_url, height=800, width=1500)  # Adjust dimensions as necessary


################################## Model Inference ##################################

if choice == "Model Inference":
    st.title("Model Inference")

    # Provide a selector for the user to choose which model's metrics to view
    model_directories = ['model_sch_lr_es_strat', 'model_sch_lr_es', 'model_fixed_lr']
    selected_model = st.selectbox('Select a model', model_directories)

    # Dynamically load and display test loss and accuracy
    metrics_path = f"{paths['results_dir']}/metrics/metrics_{selected_model}.json" 
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.write(f"Test Loss: {float(metrics['test_loss']):.4f} | Test Accuracy: {float(metrics['test_accuracy']):.2f}%")
    except FileNotFoundError:
        st.error("Model metrics file not found for the selected model. Please ensure the path is correct.")

    # Load and display the confusion matrix
    confusion_matrix_path = f"{paths['results_dir']}/confusion_matrices/confusion_matrix_{selected_model}.fig"
    try:
        with open(confusion_matrix_path, 'rb') as f:
            confusion_matrix_fig = pickle.load(f)
        st.plotly_chart(confusion_matrix_fig, use_container_width=True)
    except FileNotFoundError:
        st.error("Confusion matrix file not found for the selected model. Please ensure the path is correct.")

