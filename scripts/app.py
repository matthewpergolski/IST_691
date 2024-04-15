import streamlit as st
import pandas as pd
import numpy as np
import pickle
from bird_song_dataset import BirdSongDataset, DataPaths
import librosa
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard import program
import json
import os
import socket


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
    
    # Define a simple list of colors for the plots.
    colors = [
        "#1f77b4",  # Muted blue
        "#ff7f0e",  # Safety orange
        "#2ca02c",  # Cooked asparagus green
        "#d62728",  # Brick red
        "#9467bd",  # Muted purple
        "#8c564b",  # Chestnut brown
        "#e377c2",  # Raspberry yogurt pink
        "#7f7f7f",  # Middle gray
        "#bcbd22",  # Curry yellow-green
        "#17becf"   # Blue-teal
    ]
    
    # Function to fetch scalar data from Tensorboard event files
    def fetch_scalar_data(event_acc, tag):
        scalar_events = event_acc.Scalars(tag)
        steps, values = zip(*[(s.step, s.value) for s in scalar_events])
        return steps, values

    # Function to add traces to a Plotly figure for scalar data
    def add_scalar_trace(fig, steps, values, run_name, tag, color_index):
        fig.add_trace(
            go.Scatter(
                x=steps, 
                y=values, 
                mode='lines+markers', 
                name=f'{run_name} - {tag}', 
                line=dict(color=colors[color_index % len(colors)])  # Cycle through colors list
            )
        )

    # Function to initialize a Plotly figure
    def initialize_scalar_plot(title):
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title='Step', yaxis_title='Value')
        return fig

    # Function to plot data from all event files with color differentiation
    def load_and_plot_all_runs_data(runs_dir):
        train_fig = initialize_scalar_plot("Training Loss")
        val_fig = initialize_scalar_plot("Validation Loss")
        lr_fig = initialize_scalar_plot("Learning Rate")

        run_paths = [os.path.join(runs_dir, run) for run in os.listdir(runs_dir)]
        run_paths = [run for run in run_paths if os.path.isdir(run)]

        for idx, logdir in enumerate(run_paths):
            event_acc = EventAccumulator(logdir, size_guidance={'scalars': 0})
            event_acc.Reload()

            # Use index in colors list for the current run
            color_index = idx 

            try:
                train_steps, train_values = fetch_scalar_data(event_acc, 'Loss/train')
                add_scalar_trace(train_fig, train_steps, train_values, os.path.basename(logdir), 'Training Loss', color_index)
            except KeyError:
                st.warning(f'{os.path.basename(logdir)}: Training loss data not found.')

            try:
                val_steps, val_values = fetch_scalar_data(event_acc, 'Loss/validation')
                add_scalar_trace(val_fig, val_steps, val_values, os.path.basename(logdir), 'Validation Loss', color_index)
            except KeyError:
                st.warning(f'{os.path.basename(logdir)}: Validation loss data not found.')

            try:
                lr_steps, lr_values = fetch_scalar_data(event_acc, 'Learning Rate')
                add_scalar_trace(lr_fig, lr_steps, lr_values, os.path.basename(logdir), 'Learning Rate', color_index)
            except KeyError:
                # Plotting default learning rate if not found
                default_lr = [0.001] * len(train_steps)  # Match the number of training steps
                add_scalar_trace(lr_fig, train_steps, default_lr, os.path.basename(logdir), 'Learning Rate (default)', color_index)

        st.plotly_chart(train_fig, use_container_width=True)
        st.plotly_chart(val_fig, use_container_width=True)
        st.plotly_chart(lr_fig, use_container_width=True)

    load_and_plot_all_runs_data(paths['runs_dir'])

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

