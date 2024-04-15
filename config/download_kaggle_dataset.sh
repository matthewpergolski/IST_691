#!/bin/bash

# Set Kaggle Dataset Information
DATASET="vinayshanbhag/bird-song-data-set"
DESTINATION_DIR="data"
CONFIG_DIR="config"

# Current script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the Kaggle configuration directory to config/
export KAGGLE_CONFIG_DIR="${SCRIPT_DIR}/${CONFIG_DIR}"

# Download the dataset using Kaggle API
kaggle datasets download -d ${DATASET} -p ${DESTINATION_DIR}

# Extract the dataset
unzip "${DESTINATION_DIR}/${DATASET##*/}.zip" -d ${DESTINATION_DIR}

# Clean up the downloaded zip file
rm "${DESTINATION_DIR}/${DATASET##*/}.zip"
