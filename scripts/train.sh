#!/bin/bash

# Function to prompt for and copy audio files
copy_audio_files() {
    read -p "Please specify the path to the audio database directory: " audio_db_dir
    audio_db_dir=$(echo $audio_db_dir | sed 's/^[ \t]*//;s/[ \t]*$//')  # Trim whitespace
    audio_db_dir=${audio_db_dir%/}  # Remove trailing slash if exists

    # Remove any surrounding single quotes
    audio_db_dir=$(echo $audio_db_dir | sed "s/^'//;s/'$//")

    echo "Checking audio directory: $audio_db_dir"
    if [ -d "$audio_db_dir" ]; then
        echo "Audio directory found at $audio_db_dir"
        wav_files=("$audio_db_dir"/*.wav)
        if [ ${#wav_files[@]} -gt 0 ]; then
            echo "Audio files found, copying files..."
            cp "$audio_db_dir"/*.wav audio/
            if [ $? -ne 0 ]; then
                echo "Failed to copy audio files. Please check the directory."
                exit 1
            fi
        else
            echo "No .wav files found in the directory."
            echo "Debug info:"
            ls -l "$audio_db_dir"
            exit 1
        fi
    else
        echo "Audio directory not found at '$audio_db_dir'"
        echo "Debug info:"
        ls -ld "$audio_db_dir"
        exit 1
    fi
}

# Set the PYTHONPATH to the current directory
export PYTHONPATH=$(pwd)

# Wipe the contents of the audio and training_data folders
echo "Wiping the contents of the audio and training_data folders..."
rm -rf audio/* training_data/*

# Create necessary directories if they don't exist
mkdir -p audio training_data/exp_data

# Copy audio files to the audio directory
copy_audio_files

# Prompt the user to choose the training script
echo "Choose the training script to use:"
echo "1) train.hierarchical_with_cycle.py"
echo "2) train_multiscale.hierarchical_with_cycle.py"
read -p "Enter the number of your choice: " script_choice

if [ "$script_choice" -eq 1 ]; then
    training_script="train.hierarchical_with_cycle.py"
elif [ "$script_choice" -eq 2 ]; then
    training_script="train_multiscale.hierarchical_with_cycle.py"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Ensure the audio directory exists and contains wav files
if [ -d "audio" ] && [ "$(ls -A audio/*.wav 2>/dev/null)" ]; then
    echo "Audio directory and files found, proceeding with scripts..."

    # Run the scripts sequentially
    python scripts/collect_audio_clips.py --audio-dir "audio" --extension wav
    if [ $? -ne 0 ]; then echo "collect_audio_clips.py failed"; exit 1; fi

    python scripts/extract_mel.py
    if [ $? -ne 0 ]; then echo "extract_mel.py failed"; exit 1; fi

    python scripts/make_dataset.py
    if [ $? -ne 0 ]; then echo "make_dataset.py failed"; exit 1; fi

    python scripts/compute_mean_std.mel.py
    if [ $? -ne 0 ]; then echo "compute_mean_std.mel.py failed"; exit 1; fi

    python scripts/$training_script
    if [ $? -ne 0 ]; then echo "$training_script failed"; exit 1; fi

    echo "All scripts executed successfully."

else
    echo "Audio directory or files not found. Ensure the 'audio' directory exists and contains wav files."
    echo "Debug info:"
    ls -ld audio
    ls -l audio
    exit 1
fi
