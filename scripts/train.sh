#!/bin/bash

# Set the PYTHONPATH to the current directory
export PYTHONPATH=$(pwd)

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

    python scripts/train.hierarchical_with_cycle.py
    if [ $? -ne 0 ]; then echo "train.hierarchical_with_cycle.py failed"; exit 1; fi

    echo "All scripts executed successfully."

else
    echo "Audio directory or files not found. Ensure the 'audio' directory exists and contains wav files."
    exit 1
fi
