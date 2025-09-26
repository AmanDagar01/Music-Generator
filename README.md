# Music Generator using RNN

A deep learning model that generates music using Recurrent Neural Networks (RNN). The model learns patterns from MIDI files and generates new musical sequences.

## Features

- Converts MIDI files to trainable sequences
- Uses LSTM (Long Short-Term Memory) networks for music generation
- Generates music with controllable parameters
- Supports custom MIDI dataset input
- Outputs generated music as MIDI files

## Prerequisites

Install the required dependencies:

```sh
pip install -r requirements.txt
```

Required packages:
- tensorflow
- numpy
- pandas
- pretty_midi
- pyfluidsynth

## Usage

Basic usage with default parameters:

```sh
python music_generator.py
```

### Advanced Options

```sh
python music_generator.py \
    --midi-dir "midi_dataset" \
    --output-file "output/generated_music.mid" \
    --epochs 10 \
    --num-predictions 1200 \
    --temperature 2.0
```

### Parameters

- `--midi-dir`: Directory containing training MIDI files (default: "midi_dataset")
- `--output-file`: Path to save the generated music (default: "output/generated_music.mid")
- `--epochs`: Number of training epochs (default: 10)
- `--num-predictions`: Number of notes to generate (default: 1200)
- `--temperature`: Controls randomness in generation (default: 2.0)
  - Higher values (>1) produce more random output
  - Lower values (<1) produce more focused/conservative output

## Model Architecture

The model uses:
- LSTM layer with 128 units
- Three output heads for pitch, step, and duration prediction
- Sparse Categorical Crossentropy loss for pitch
- Mean Squared Error loss for step and duration

## Project Structure

```
music_generator/
├── music_generator.py    # Main script
├── requirements.txt      # Dependencies
├── midi_dataset/        # Training data directory
└── output/              # Generated music output directory
```

## Notes

- The model expects MIDI files as input for training
- Generated music is saved in MIDI format
- Default instrument is Acoustic Grand Piano
- Training time depends on dataset size and epochs
