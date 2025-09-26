import numpy as np
import tensorflow as tf
import pandas as pd
import collections
import glob
import pretty_midi
import argparse
import os

# --- Constants ---
SEQ_LENGTH = 20
VOCAB_SIZE = 128
LEARNING_RATE = 0.005
BATCH_SIZE = 64
BUFFER_SIZE = 5000
KEY_ORDER = ["pitch", "step", "duration"]

# --- Function Definitions from Notebook ---

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Converts a MIDI file to a pandas DataFrame of notes."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        if not pm.instruments:
            return pd.DataFrame() # Return empty if no instruments
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        if not sorted_notes:
            return pd.DataFrame() # Return empty if no notes
            
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes["pitch"].append(note.pitch)
            notes["start"].append(start)
            notes["end"].append(end)
            notes["step"].append(start - prev_start)
            notes["duration"].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return pd.DataFrame()


def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100) -> pretty_midi.PrettyMIDI:
    """Converts a DataFrame of notes to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

def create_sequences(train_notes, seq_length, vocab_size=128):
    """Creates training sequences and targets from notes."""
    sequences = []
    targets = []
    num_seq = train_notes.shape[0] - seq_length
    for i in range(num_seq):
        sequence = train_notes[i:i + seq_length - 1, :] / [vocab_size, 1, 1]
        target = train_notes[i + seq_length] / [vocab_size, 1, 1]  # Normalize target as well
        sequences.append(sequence)
        targets.append(target)
        
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    dataset = tf.data.Dataset.from_tensor_slices((sequences, {"pitch": targets[:, 0], "step": targets[:, 1], "duration": targets[:, 2]}))
    return dataset

def build_model(seq_length):
    """Builds the Keras model for music generation."""
    input_data = tf.keras.Input(shape=(seq_length-1, 3))
    x = tf.keras.layers.LSTM(128)(input_data)
    
    outputs = {
        "pitch": tf.keras.layers.Dense(VOCAB_SIZE, name="pitch")(x),
        "step": tf.keras.layers.Dense(1, name="step")(x),
        "duration": tf.keras.layers.Dense(1, name="duration")(x),
    }

    model = tf.keras.Model(input_data, outputs)

    loss = {
        "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        "step": tf.keras.losses.MeanSquaredError(),
        "duration": tf.keras.losses.MeanSquaredError(),
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        loss=loss,
        loss_weights={'pitch': 0.05, 'step': 1.0, 'duration': 1.0},
        optimizer=optimizer
    )
    
    model.summary()
    return model

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0):
    """Predicts the next note based on a sequence of input notes."""
    assert temperature > 0
    inputs = np.expand_dims(notes, 0)
    
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions["step"]
    duration = predictions["duration"]

    pitch_logits /= temperature
    # Note: Using tf.random.categorical for pitch prediction as it's a classification task
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    
    # Squeeze step and duration
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # Ensure step and duration are non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    
    # Denormalize the pitch
    pitch_val = int(pitch[0])
    step_val = float(step[0])
    duration_val = float(duration[0])

    return pitch_val, step_val, duration_val

# --- Main Execution Block ---

def main(args):
    """Main function to run the music generation process."""
    # 1. Load and process data
    print("Loading MIDI files...")
    all_notes = []
    filenames = glob.glob(os.path.join(args.midi_dir, '*.mid'))
    for f in filenames:
        notes = midi_to_notes(f)
        if not notes.empty:
            all_notes.append(notes)
    
    if not all_notes:
        print(f"No valid MIDI files found in {args.midi_dir}. Exiting.")
        return

    all_notes = pd.concat(all_notes)
    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)

    # 2. Create dataset
    print("Creating sequences for training...")
    seq_ds = create_sequences(train_notes, SEQ_LENGTH, VOCAB_SIZE)
    train_ds = seq_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # 3. Build and train model
    print("Building and training the model...")
    model = build_model(SEQ_LENGTH)
    model.fit(train_ds, epochs=args.epochs)
    
    # 4. Generate music
    print("Generating music...")
    sample_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)
    
    # Create the initial sequence for prediction
    input_notes = (sample_notes[:SEQ_LENGTH-1] / np.array([VOCAB_SIZE, 1, 1]))
    
    generated_notes = []
    prev_start = 0
    for _ in range(args.num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, args.temperature)
        
        # Denormalize step and duration (pitch is already an integer index)
        # Note: In create_sequences, we normalized the target. Here we must work with the model's output.
        # The model's step and duration outputs are already scaled correctly relative to pitch.
        
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        
        # Update the input sequence for the next prediction
        input_notes = np.delete(input_notes, 0, axis=0)
        # The new note needs to be normalized before appending
        normalized_note = np.array([pitch / VOCAB_SIZE, step, duration])
        input_notes = np.append(input_notes, np.expand_dims(normalized_note, 0), axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=(*KEY_ORDER, 'start', 'end'))

    # 5. Save the generated music
    print(f"Saving generated music to {args.output_file}...")
    # Use a default instrument name; Acoustic Grand Piano is program 0
    instrument_name = pretty_midi.program_to_instrument_name(0)
    notes_to_midi(generated_notes_df, out_file=args.output_file, instrument_name=instrument_name)
    print("Music generation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate music using an RNN model.")
    parser.add_argument('--midi-dir', type=str, default='midi_dataset', help='Directory containing MIDI files for training.')
    parser.add_argument('--output-file', type=str, default='output/generated_music.mid', help='Path to save the generated MIDI file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--num-predictions', type=int, default=1200, help='Number of notes to generate.')
    parser.add_argument('--temperature', type=float, default=2.0, help='Controls the randomness of predictions.')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    main(args)