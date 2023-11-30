import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, butter, sosfilt

def bandpass_filter(data, sample_rate, lowcut=18, highcut=22, order=5):
    sos = butter(order, [lowcut / (0.5 * sample_rate), highcut / (0.5 * sample_rate)], btype='band', output='sos')
    return sosfilt(sos, data)

def detect_sweep_start(data, sample_rate, post_pilot_duration=40):
    # Apply bandpass filter
    filtered_data = bandpass_filter(data, sample_rate)

    # Perform STFT
    f, t, Zxx = stft(filtered_data, fs=sample_rate, nperseg=1024)

    # Find the index of the frequency closest to 20 Hz
    target_freq_index = np.argmin(np.abs(f - 20))

    # Get the magnitude of the STFT at 20 Hz
    magnitude = np.abs(Zxx[target_freq_index, :])
    threshold = np.max(magnitude) * 0.1  # Arbitrary threshold, may need adjustment

    # Detect the start of the sweep
    sweep_start_idx = np.where(magnitude > threshold)[0][0]
    sweep_start_time = t[sweep_start_idx]

    # Calculate the end time, 40 seconds after the pilot tone ends
    sweep_end_time = sweep_start_time + post_pilot_duration

    return sweep_start_time, sweep_end_time

def save_trimmed_audio(data, sample_rate, start_time, end_time, source_file, file_prefix='wavefile'):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Trim the data
    trimmed_data_left = data[start_sample:end_sample, 0]  # Left channel
    trimmed_data_right = data[start_sample:end_sample, 1]  # Right channel

    # Determine the output directory based on the source file
    output_dir = os.path.dirname(source_file) if source_file else '.'

    # Save the trimmed data as mono files
    left_output_path = os.path.join(output_dir, f"{file_prefix}1.wav")
    right_output_path = os.path.join(output_dir, f"{file_prefix}2.wav")

    wavfile.write(left_output_path, sample_rate, trimmed_data_left)
    wavfile.write(right_output_path, sample_rate, trimmed_data_right)

    return left_output_path, right_output_path

# Usage example
source_audio_file = 'TestStereoWave.wav'  # Replace with your file path
sample_rate, data = wavfile.read(source_audio_file)
mono_data = data.mean(axis=1) if data.ndim > 1 else data

sweep_start_time, sweep_end_time = detect_sweep_start(mono_data, sample_rate)
left_file, right_file = save_trimmed_audio(data, sample_rate, sweep_start_time, sweep_end_time, source_audio_file)
print(f"Trimmed audio files saved as {left_file} and {right_file}")
