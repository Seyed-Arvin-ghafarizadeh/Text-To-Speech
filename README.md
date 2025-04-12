# Text-To-Speech

**CMU Arctic TTS Dataset**
This repository contains a custom implementation of a dataset loader for the CMU Arctic corpus, specifically tailored for the bdl (male) speaker. The CMU Arctic dataset provides paired audio (WAV) and text transcriptions, commonly used in text-to-speech (TTS) systems. The CMUArcticDataset class, located in the cmu_dataset.py file, reads and processes the paired audio-text data, ensuring that only valid pairs of audio and transcription are loaded. The data is retrieved from a local directory and prepared for further use in training a speech synthesis model.

The dataset class loads audio files (in WAV format) and their corresponding transcriptions from the dataset's "txt.done.data" file. It provides an easy interface to fetch the waveform data and its corresponding text transcription, with optional transformations applied to the waveform. This setup is critical for training and evaluating TTS systems using Tacotron2 or similar models. For efficient data loading, it utilizes PyTorch’s Dataset and DataLoader abstractions, ensuring smooth integration with machine learning pipelines.


**CMU Arctic TTS Dataset and Tacotron2 Model Training**
This repository provides an end-to-end implementation for training a Text-to-Speech (TTS) model using the CMU Arctic dataset and the Tacotron2 architecture. The project is divided into several parts, beginning with dataset loading and ending with model evaluation. The dataset used is the CMU Arctic dataset for the bdl speaker, which contains paired audio and text transcriptions.

Key Features:
Dataset Loader: The CMUArcticDataset class loads paired audio and text transcriptions, ensuring valid pairs and providing a seamless interface for data processing.

Train-Test Split: The dataset is split into 80% training and 20% testing data using PyTorch's random_split method. This ensures that the model is evaluated on unseen data during training.

Tacotron2 Training: The pre-trained Tacotron2 model is fine-tuned using the dataset, with mel-spectrograms used as ground-truth targets. The training loop includes model optimization using the Adam optimizer and periodic saving of model checkpoints.

Evaluation and Plotting: After each epoch, the model is evaluated on the test set, and the training and test losses are stored. Finally, the training process is visualized through a learning curve plot, showing the change in loss over epochs for both training and testing data.

The repository demonstrates the process of fine-tuning a TTS model and visualizing its performance, which can be adapted for other datasets and architectures.



**To generate speech from text using a trained Tacotron2 model**

Place your fine-tuned model checkpoint in the checkpoints directory.

Modify the input text in inference.py.

Run the script to generate the output .wav file.
