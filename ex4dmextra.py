
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from PIL import Image
import matplotlib.pyplot as plt

import librosa
import librosa.display

import tkinter as tk
from tkinter import filedialog

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

root = tk.Tk()
root.withdraw()

print("="*60)
print("PART A: TEXT PREPROCESSING")
print("="*60)

sample_text = """Lorem Ipsum is simply dummy text of the printing and
typesetting industry. Lorem Ipsum has been the industry's standard dummy text
ever since 1966, when designers at Letraset and James Mosley, the librarian at
St Bride Printing Library in London, took a 1914 Cicero translation
and scrambled it to make dummy text for Letraset's Body Type sheets. """

print("\nOriginal Text:\n", sample_text)

text = sample_text.lower()

text = re.sub(r'http\S+|www\S+', '', text)

text = re.sub(r'\d+', '', text)

text = text.translate(str.maketrans('', '', string.punctuation))

text = re.sub(r'\s+', ' ', text).strip()
print("\nCleaned Text:\n", text)

tokens = word_tokenize(text)
print("\nTokens:\n", tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w not in stop_words]
print("\nAfter Stopword Removal:\n", filtered_tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
print("\nLemmatized Tokens:\n", lemmatized_tokens)

processed_text = ' '.join(lemmatized_tokens)
print("\nFinal Processed Text:\n", processed_text)


print("\n" + "="*60)
print("PART B: IMAGE PREPROCESSING")
print("="*60)

print("\nSelect an image file (jpg/png)...")
img_path = filedialog.askopenfilename(
    title="Select Image File",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if img_path:
    img = Image.open(img_path)
    print("\nOriginal Image Size:", img.size, "| Mode:", img.mode)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    img_resized = img.resize((128, 128))

    img_gray = img_resized.convert('L')

    img_array = np.array(img_gray)
    print("\nImage Array Shape:", img_array.shape)
    print("Pixel Value Range Before Normalization:", img_array.min(), "-", img_array.max())

    img_normalized = img_array / 255.0
    print("Pixel Value Range After Normalization:", img_normalized.min(), "-", img_normalized.max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_resized)
    axes[0].set_title("Resized (128x128)")
    axes[0].axis('off')
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title("Grayscale + Normalized")
    axes[1].axis('off')
    plt.show()
else:
    print("No image selected. Skipping image preprocessing.")


print("\n" + "="*60)
print("PART C: AUDIO PREPROCESSING")
print("="*60)

print("\nSelect an audio file (wav/mp3)...")
audio_path = filedialog.askopenfilename(
    title="Select Audio File",
    filetypes=[("Audio files", "*.wav *.mp3")]
)

if audio_path:
    signal, sr = librosa.load(audio_path, sr=None)
    print("\nSample Rate:", sr, "| Duration (s):", len(signal) / sr)

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(signal, sr=sr)
    plt.title("Original Waveform")
    plt.show()

    target_sr = 16000
    signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
    print("Resampled Sample Rate:", target_sr)

    signal_normalized = signal_resampled / np.max(np.abs(signal_resampled))

    signal_trimmed, _ = librosa.effects.trim(signal_normalized)
    print("Length Before Trimming:", len(signal_normalized), "| After Trimming:", len(signal_trimmed))

    mfccs = librosa.feature.mfcc(y=signal_trimmed, sr=target_sr, n_mfcc=13)
    print("MFCC Feature Shape:", mfccs.shape)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=target_sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCC Features")
    plt.show()
else:
    print("No audio selected. Skipping audio preprocessing.")

print("\nText, Image, and Audio preprocessing completed successfully!")
