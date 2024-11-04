import os
import torch
import numpy as np
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from models.tacotron2 import Tacotron2
from models.waveglow import WaveGlow

# Configuración del entorno y parámetros
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
print(DATA_DIR)
TRANSCRIPT_FILE = os.path.join(DATA_DIR, "transcripts pc usuario.txt")
PHONEME_DICT_FILE = os.path.join(DATA_DIR, "mapudungun_fonemas_diccionario.txt")
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 16
EPOCHS = 10

# Verificación de la existencia de archivos necesarios
if not os.path.exists(TRANSCRIPT_FILE):
    print(f"Archivo no encontrado: {TRANSCRIPT_FILE}")
if not os.path.exists(PHONEME_DICT_FILE):
    print(f"Archivo no encontrado: {PHONEME_DICT_FILE}")

# Cargar diccionario de fonemas desde archivo
def load_phoneme_dict(filepath):
    phoneme_dict = {}
    if not os.path.exists(filepath):
        print(f"Error: El archivo de fonemas no existe en la ruta especificada: {filepath}")
        return phoneme_dict
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            phonemes = parts[1:]
            phoneme_dict[word] = phonemes
    return phoneme_dict

phoneme_dict = load_phoneme_dict(PHONEME_DICT_FILE)

# Verificar que el diccionario de fonemas no esté vacío
if not phoneme_dict:
    print("Error: Diccionario de fonemas vacío o no encontrado.")
else:
    print("Diccionario de fonemas cargado correctamente.")

# Convertir fonemas a índices
def create_phoneme_to_idx(phoneme_dict):
    unique_phonemes = set(phoneme for phonemes in phoneme_dict.values() for phoneme in phonemes)
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(unique_phonemes, start=1)}
    phoneme_to_idx["<pad>"] = 0
    return phoneme_to_idx

phoneme_to_idx = create_phoneme_to_idx(phoneme_dict)

# Verificar el tamaño del diccionario de fonemas
print(f"Tamaño del diccionario de fonemas: {len(phoneme_to_idx)}")

# Carga de datos
def load_data(transcript_file):
    if not os.path.exists(transcript_file):
        print(f"Error: El archivo de transcripciones no existe en la ruta especificada: {transcript_file}")
        return [], []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    audio_paths = []
    texts = []
    for line in lines:
        audio_path, text = line.strip().split('|')
        audio_paths.append(audio_path)
        texts.append(text)
    return audio_paths, texts

# Función para convertir texto a índices de fonemas
def text_to_phoneme_indices(text, phoneme_dict, phoneme_to_idx):
    phoneme_indices = []
    for word in text.lower().split():
        if word in phoneme_dict:
            phoneme_indices.extend([phoneme_to_idx.get(phoneme, 0) for phoneme in phoneme_dict[word]])
        else:
            print(f"Palabra desconocida en diccionario de fonemas: {word}")
    return phoneme_indices

# Función para hacer padding
def text_to_indices_padded(texts, phoneme_dict, phoneme_to_idx, max_length):
    text_indices = [text_to_phoneme_indices(text, phoneme_dict, phoneme_to_idx) for text in texts]
    padded_texts = [indices + [0] * (max_length - len(indices)) for indices in text_indices]
    return torch.tensor(padded_texts, dtype=torch.long)

# Definición del modelo
def initialize_models():
    tacotron2 = Tacotron2()
    tacotron2.embedding = nn.Embedding(num_embeddings=len(phoneme_to_idx), embedding_dim=512)
    waveglow = WaveGlow()
    return tacotron2, waveglow

# Función para calcular la pérdida
def compute_loss(output, target):
    return torch.nn.functional.mse_loss(output, target)

# Configurar el backend de audio a 'soundfile'
#torchaudio.set_audio_backend("soundfile")

# Crear un espectrograma mel adecuado con el número de filtros mel a 80
def create_mel_spectrogram(audio_path, n_mels=80):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convertir a mono si el audio está en estéreo
        if waveform.shape[0] == 2:  # Si el audio tiene dos canales (estéreo)
            waveform = waveform.mean(dim=0, keepdim=True)  # Promediar los canales para convertir a mono
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
        return mel_spectrogram
    except RuntimeError as e:
        print(f"Error al cargar el archivo de audio {audio_path}: {e}")
        return None

# Ajuste de canales para que coincidan con el espectrograma mel
channel_adjust = nn.Conv1d(in_channels=512, out_channels=80, kernel_size=1)

# Entrenamiento
def train(tacotron2, waveglow, audio_paths, texts):
    tacotron2.train()
    waveglow.train()
    optimizer = torch.optim.Adam(list(tacotron2.parameters()) + list(waveglow.parameters()), lr=0.001)

    for epoch in range(EPOCHS):
        for i in range(0, len(audio_paths), BATCH_SIZE):
            batch_audio_paths = audio_paths[i:i + BATCH_SIZE]
            batch_text = texts[i:i + BATCH_SIZE]
            max_length = max(len(text_to_phoneme_indices(text, phoneme_dict, phoneme_to_idx)) for text in batch_text)
            batch_text_tensor = text_to_indices_padded(batch_text, phoneme_dict, phoneme_to_idx, max_length)

            # Forward pass Tacotron2
            mel_spectrogram = tacotron2(batch_text_tensor)

            if mel_spectrogram.shape[1] != 80:
                mel_spectrogram = channel_adjust(mel_spectrogram.transpose(1, 2))

            audio = waveglow(mel_spectrogram)

            batch_audio_tensors = [create_mel_spectrogram(path, n_mels=80) for path in batch_audio_paths]
            batch_audio_tensors = [tensor for tensor in batch_audio_tensors if tensor is not None]
            if not batch_audio_tensors:
                print("No se pudo cargar ningún archivo de audio en este lote.")
                continue

            max_audio_length = max(tensor.shape[-1] for tensor in batch_audio_tensors)
            padded_audio_tensors = [F.pad(tensor, (0, max_audio_length - tensor.shape[-1])) for tensor in batch_audio_tensors]

            # Concatenar tensores con tamaños iguales
            batch_audio_tensor = torch.cat(padded_audio_tensors, dim=0)

            # Ajustar el tamaño de lote si es necesario
            if audio.shape[0] != batch_audio_tensor.shape[0]:
                print(f"Ajuste del tamaño del lote de {audio.shape[0]} a {batch_audio_tensor.shape[0]}")
                audio = F.interpolate(audio, size=(batch_audio_tensor.shape[1], batch_audio_tensor.shape[2]), mode="nearest")

            loss = compute_loss(audio, batch_audio_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                save_checkpoint(tacotron2, waveglow, epoch, i)

# Guardar checkpoints
def save_checkpoint(tacotron2, waveglow, epoch, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(tacotron2.state_dict(), os.path.join(CHECKPOINT_DIR, f'tacotron2_epoch{epoch}_step{step}.pt'))
    torch.save(waveglow.state_dict(), os.path.join(CHECKPOINT_DIR, f'waveglow_epoch{epoch}_step{step}.pt'))

# Función principal
def main():
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"Archivo no encontrado: {TRANSCRIPT_FILE}")
        return
    
    audio_paths, texts = load_data(TRANSCRIPT_FILE)
    tacotron2, waveglow = initialize_models()
    train(tacotron2, waveglow, audio_paths, texts)

if __name__ == "__main__":
    main()
