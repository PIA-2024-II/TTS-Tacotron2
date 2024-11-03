import os
import torch
import torchaudio
from models.tacotron2 import Tacotron2
from models.waveglow import WaveGlow

# Configuración
CHECKPOINT_DIR = "checkpoints"
PHONEME_DICT_FILE = "data/mapudungun_fonemas_diccionario.txt"

# Cargar diccionario de fonemas
def load_phoneme_dict(filepath):
    phoneme_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            phonemes = parts[1:]
            phoneme_dict[word] = phonemes
    return phoneme_dict

# Convertir fonemas a índices
def create_phoneme_to_idx(phoneme_dict):
    unique_phonemes = set(phoneme for phonemes in phoneme_dict.values() for phoneme in phonemes)
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(unique_phonemes, start=1)}
    phoneme_to_idx["<pad>"] = 0
    return phoneme_to_idx

# Función para convertir texto a índices de fonemas
def text_to_phoneme_indices(text, phoneme_dict, phoneme_to_idx):
    phoneme_indices = []
    for word in text.lower().split():
        if word in phoneme_dict:
            phoneme_indices.extend([phoneme_to_idx.get(phoneme, 0) for phoneme in phoneme_dict[word]])
        else:
            print(f"Palabra desconocida en diccionario de fonemas: {word}")
    return phoneme_indices

# Cargar los modelos
def load_models():
    tacotron2 = Tacotron2()
    waveglow = WaveGlow()
    
    # Cargar pesos
    tacotron2.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'tacotron2_epochX_stepY.pt')))
    waveglow.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'waveglow_epochX_stepY.pt')))
    
    tacotron2.eval()
    waveglow.eval()
    return tacotron2, waveglow

# Generar audio a partir de texto
def generate_audio(text, tacotron2, waveglow, phoneme_dict, phoneme_to_idx):
    phoneme_indices = text_to_phoneme_indices(text, phoneme_dict, phoneme_to_idx)
    input_tensor = torch.tensor(phoneme_indices).unsqueeze(0)  # Añadir dimensión de batch

    # Generar espectrograma mel
    mel_spectrogram = tacotron2(input_tensor)

    # Convertir espectrograma mel a audio
    with torch.no_grad():
        audio = waveglow(mel_spectrogram)

    return audio.squeeze(0)  # Eliminar la dimensión de batch

# Reproducir audio
def play_audio(audio, sample_rate=22050):
    torchaudio.save('output.wav', audio.unsqueeze(0), sample_rate)  # Guardar como archivo
    print("Audio guardado como output.wav")

def main():
    phoneme_dict = load_phoneme_dict(PHONEME_DICT_FILE)
    phoneme_to_idx = create_phoneme_to_idx(phoneme_dict)  # Ahora la función está definida

    tacotron2, waveglow = load_models()
    
    text = "Tu texto aquí"  # Cambia esto al texto que deseas generar
    audio = generate_audio(text, tacotron2, waveglow, phoneme_dict, phoneme_to_idx)

    play_audio(audio)

if __name__ == "__main__":
    main()