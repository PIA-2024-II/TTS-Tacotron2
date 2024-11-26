import torch
from models.tacotron2 import Tacotron2
from models.waveglow import WaveGlow
from generate_audio import generate_audio, play_audio

PHONEME_DICT_FILE = "path/to/phoneme_dict.txt"

def load_models():
    tacotron2 = Tacotron2()
    waveglow = WaveGlow()

    tacotron2_checkpoint = torch.load("path/to/tacotron2_checkpoint.pt")
    waveglow_checkpoint = torch.load("path/to/waveglow_checkpoint.pt")

    tacotron2.load_state_dict(tacotron2_checkpoint['state_dict'])
    waveglow.load_state_dict(waveglow_checkpoint['state_dict'])

    tacotron2.eval()
    waveglow.eval()

    return tacotron2, waveglow

def create_phoneme_to_idx(phoneme_dict):
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(phoneme_dict)}
    return phoneme_to_idx

def load_phoneme_dict(file_path):
    phoneme_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            phoneme, idx = line.strip().split()
            phoneme_dict[phoneme] = int(idx)
    return phoneme_dict

def evaluate_model(tacotron2, waveglow, phoneme_dict, phoneme_to_idx):
    test_words = [
        "word1", "word2", "word3", "word4", "word5",  # From training set
        "word6", "word7", "word8", "word9", "word10",  # From validation set
        "new_word1", "new_word2", "new_word3"  # New words
    ]

    for word in test_words:
        audio = generate_audio(word, tacotron2, waveglow, phoneme_dict, phoneme_to_idx)
        play_audio(audio)

if __name__ == "__main__":
    tacotron2, waveglow = load_models()
    phoneme_dict = load_phoneme_dict(PHONEME_DICT_FILE)    
    phoneme_to_idx = create_phoneme_to_idx(phoneme_dict)
    evaluate_model(tacotron2, waveglow, phoneme_dict, phoneme_to_idx)