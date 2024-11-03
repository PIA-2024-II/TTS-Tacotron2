import torchaudio

# Cambiar el backend
torchaudio.set_audio_backend("sox_io")  # Puedes probar con "soundfile" también

# Ruta del archivo de audio
audio_file_path = "C:/Users/sonis/Desktop/ScriptTacotron-main/data/Mpg_Chl_Rgn9_Puerto_Saavedra_3139_pisar.wav"

# Cargar el archivo de audio
waveform, sample_rate = torchaudio.load(audio_file_path)

# Imprimir las dimensiones del audio
print("Forma del waveform:", waveform.shape)
print("Frecuencia de muestreo:", sample_rate)
