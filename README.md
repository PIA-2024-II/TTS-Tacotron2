# TTS Microservice

Este microservicio convierte texto a audio utilizando GTTS. Para entrenar es necesario tener audios Wav dentro de data mas el diccionario.
Link a los audios "https://drive.google.com/file/d/1rQUyL0D-Ejef4_-aboUDKbe2hJtVw3IS/view?usp=drive_link"

## Crear entorno virtual, para crear uno nuevo:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Preparación de entorno

```bash
bash setup_environment.sh
```

## Preparación del dataset

```bash
python process_transcripts.py
```

## Entrenar
Ejecuta el servicio con:

```bash
python train.py
```
## Generar audio
Ejecuta el servicio con:

```bash
python generate_audio.py
```