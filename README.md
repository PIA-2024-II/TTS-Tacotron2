# TTS Microservice

Este microservicio convierte texto a audio utilizando GTTS.

Para entrenar es necesario tener audios Wav dentro de data mas el diccionario.

## Crear entorno virtual, para crear uno nuevo:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Para entrenar
Ejecuta el servicio con:

```bash
python train.py
```
## Para generar audio
Ejecuta el servicio con:

```bash
python generate_audio.py
```#   T T S - T a c o t r o n 2  
 