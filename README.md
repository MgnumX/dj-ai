# dj-ai

Pipeline local para analizar audio y planificar y mezclar sets de DJ. Recibe archivos de audio que ya tengas en una carpeta; la descarga y el etiquetado externo quedan fuera de este repositorio.

## Stack

Python, librosa, NumPy, SciPy, SoundFile y tqdm.

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python main.py --songs /ruta/a/tu/audio --cache ./cache --output ./output --count 20
```

El pipeline analiza BPM, tonalidad y energía; ordena los tracks, corta segmentos y escribe la mezcla en `output/`. Los audios, cachés y resultados se mantienen locales.

## Ejemplo de salida

Salida capturada del planificador con metadatos sintéticos de prueba (sin archivos de audio):

```text
3 tracks ordenados -> <tmp>/setlist.json
 # Canción                                     BPM  Cam LVL       Rol
 1 Demo B.wav                                120.0   8A   2    banger
 2 Demo C.wav                                124.0   8A   4    banger
 3 Demo A.wav                                116.0   8A   1    banger
```

## Tests

```bash
pytest -q
```
