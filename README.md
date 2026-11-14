# dj-ai

Pipeline local para analizar audio y planificar y mezclar sets de DJ. Recibe archivos que ya tengas en una carpeta; la descarga y el etiquetado externo quedan fuera de este repositorio.

Analiza BPM, tonalidad y energía, ordena los tracks por compatibilidad, corta segmentos y escribe la mezcla. Todo ocurre en tu máquina: los audios, la caché y el resultado no se suben a ningún servicio. A diferencia de un editor de DJ, no hay interfaz ni sesión: un comando deja el set en disco y `--random-seed` permite repetirlo.

## Requisitos

- Python 3.12 (la versión usada en desarrollo)
- `ffmpeg` en el `PATH` solo para `--preview`, que codifica el MP3

macOS: `brew install ffmpeg` · Debian/Ubuntu: `sudo apt install ffmpeg` · Windows: `winget install ffmpeg`

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python main.py --songs /ruta/a/tu/audio --cache ./cache --output ./output --count 20
```

El pipeline escribe `cache/analysis.json`, `cache/setlist.json`, los cortes en `cache/segments/` y la mezcla en `output/mix_final.wav`.

Opciones frecuentes:

| Flag | Para qué |
|---|---|
| `--count 20` | Muestrea 20 tracks; sin el flag usa todos |
| `--preview` | Genera `output/preview.mp3` con las primeras transiciones (requiere ffmpeg) |
| `--narrative` | Ordena el set en tres actos con roles de corte |
| `--keylock` | Ajusta el pitch hacia una tonalidad compatible |
| `--format flac` | Escribe la mezcla en FLAC en vez de WAV |

`--skip-analyze`, `--skip-plan`, `--skip-cut` y `--skip-mix` permiten reanudar desde una etapa. `python main.py --help` lista el resto.

## Ejemplo de salida

Salida capturada del planificador con metadatos sintéticos de prueba (sin archivos de audio):

```text
3 tracks ordenados -> <tmp>/setlist.json
 # Canción                                     BPM  Cam LVL       Rol
 1 Demo B.wav                                120.0   8A   2    banger
 2 Demo C.wav                                124.0   8A   4    banger
 3 Demo A.wav                                116.0   8A   1    banger
```

## Estado

El pipeline funciona de punta a punta con WAV sintéticos y tests offline. La mezcla todavía no se probó con audio real, así que los niveles y las transiciones pueden necesitar ajuste. No exporta a rekordbox, Serato ni otros formatos de biblioteca.

## Tests

```bash
pytest -q
```

## Licencia

MIT — ver [LICENSE](LICENSE).
