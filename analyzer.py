#!/usr/bin/env python3
"""analyzer.py — Análisis musical local de cada canción en songs/ (v3).

Extrae por track: BPM, key (Camelot) con doble verificación, energía
(RMS + transients + banda 2-8 kHz), loudness integrador (LUFS, BS.1770),
duración, downbeats, frases de 8 barras, el cue point del primer beat drop
(híbrido: onset flux + RMS) y el outro (fin del cuerpo de la canción).

El resultado se guarda incrementalmente en cache/analysis.json: al volver a
ejecutar, las canciones ya analizadas se omiten (el campo "version" invalida
el caché cuando cambia el schema). Si un MP3/FLAC está corrupto, se registra
en "warnings" y se salta.

Uso:  python3 analyzer.py [--songs ./songs] [--cache ./cache] [--limit N]
                          [--test-drop CANCION]
"""

import argparse
import json
import os
import re
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, lfilter, sosfilt
from tqdm import tqdm

HOP = 512            # frames de audio entre ventanas de análisis
SR_ANALISIS = 22050  # resample para acelerar el análisis (BPM/key no cambian)
VERSION_CACHE = 6    # bump al cambiar el schema de analysis.json

# Perfiles de Krumhansl-Schmuckler (croma de una tonalidad típica) para
# estimar la key por correlación sobre las 12 rotaciones de cada modo.
PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19,
                         2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75,
                         3.98, 2.69, 3.34, 3.17])
# Spelling de cada pitch class con bemoles (convención de la rueda Camelot).
NOTAS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

BARRAS_POR_FRASE = 8  # límites de frase: cada 8 barras (16/32 son múltiplos)

# Energy mapping v2: el analyzer guarda las 3 features crudas (RMS medio,
# transients/min y presencia 2-8 kHz); la clasificación 1-5 por quintiles
# de la librería la hace set_planner (adaptativa, ver set_planner.py).

# Drop híbrido (v3): ventana de búsqueda 15-30 s (descarte implícito <10 s),
# umbral relativo 1.4x sobre la media de los primeros 20 s, sostenido >= 2 s.
DROP_MIN_S, DROP_MAX_S = 15.0, 30.0
DROP_UMBRAL = 1.4
DROP_SOSTENIDO_S = 2.0
PESO_ONSET, PESO_RMS_DROP = 0.6, 0.4
# Outro (v3): última ventana sostenida (>= 2 s) con energía > 50% del máximo,
# escaneada de atrás hacia adelante SOLO en la segunda mitad del track.
OUTRO_UMBRAL = 0.5
SUAVIZADO_OUTRO_S = 0.5
# Key: confianza de Pearson bajo la cual la armonía se considera no fiable.
UMBRAL_CONFIANZA = 0.45
# Breakdown (v6): valle sostenido >= 4 s con energía < 60% del pico post-drop.
BREAKDOWN_SOSTENIDO_S = 4.0
BREAKDOWN_UMBRAL = 0.6


def stem_normalizado(path):
    """Stem del filename normalizado para matchear genres.json: lowercase,
    sin prefijo numérico ("01 - "), sin extensión, espacios recortados."""
    return re.sub(r"^\d+\s*-\s*", "", Path(path).stem).lower().strip()


def cargar_genres(cache_dir):
    """Lee cache/genres.json (metadatos manuales: familia, carácter, bpm_web)."""
    p = Path(cache_dir) / "genres.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return {}


def corregir_bpm(bpm, genres, path):
    """Snap del BPM medido a un múltiplo del bpm_web (0.5x/1x/2x) si difieren
    <10%: valida el octavo que eligió el tracker, no lo reescribe. Devuelve
    (bpm_corregido, bpm_original)."""
    web = genres.get(stem_normalizado(path), {}).get("bpm_web")
    if not web:
        return round(bpm, 1), round(bpm, 1)
    k = min((0.5, 1.0, 2.0), key=lambda k: abs(bpm - k * web))
    corregido = k * web
    if abs(bpm - corregido) / corregido < 0.10:
        return round(corregido, 1), round(bpm, 1)
    return round(bpm, 1), round(bpm, 1)


def detectar_breakdown(rms, frame_sec, cue_time):
    """Primer valle de energía sostenido (>= BREAKDOWN_SOSTENIDO_S) con
    energía < BREAKDOWN_UMBRAL x pico posterior al drop. Devuelve el inicio
    del valle en segundos, o None si no existe (rock/folk no tienen
    breakdowns; el cutter usa outro como fallback)."""
    idx_cue = np.nonzero(np.arange(len(rms)) * frame_sec >= cue_time)[0]
    if len(idx_cue) < 2:
        return None
    k = max(1, int(SUAVIZADO_OUTRO_S / frame_sec))
    env = np.convolve(rms, np.ones(k) / k, "same")
    # Umbral contra la MEDIANA del tramo post-drop (no contra el pico: el
    # spike del drop vuelve "valle" a cualquier groove estable como el
    # reggae). Un breakdown real es un dip profundo vs. el nivel sostenido.
    umbral = BREAKDOWN_UMBRAL * float(np.median(env[idx_cue[0]:]))
    alto = 0.7 * float(np.median(env[idx_cue[0]:]))
    n_sost = int(BREAKDOWN_SOSTENIDO_S / frame_sec)
    n_alto = n_sost  # el tramo alto previo debe durar una frase (4 s) completa
    # El scan arranca 8 s después del drop: el aterrizaje del golpe (dip de
    # 2-3 s post-spike) no es un breakdown, y el valle anterior al drop es
    # el intro. Además, el valle debe estar precedido por un tramo sostenido
    # ALTO (>= 4 s): refuerzo contra dips de aterrizaje.
    inicio_scan = idx_cue[0] + int(8.0 / frame_sec)
    medias = np.convolve(env, np.ones(n_sost) / n_sost, "same")
    vi_alto = False
    for i in range(inicio_scan, len(env) - 1):
        if not vi_alto:
            if env[i] > alto and (i + n_alto >= len(env)
                                  or env[i:i + n_alto].all() > alto):
                vi_alto = True
        elif medias[i] < umbral:
            return float(i * frame_sec)
    return None


def camelot(pc, menor):
    """Devuelve el código Camelot (p. ej. "8A") de un pitch class 0-11.

    La menor relativa de una mayor está 3 semitonos arriba (Am ↔ C), así que
    el número de rueda de un modo menor se calcula sobre su relativa mayor.
    """
    if menor:
        pc = (pc + 3) % 12
    num = (7 * pc + 8) % 12
    if num == 0:
        num = 12
    return f"{num}{'A' if menor else 'B'}"


def estimar_key(croma):
    """Key por correlación de Pearson del croma promedio con los 24 perfiles.

    Devuelve (key, camelot, confianza) donde confianza es el coeficiente
    de correlación del mejor perfil (más alto = más clara la tonalidad).
    """
    croma_medio = croma.mean(axis=1)
    mejor = (-1.0, 0, False)
    for menor, perfil in ((False, PERFIL_MAYOR), (True, PERFIL_MENOR)):
        for rot in range(12):
            p = np.roll(perfil, rot)
            r = np.corrcoef(croma_medio, p)[0, 1]
            if r > mejor[0]:
                mejor = (r, rot, menor)
    r, pc, menor = mejor
    key = NOTAS[pc] + ("m" if menor else "")
    return key, camelot(pc, menor), round(float(r), 3)


def lufs_integrado(y, sr, ventana=0.4, hop=0.1):
    """Loudness integradora BS.1770-4 (LUFS) de una señal mono.

    K-weighting (high-pass Butterworth 2º a 38 Hz + high-shelf RBJ +4 dB a
    1.5 kHz), ventanas de 400 ms cada 100 ms, gating absoluto (-70 LUFS) y
    relativo (media - 10 LU), offset -0.691 dB. Validado contra
    ffmpeg loudnorm: error < 1 LU en canciones reales.
    """
    y = sosfilt(butter(2, 38.0 / (sr / 2), "highpass", output="sos"), y)
    db, f0 = 4.0, 1500.0
    A = 10 ** (db / 40)
    w0 = 2 * np.pi * f0 / sr
    alfa = np.sin(w0) / 2 * np.sqrt(2)  # S=1
    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alfa)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alfa)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alfa
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alfa
    y = lfilter([b0, b1, b2], [a0, a1, a2], y)

    n, h = int(ventana * sr), int(hop * sr)
    energias = np.array([np.mean(y[i:i + n] ** 2)
                         for i in range(0, len(y) - n, h)])
    db_ventana = 10 * np.log10(energias + 1e-12)
    g1 = db_ventana[db_ventana > -70.0]                      # gate absoluto
    g2 = g1[g1 > (g1.mean() - 10.0)]                         # gate relativo
    return round(10 * np.log10(np.mean(10 ** (g2 / 10))) - 0.691, 2)


def detectar_drop_hibrido(onset_env, rms, downbeats, frame_sec):
    """Primer beat drop con detector híbrido (onset flux 60% + RMS 40%).

    Baseline = media de los primeros 20 s; el drop es el primer pico de la
    ventana 15-30 s que supera 1.4x el baseline y se sostiene >= 2 s (con
    tolerancia 0.8x). Se snap-ea al siguiente downbeat >= drop.

    Devuelve (cue_idx, cue_time) o (None, None) si el track arranca caliente
    o no hay pico en la ventana (el llamador aplica fallbacks).
    """
    n = len(onset_env)
    t = np.arange(n) * frame_sec
    on = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-8)
    rm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
    comb = PESO_ONSET * on + PESO_RMS_DROP * rm

    # Arranca caliente (sin intro) si la energía de los primeros 5 s ya está
    # al nivel de los 10-15 s (>= 90%): un track con intro real tiene un
    # valle temprano (la comparación es temprano vs. medio, no vs. máximo,
    # porque hasta los tracks hot suben en el chorus).
    k = max(1, int(SUAVIZADO_OUTRO_S / frame_sec))
    env_rms = np.convolve(rms, np.ones(k) / k, "same")
    f10 = int(10 / frame_sec)
    f5, f15 = int(5 / frame_sec), int(15 / frame_sec)
    early = env_rms[:min(f5, len(env_rms))].mean()
    mid = env_rms[min(f10, len(env_rms)):min(f15, len(env_rms))].mean()
    if mid < 1e-9 or early >= 0.9 * mid:
        return 0, float(downbeats[0])

    n_sost = int(DROP_SOSTENIDO_S / frame_sec)
    baseline = comb[:int(20 / frame_sec)].mean()
    umbral = baseline * DROP_UMBRAL
    dbs = np.array(downbeats, dtype=float)
    for i in np.nonzero((t >= DROP_MIN_S) & (t <= DROP_MAX_S))[0]:
        if comb[i] > umbral:
            fin = min(i + n_sost, n)
            if comb[i:fin].mean() > umbral * 0.8:
                candidatos = np.nonzero(dbs >= float(t[i]))[0]
                if len(candidatos):
                    return int(candidatos[0]), float(dbs[candidatos[0]])
    # Sin pico claro: fallback = primer downbeat después de 20 s.
    candidatos = np.nonzero(dbs >= 20.0)[0]
    if len(candidatos):
        return int(candidatos[0]), float(dbs[candidatos[0]])
    return None, None


def detectar_outro(rms, frame_sec):
    """Fin del cuerpo de la canción: última ventana (>= 2 s) con energía
    > 50% del máximo, escaneada de atrás hacia adelante y solo en la
    segunda mitad (evita falsos positivos en breakdowns). Devuelve el
    instante del final de esa ventana, o None (llamador usa la duración).
    """
    k = max(1, int(SUAVIZADO_OUTRO_S / frame_sec))
    env = np.convolve(rms, np.ones(k) / k, "same")
    umbral = OUTRO_UMBRAL * env.max()
    n_sost = int(DROP_SOSTENIDO_S / frame_sec)
    inicio = len(env) // 2
    for i in range(len(env) - 1, inicio, -1):
        ini = max(i - n_sost, inicio)
        if env[ini:i + 1].mean() > umbral:
            return float((i + 1) * frame_sec)
    return None


def analizar_archivo(path, genres=None):
    """Analiza un archivo de audio y devuelve el dict con todos los datos."""
    genres = genres or {}
    # mono=False conserva estéreo; librosa usa soundfile (FLAC) y audioread
    # (MP3) como respaldo automáticamente.
    y, sr_orig = librosa.load(str(path), sr=None, mono=False)
    y = y.mean(axis=0) if y.ndim > 1 else y          # mezcla a mono
    duracion = len(y) / sr_orig

    # Resample para análisis (acelera beat tracking y croma ~4x).
    y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR_ANALISIS)
    sr = SR_ANALISIS
    frame_sec = HOP / sr

    # --- Ritmo: BPM, beats y downbeats (cada 4º beat, 4/4) ----------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    bpm, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP,
                                         onset_envelope=onset_env)
    bpm = float(np.ravel(bpm)[0])  # beat_track devuelve un array 1x1
    # Snap a bpm_web (genres.json) si el tracker quedó en un múltiplo cercano
    # (0.5x/1x/2x): valida la octava elegida, no la reescribe. El mixer NO
    # depende de este valor (usa el grid de downbeats).
    bpm, bpm_original = corregir_bpm(bpm, genres, path)

    # Downbeats: cada 4º beat, anclado al pico de onset más cercano (±0.25 s)
    # para absorber la "humanización" del timing.
    downbeats = []
    for t in beats[::4] * frame_sec:
        lo = max(0, int((t - 0.25) / frame_sec))
        hi = min(len(onset_env) - 1, int((t + 0.25) / frame_sec))
        ventana = onset_env[lo:hi + 1]
        downbeats.append(float((lo + int(np.argmax(ventana))) * frame_sec))

    # --- Armónía: key + Camelot con doble verificación ---------------------
    croma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, codigo_camelot, conf = estimar_key(croma)
    # Segunda estimación independiente (chroma_cens es robusta a timbre/
    # dinámica): si ambos estimadores coinciden, la key es fiable.
    croma_alt = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP)
    key_alt, _, _ = estimar_key(croma_alt)
    key_agreement = key == key_alt

    # --- Energía: RMS, transients y banda 2-8 kHz (features crudas) --------
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    tiempos = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    energia_media = float(rms.mean())

    transients = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=sr, hop_length=HOP)
    transients_min = len(transients) / max(1.0, duracion / 60.0)

    stft = np.abs(librosa.stft(y, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2 * (stft.shape[0] - 1))
    banda = (freqs >= 2000.0) & (freqs <= 8000.0)
    ratio_banda = float(stft[banda].sum() / max(stft.sum(), 1e-12))

    # --- Cue point (primer beat drop, híbrido) + outro --------------------
    cue_idx, _ = detectar_drop_hibrido(onset_env, rms, downbeats, frame_sec)
    if cue_idx is None:
        cue_idx = 0
    # Snap al límite de frase (múltiplo de 8) SOLO si no se aleja más de 4
    # barras: más de media frase y el build-up pierde el impacto real del
    # drop (estilo drop-to-drop). Fuera del cap se usa el cue crudo.
    snap = int(np.ceil(cue_idx / BARRAS_POR_FRASE)) * BARRAS_POR_FRASE
    if snap - cue_idx <= BARRAS_POR_FRASE // 2:
        cue_idx = snap
    cue_idx = min(cue_idx, max(0, len(downbeats) - 1))
    cue_time = float(downbeats[cue_idx])

    outro = detectar_outro(rms, frame_sec)
    outro_time = float(outro) if outro is not None else duracion
    outro_time = max(outro_time, cue_time)  # el outro no puede ir antes del drop

    # --- Breakdown (valle post-drop) y brillo (spectral centroid) ----------
    breakdown = detectar_breakdown(rms, frame_sec, cue_time)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    centroid_mean = float(centroid.mean() / (sr / 2))  # normalizado 0-1

    # --- Energía: media global y por frase de 8 barras ---------------------
    frases = []
    n_barras = len(downbeats) - 1
    for i in range(0, n_barras - BARRAS_POR_FRASE + 1, BARRAS_POR_FRASE):
        inicio, fin = downbeats[i], downbeats[i + BARRAS_POR_FRASE]
        mascara = (tiempos >= inicio) & (tiempos < fin)
        energia = float(rms[mascara].mean()) if mascara.any() else 0.0
        frases.append({"bar": i + 1, "start": inicio, "end": fin,
                       "energy": energia})

    # Duración media de barra (útil para el mixer).
    bar_duration = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 0.0

    return {
        "file": path.name,
        "duration": round(duracion, 3),
        "sr": sr_orig,
        "bpm": bpm,
        "bpm_original": bpm_original,
        "key": key,
        "camelot": codigo_camelot,
        "key_confidence": conf,
        "key_alt": key_alt,
        "key_agreement": key_agreement,
        "energy_mean": energia_media,
        "transients_per_min": round(transients_min, 1),
        "mid_high_ratio": round(ratio_banda, 3),
        "lufs": lufs_integrado(y, sr),
        "cue_time": round(cue_time, 3),
        "cue_idx": cue_idx,
        "outro_time": round(outro_time, 3),
        "breakdown_time": (round(breakdown, 3) if breakdown is not None
                           else None),
        "spectral_centroid_mean": round(centroid_mean, 3),
        "bar_duration": round(bar_duration, 4),
        "downbeats": downbeats,
        "phrases": frases,
    }


def cargar_cache(cache_path):
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if data.get("version") == VERSION_CACHE:
                return data
        except json.JSONDecodeError:
            pass  # caché corrupto: se regenera
    return {"version": VERSION_CACHE, "songs": {}, "warnings": []}


def main():
    parser = argparse.ArgumentParser(description="Analiza canciones para DJ (BPM, key, energía)")
    parser.add_argument("--songs", default="songs", help="carpeta con los MP3/FLAC")
    parser.add_argument("--cache", default="cache", help="carpeta de salida")
    parser.add_argument("--limit", type=int, default=None,
                        help="analizar solo las primeras N canciones")
    parser.add_argument("--test-drop", metavar="CANCION", default=None,
                        help="analiza un solo archivo e imprime drop/outro/etc")
    args = parser.parse_args()

    songs_dir = Path(args.songs)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "analysis.json"
    genres = cargar_genres(cache_dir)

    if args.test_drop:
        r = analizar_archivo(songs_dir / args.test_drop, genres)
        print(f"{args.test_drop}:")
        for k in ("bpm", "bpm_original", "key", "camelot", "key_confidence",
                  "energy_mean", "cue_time", "cue_idx", "outro_time",
                  "breakdown_time", "spectral_centroid_mean",
                  "duration", "lufs"):
            print(f"  {k:<22} {r[k]}")
        return

    data = cargar_cache(cache_file)

    archivos = sorted(songs_dir.glob("*"))
    archivos = [p for p in archivos if p.suffix.lower() in (".mp3", ".flac",
                                                            ".wav", ".m4a", ".ogg",
                                                            ".webm", ".opus")]
    if args.limit:
        archivos = archivos[:args.limit]

    for path in tqdm(archivos, desc="Analizando canciones", unit="track"):
        clave = path.name
        if clave in data["songs"]:
            continue  # ya analizada y cacheada

        try:
            data["songs"][clave] = analizar_archivo(path, genres)
        except Exception as e:  # archivo corrupto o ilegible
            data["warnings"].append({"file": path.name, "error": str(e)})
            tqdm.write(f"  ! Saltando {path.name}: {e}")

        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # Resumen
    validas = data["songs"]
    print(f"\n{len(validas)} canciones analizadas, {len(data['warnings'])} con error.")
    print(f"{'Canción':<34} {'BPM':>6} {'Key':>4} {'Cam':>4} "
          f"{'CUE(s)':>7} {'OUT(s)':>7} {'LUFS':>6}")
    for nombre, s in sorted(validas.items()):
        riesgo = "" if s["key_agreement"] else " ?"
        print(f"{nombre[:33]:<34} {s['bpm']:>6.1f} {s['key']:>4}{riesgo:>2} "
              f"{s['camelot']:>4} {s['cue_time']:>7.1f} {s['outro_time']:>7.1f} "
              f"{s['lufs']:>6.1f}")
    for w in data["warnings"]:
        print(f"  ! {w['file']}: {w['error']}")


if __name__ == "__main__":
    main()
