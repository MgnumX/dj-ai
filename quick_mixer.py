#!/usr/bin/env python3
"""quick_mixer.py — Transiciones rápidas festival sobre los segmentos (v3).

Lee cache/setlist.json + cache/segments.json y encadena los segmentos con:

  1. AUTO-GAIN: cada segmento se normaliza a -14 LUFS (medido en analyzer).
  2. Crossfade corto de `--beats` beats (4 u 8) estilo festival: el outgoing
     baja sus graves (low-shelf a -12 dB) mientras el incoming los sube,
     con dip de medios en el centro — filtros ESTÁTICOS con rampas de
     ganancia (reutiliza ecualizar() de mixer.py, sin Butterworth variables).
  3. Beat-matching por grid de downbeats (rate de intervalos de barra).
  4. Los drops caen en beat 1 de frase por construcción del segment_cutter
     (cada segmento arranca en el build-up, drop a 2-4 barras del inicio).
  5. --keylock opcional: pitch-shift sutil (<= 3 semitonos) del incoming
     hacia una key compatible cuando el BPM no lo requiere.
  6. Micro-fades anti-clic y headroom final.

Uso:  python3 quick_mixer.py [--cache ./cache] [--output ./output]
                             [--beats 4|8] [--keylock] [--format wav|flac]
                             [--preview]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

SEAM_MS = 10
CORTE_BAJO, CORTE_ALTO = 250.0, 4000.0
TARGET_LUFS = -14.0
GAIN_MAX_DB = 8.0
SR_SALIDA = 44100     # sr canónico del mix (la librería es 44.1k-192k)
PREVIEW_SEG_S = 12.0  # largo de cada segmento en el preview (~35 s total)
PREVIEW_MAX_S = 40.0


def cargar_segmento(meta, sr_objetivo):
    """Carga un segmento (estéreo), resampleado al sr objetivo y con auto-gain."""
    y, sr = librosa.load(meta["segment_file"], sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    if sr_objetivo is not None and sr != sr_objetivo:
        y = np.stack([librosa.resample(ch, orig_sr=sr, target_sr=sr_objetivo)
                      for ch in y])
        sr = sr_objetivo
    if meta.get("lufs") is not None:
        ganancia = float(np.clip(TARGET_LUFS - meta["lufs"], -GAIN_MAX_DB, GAIN_MAX_DB))
        y = y * (10 ** (ganancia / 20))
    dbs = np.array(meta["downbeats_rel"], dtype=float)
    return y, sr, dbs


def _num_camelot(pc, menor):
    """Número de rueda Camelot (1-12) de un pitch class 0-11 y su modo."""
    if menor:
        pc = (pc + 3) % 12  # menor relativa
    n = (7 * pc + 8) % 12
    return 12 if n == 0 else n


def key_compatible_shift(cam_out, cam_in):
    """Pitch-shift (semitono) que hace compatible cam_in con cam_out.

    Devuelve 0 si ya son compatibles, el desplazamiento mas chico (<= 3
    semitonos) que los vuelve compatibles, o None si no hay ninguno.
    """
    n_out = int(cam_out[:-1])
    menor = cam_in.endswith("A")
    pc = ((int(cam_in[:-1]) - 8) * 7) % 12
    if menor:
        pc = (pc - 3) % 12  # pitch class de la menor
    for shift in (0, 1, -1, 2, -2, 3, -3):
        n = _num_camelot((pc + shift) % 12, menor)
        dif = min(abs(n - n_out), 12 - abs(n - n_out))
        if dif <= 1:
            return shift
    return None


def transicion(master, slave, sr, beats):
    """Encadena master -> slave y devuelve (mezcla, offset).

    master/slave = (y, sr, downbeats_rel). 'offset' desplaza los downbeats
    del slave dentro de la mezcla resultante (para encadenar). 'info' trae
    la geometría de la transición para el log.
    """
    y_m, _, db_m = master
    y_s, _, db_s = slave
    barras = max(1, beats // 4)  # 4 beats = 1 barra, 8 beats = 2

    # Outgoing: últimas `barras` barras del segmento (termina en downbeat).
    tail_start = len(db_m) - barras - 1
    inicio_tail = int(db_m[tail_start] * sr)
    fin_tail = int(db_m[-1] * sr)
    tail = y_m[:, inicio_tail:fin_tail]

    # Incoming: primeras `barras` barras; el resto sigue después del cruce.
    fin_head = int(db_s[barras] * sr)
    head = y_s[:, :fin_head]
    cola = y_s[:, fin_head:]

    # Beat-matching por grid (rate de intervalos de barra reales).
    barra_m = np.median(np.diff(db_m[tail_start:]))
    barra_s = np.median(np.diff(db_s[:barras + 1]))
    rate = barra_s / barra_m
    head = np.stack([librosa.effects.time_stretch(ch, rate=rate) for ch in head])

    # Alinear longitudes y cruzar con fade equal-power.
    largo = tail.shape[1]
    head = np.stack([np.interp(np.linspace(0, 1, largo),
                               np.linspace(0, 1, head.shape[1]), ch)
                     for ch in head])
    t = np.linspace(0, 1, largo)
    eq_m, eq_s = ecualizar(tail, head, sr, t)
    fade_m = np.cos(0.5 * np.pi * t)
    fade_s = np.sin(0.5 * np.pi * t)
    cruce = eq_m * fade_m + eq_s * fade_s

    cruce = micro_fades(cruce, sr)
    cola = micro_fades(cola, sr, al_inicio=False)

    mezcla = np.concatenate([y_m[:, :inicio_tail], cruce, cola], axis=1)
    offset = (inicio_tail + largo) / sr - db_s[barras]
    info = {"tail_bars": barras, "rate": round(float(rate), 3),
            "cruce_s": round(largo / sr, 2)}
    return mezcla, offset, info


def ecualizar(tail, head, sr, t):
    """EQ estático de 3 bandas con rampas de ganancia (ver mixer.py).

    Outgoing: graves/agudos 0 -> -12 dB. Incoming: -12 -> 0 dB. Medios -3 dB.
    """
    sos_bajo = butter(2, CORTE_BAJO / (sr / 2), "lowpass", output="sos")
    sos_medio = butter(2, np.array([CORTE_BAJO, CORTE_ALTO]) / (sr / 2),
                       "bandpass", output="sos")
    sos_alto = butter(2, CORTE_ALTO / (sr / 2), "highpass", output="sos")
    g_bajo_m = 10 ** ((-12 * t) / 20)
    g_bajo_s = 10 ** ((-12 * (1 - t)) / 20)
    g_medio = 10 ** (-3 / 20)

    def ec(q, g_bajo):
        canales = []
        for ch in q:
            lo = sosfiltfilt(sos_bajo, ch)
            me = sosfiltfilt(sos_medio, ch)
            hi = sosfiltfilt(sos_alto, ch)
            canales.append(lo * g_bajo + me * g_medio + hi)
        return np.stack(canales)

    return ec(tail, g_bajo_m), ec(head, g_bajo_s)


def micro_fades(y, sr, al_inicio=True):
    n = int(SEAM_MS * sr / 1000)
    if y.shape[1] <= 2 * n:
        return y
    rampa = np.linspace(0, 1, n)
    if not al_inicio:
        rampa = rampa[::-1]
    y = y.copy()
    if al_inicio:
        y[:, :n] *= rampa
    else:
        y[:, -n:] *= rampa
    return y


def main():
    parser = argparse.ArgumentParser(description="Mix rápido festival sobre segmentos")
    parser.add_argument("--cache", default="cache", help="carpeta con setlist/segments")
    parser.add_argument("--output", default="output", help="carpeta de salida")
    parser.add_argument("--beats", type=int, default=4, choices=[4, 8],
                        help="beats de crossfade (festival)")
    parser.add_argument("--keylock", action="store_true",
                        help="pitch-shift sutil hacia key compatible (opt-in)")
    parser.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"])
    parser.add_argument("--preview", action="store_true",
                        help="genera preview.mp3 con 3 transiciones de muestra")
    parser.add_argument("--name", default=None,
                        help="nombre del set para la salida (default: mix_final)")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    setlist = json.loads((cache_dir / "setlist.json").read_text())
    segs = json.loads((cache_dir / "segments.json").read_text())["segments"]
    orden = setlist["order"]
    metas = [segs[nombre] for nombre in orden if nombre in segs]
    faltantes = [n for n in orden if n not in segs]
    if faltantes:
        raise SystemExit(f"sin segmentos para: {faltantes[:5]}... (corré segment_cutter.py)")

    if args.preview:
        metas = [dict(m) for m in metas[:4]]  # 4 segmentos -> 3 transiciones
        for m in metas:
            dbs = np.array(m["downbeats_rel"])
            m["downbeats_rel"] = [float(d) for d in dbs[dbs < PREVIEW_SEG_S]]
            m["segment_file"] = m["segment_file"]  # se recorta al cargar
        # recorte: se hace post-carga con el flag preview

    # Primer segmento define el sample rate (canónico, la librería mezcla
    # FLAC de 44.1k a 192k: el resto se resamplea a este).
    y, sr, dbs = cargar_segmento(metas[0], SR_SALIDA)
    if args.preview:
        y = y[:, :int(PREVIEW_SEG_S * sr)]
    mix = y
    master = (mix, sr, dbs)

    infos = []
    for i, m in enumerate(tqdm(metas[1:], desc="Transiciones festival", unit="transición")):
        siguiente = cargar_segmento(m, sr)
        if args.preview:
            siguiente = (siguiente[0][:, :int(PREVIEW_SEG_S * sr)],
                         siguiente[1],
                         siguiente[2][siguiente[2] < PREVIEW_SEG_S])
        if args.keylock:
            # Pitch-shift sutil del incoming completo hacia key compatible.
            shift = key_compatible_shift(metas[i]["camelot"], m["camelot"])
            if shift:
                siguiente = (np.stack([librosa.effects.pitch_shift(ch, sr=sr,
                                                                   n_steps=shift)
                                       for ch in siguiente[0]]),
                             siguiente[1], siguiente[2])
        mix, offset, info = transicion(master, siguiente, sr, args.beats)
        infos.append(info)
        master = (mix, sr, siguiente[2] + offset)

    pico = np.abs(mix).max()
    if pico > 0.95:
        mix = mix * (0.95 / pico)

    if args.preview:
        mix = mix[:, :int(PREVIEW_MAX_S * sr)]
        tmp = out_dir / "_preview_tmp.wav"
        sf.write(tmp, mix.T, sr, format="WAV")
        base = args.name or "preview"
        salida = out_dir / f"{base}.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(salida)],
                       check=True)
        tmp.unlink()
        etiqueta = salida
    else:
        base = args.name or "mix_final"
        salida = out_dir / f"{base}.{args.format}"
        sf.write(salida, mix.T, sr,
                 format="FLAC" if args.format == "flac" else "WAV")

    print(f"\n{len(infos)} transiciones ({args.beats} beats) | "
          f"rates: {min(i['rate'] for i in infos):.3f}-{max(i['rate'] for i in infos):.3f} "
          f"| cruces: {min(i['cruce_s'] for i in infos):.1f}-{max(i['cruce_s'] for i in infos):.1f} s")
    duracion = mix.shape[1] / sr
    print(f"Salida: {salida} ({duracion / 60:.1f} min, sr {sr})")


if __name__ == "__main__":
    main()
