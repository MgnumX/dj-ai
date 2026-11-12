#!/usr/bin/env python3
"""segment_cutter.py — Corta cada canción en un segmento (v3 + roles).

Estilo festival drop-to-drop con rol narrativo leído de setlist.json
(decidido por set_planner --narrative):

  * opener  -> drop - 16 beats (más build-up, tensión)
  * banger  -> drop - 8 beats  (el corte clásico)
  * breather-> breakdown_time (valle post-drop; fallback: outro - largo)
  * closer  -> drop - 4 beats  (impacto inmediato)

Sin --narrative todos los roles son "banger" (comportamiento v3 exacto).

Exporta cache/segments/{nombre}_cut.wav y cache/segments.json con los
downbeats relativos y metadatos por segmento (para quick_mixer).

Uso:  python3 segment_cutter.py [--songs ./songs] [--cache ./cache]
                                [--seg-length 75] [--build-bars 2|4]
                                [--test] [--limit N]
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

SEG_MIN = 60.0  # bajo este largo el segmento se usa completo con warning

# Build-up (downbeats = barras) según rol: 4/2/1 barras antes del drop.
PREROLL = {"opener": 4, "banger": 2, "closer": 1}


def inicio_por_rol(analisis, rol):
    """Índice de downbeat donde arranca el segmento según el rol."""
    cue = int(analisis["cue_idx"])
    if rol in PREROLL:
        return max(0, cue - PREROLL[rol])
    # breather: el valle post-drop (o fallback al final del cuerpo).
    breakdown = analisis.get("breakdown_time")
    outro = float(analisis["outro_time"])
    downbeats = np.array(analisis["downbeats"], dtype=float)
    if breakdown is not None and breakdown < outro - 4.0:
        idx = int(np.searchsorted(downbeats, breakdown, "right") - 1)
        return max(0, idx)
    # Fallback: apuntar al final del cuerpo (outro - build-up estándar).
    idx = int(np.searchsorted(downbeats, outro - 8.0, "right") - 1)
    return max(0, idx)


def cortar_segmento(path, analisis, seg_length, rol):
    """Corta un track en su segmento festival.

    Devuelve (segmento, sr, metadatos) o (None, None, {"error": ...}).
    """
    downbeats = np.array(analisis["downbeats"], dtype=float)
    cue_time = float(analisis["cue_time"])
    outro = float(analisis["outro_time"])

    # INICIO por rol, por INDICE de downbeat (grid-alineado, sin drift).
    start_idx = inicio_por_rol(analisis, rol)
    start = float(downbeats[start_idx])
    # Los breathers deben durar: si el valle está pegado al outro, mover el
    # inicio hacia atrás para que el segmento respire completo (termina en
    # el decay natural del track).
    if rol == "breather" and outro - start < seg_length:
        limite = max(0.0, outro - seg_length)
        idx = int(np.searchsorted(downbeats, limite, "right") - 1)
        start = float(downbeats[max(0, idx)])

    # FIN: min(outro, start + seg_length), snap-eado hacia abajo al grid.
    fin = min(outro, start + seg_length)
    fin = max(fin, start + 4.0)               # piso de 4 s contra degenerados
    idx_fin = np.nonzero(downbeats <= fin)[0]
    fin = float(downbeats[idx_fin[-1]]) if len(idx_fin) else fin

    y, sr = librosa.load(str(path), sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    seg = y[:, int(start * sr):int(fin * sr)]
    if seg.shape[1] < int(SEG_MIN * sr) and (analisis["duration"] - start) < SEG_MIN:
        warning = "track demasiado corto: segmento incompleto"
    else:
        warning = None

    dbs_rel = downbeats[(downbeats >= start) & (downbeats <= fin + 0.01)] - start
    metas = {
        "segment_file": None,  # se completa en main()
        "start_sec": round(start, 3),
        "end_sec": round(fin, 3),
        "duration": round(float(fin - start), 3),
        "drop_offset_sec": round(cue_time - start, 3),
        "downbeats_rel": [round(float(d), 4) for d in dbs_rel],
        "role": rol,
        "bpm": analisis["bpm"],
        "camelot": analisis["camelot"],
        "energy_mean": analisis["energy_mean"],
        "transients_per_min": analisis["transients_per_min"],
        "mid_high_ratio": analisis["mid_high_ratio"],
        "lufs": analisis["lufs"],
        "warning": warning,
    }
    return seg, sr, metas


def main():
    parser = argparse.ArgumentParser(description="Corta cada canción en un segmento festival")
    parser.add_argument("--songs", default="songs", help="carpeta con el audio")
    parser.add_argument("--cache", default="cache", help="carpeta con analysis/setlist")
    parser.add_argument("--seg-length", type=float, default=75.0,
                        help="duración objetivo del segmento en segundos")
    parser.add_argument("--test", action="store_true",
                        help="cortar solo la primera canción e imprimir detalles")
    parser.add_argument("--limit", type=int, default=None,
                        help="cortar solo las primeras N canciones")
    args = parser.parse_args()

    songs_dir = Path(args.songs)
    cache_dir = Path(args.cache)
    seg_dir = cache_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    datos = json.loads((cache_dir / "analysis.json").read_text())
    # Roles decididos por set_planner (--narrative); sin él, todos "banger".
    setlist = json.loads((cache_dir / "setlist.json").read_text())
    roles = dict(zip(setlist["order"], setlist.get("roles", ["banger"] * len(setlist["order"]))))

    archivos = sorted(songs_dir.glob("*"))
    archivos = [p for p in archivos if p.suffix.lower() in (".mp3", ".flac",
                                                            ".wav", ".m4a", ".ogg",
                                                            ".webm", ".opus")]
    if args.limit:
        archivos = archivos[:args.limit]
    if args.test:
        archivos = archivos[:1]

    resultado = {}
    warnings = []
    for path in tqdm(archivos, desc="Cortando segmentos", unit="track"):
        analisis = datos["songs"].get(path.name)
        if analisis is None:
            warnings.append({"file": path.name, "error": "sin análisis previo"})
            continue
        try:
            seg, sr, metas = cortar_segmento(path, analisis, args.seg_length,
                                             roles.get(path.name, "banger"))
            salida = seg_dir / f"{path.stem}_cut.wav"
            sf.write(salida, seg.T, sr, format="WAV")
            metas["segment_file"] = str(salida)
            resultado[path.name] = metas
            if metas["warning"]:
                warnings.append({"file": path.name, "error": metas["warning"]})
        except Exception as e:
            warnings.append({"file": path.name, "error": str(e)})
            tqdm.write(f"  ! Saltando {path.name}: {e}")

    out = cache_dir / "segments.json"
    out.write_text(json.dumps({"segments": resultado, "warnings": warnings},
                              ensure_ascii=False, indent=2))

    print(f"\n{len(resultado)} segmentos en {seg_dir} -> {out}")
    print(f"{'Canción':<34} {'dur':>6} {'drop@':>7} {'rol':>9}")
    for nombre, m in sorted(resultado.items()):
        print(f"{nombre[:33]:<34} {m['duration']:>6.1f} {m['drop_offset_sec']:>7.1f} "
              f"{m['role']:>9}")
    for w in warnings:
        print(f"  ! {w['file']}: {w['error']}")


if __name__ == "__main__":
    main()
