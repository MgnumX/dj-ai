#!/usr/bin/env python3
"""main.py — Entry point: pipeline Random DJ Set (v3 + narrativo).

Flujo:  ANALYZER -> SET_PLANNER -> SEGMENT_CUTTER -> QUICK_MIXER
(el planner decide los roles de corte; el cutter los lee de setlist.json)

Uso:  python3 main.py [--songs ./songs] [--cache ./cache] [--output ./output]
                      [--count 30] [--seg-length 75] [--beats 4]
                      [--max-bpm 4] [--random-seed 42] [--narrative]
                      [--preview] [--keylock] [--format wav|flac]
                      [--skip-analyze] [--skip-cut] [--skip-plan] [--skip-mix]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = ["analyzer.py", "set_planner.py", "segment_cutter.py", "quick_mixer.py"]


def correr(script, args):
    print(f"\n=== {script} ===")
    proc = subprocess.run([sys.executable, script, *args])
    if proc.returncode != 0:
        raise SystemExit(f"{script} falló (código {proc.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Random DJ Set (analizar -> cortar -> planificar -> mezclar)")
    parser.add_argument("--songs", default="songs", help="carpeta con los MP3/FLAC")
    parser.add_argument("--cache", default="cache", help="carpeta de trabajo/caché")
    parser.add_argument("--output", default="output", help="carpeta del mix")
    parser.add_argument("--count", type=int, default=None,
                        help="cuántos tracks samplear para el set (default: todos)")
    parser.add_argument("--seg-length", type=float, default=75.0,
                        help="duración objetivo por segmento (segundos)")
    parser.add_argument("--beats", type=int, default=4, choices=[4, 8],
                        help="beats de crossfade (festival)")
    parser.add_argument("--max-bpm", type=float, default=4.0,
                        help="salto BPM máximo entre tracks")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="semilla del sampling aleatorio")
    parser.add_argument("--narrative", action="store_true",
                        help="modo narrativo: 3 actos + roles de corte")
    parser.add_argument("--preview", action="store_true",
                        help="generar preview.mp3 con 3 transiciones")
    parser.add_argument("--name", default=None,
                        help="nombre del set para el archivo de salida")
    parser.add_argument("--curated", default=None,
                        help="JSON curado para set_planner (order/selected)")
    parser.add_argument("--keylock", action="store_true",
                        help="pitch-shift sutil hacia key compatible")
    parser.add_argument("--format", default="wav", choices=["wav", "flac"])
    parser.add_argument("--skip-analyze", action="store_true", help="omitir el análisis")
    parser.add_argument("--skip-cut", action="store_true", help="omitir el corte de segmentos")
    parser.add_argument("--skip-plan", action="store_true", help="omitir el planeo")
    parser.add_argument("--skip-mix", action="store_true", help="omitir la mezcla")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_analyze:
        correr(SCRIPTS[0], [f"--songs={args.songs}", f"--cache={args.cache}"])
    if not args.skip_plan:
        plan_flags = [f"--cache={args.cache}", f"--max-bpm={args.max_bpm}",
                      f"--random-seed={args.random_seed}"]
        if args.count:
            plan_flags.append(f"--count={args.count}")
        if args.narrative:
            plan_flags.append("--narrative")
        if args.curated:
            plan_flags.append(f"--curated={args.curated}")
        correr(SCRIPTS[1], plan_flags)
    if not args.skip_cut:
        correr(SCRIPTS[2], [f"--songs={args.songs}", f"--cache={args.cache}",
                            f"--seg-length={args.seg_length}"])
    if not args.skip_mix:
        flags = [f"--cache={args.cache}", f"--output={args.output}",
                 f"--beats={args.beats}", f"--format={args.format}"]
        if args.preview:
            flags.append("--preview")
        if args.keylock:
            flags.append("--keylock")
        if args.name:
            flags.append(f"--name={args.name}")
        correr(SCRIPTS[3], flags)

    print("\nPipeline completo. Resumen:")
    print(f"  analysis.json: {cache_dir / 'analysis.json'}")
    print(f"  segments/    : {cache_dir / 'segments/'}")
    print(f"  setlist.json : {cache_dir / 'setlist.json'}")
    print(f"  salida       : {args.output}/" + (f"{args.name}.mp3" if args.preview
                                                else f"{args.name or 'mix_final'}.{args.format}"))


if __name__ == "__main__":
    main()
