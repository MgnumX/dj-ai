#!/usr/bin/env python3
"""grafico.py — Visualiza el setlist en ASCII (cero dependencias).

Imprime en terminal la curva de energía (niveles 1-5) y el mapa de BPM del
set, con las transiciones no armónicas marcadas con 'x'.

Uso:  python3 grafico.py [--cache ./cache]
"""

import argparse
import json
from pathlib import Path

ANCHO = 60  # columnas del gráfico


def grafico(serie, minimo, maximo, etiqueta, marcas=None):
    """Dibuja una serie como barras verticales ASCII."""
    marcas = marcas or {}
    print(f"{etiqueta}   ^")
    for nivel in range(maximo, minimo - 1, -1):
        fila = f"{nivel:>3}   |"
        for i, v in enumerate(serie):
            if v >= nivel:
                fila += "####"
            elif i in marcas:
                fila += " xx "
            else:
                fila += "    "
        print(fila)
    eje = " " * 7 + "+" + "-" * (ANCHO * 4)
    print(eje)
    print(" " * 7 + "".join(f"T{str((i % 30) + 1):>4}" for i in range(ANCHO // 5)))


def main():
    parser = argparse.ArgumentParser(description="Gráfico ASCII del setlist")
    parser.add_argument("--cache", default="cache", help="carpeta con setlist.json")
    args = parser.parse_args()

    sl = json.loads((Path(args.cache) / "setlist.json").read_text())
    niveles = sl["energy_levels"]
    transiciones = sl["transitions"]
    no_arm = {i for i, t in enumerate(transiciones) if not t["harmonic_compat"]}
    bpm_serie = ([transiciones[0]["bpm_from"]] if transiciones else [])
    for t in transiciones:
        bpm_serie.append(t["bpm_to"])

    print("\n=== CURVA DE ENERGÍA (niveles 1-5) ===")
    grafico(niveles, 1, 5, "LVL", marcas=no_arm)
    print("   x = transición sin compatibilidad armónica\n")

    print("=== MAPA BPM (deltas acumulados, aprox.) ===")
    grafico(bpm_serie, int(min(bpm_serie)) // 20 * 20,
            max(60, int(max(bpm_serie)) // 20 * 20 + 20), "BPM")


if __name__ == "__main__":
    main()
