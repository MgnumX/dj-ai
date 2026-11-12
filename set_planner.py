#!/usr/bin/env python3
"""set_planner.py — Ordena el setlist según armonía, BPM y energía.

Lee cache/analysis.json (generado por analyzer.py) y decide el orden de
reproducción con un greedy que respeta:

  * Compatibilidad armónica: rueda Camelot — mismo número o ±1 número
    (con vuelta 12↔1), cualquier letra (A/B).
  * Rampa de BPM: salto de ≤6 BPM entre tracks consecutivos (relajado a
    ≤10 y luego sin restricción solo si no hay candidato válido).
  * Curva de energía: arranca en el tercio más bajo, sube hasta un peak
    en el 65% del set y luego desciende.

Exporta el orden final a cache/setlist.json.

Uso:  python3 set_planner.py [--cache ./cache] [--max-bpm 6] [--peak 0.65]
"""

import argparse
import json
import math
import re
from pathlib import Path

from analyzer import cargar_genres, stem_normalizado


def compatibles(c1, c2):
    """True si dos códigos Camelot ('8A', '9B') son compatibles (mismo
    número o ±1 en la rueda, que es circular: 12 es vecino de 1)."""
    n1, n2 = int(c1[:-1]), int(c2[:-1])
    dif = min(abs(n1 - n2), 12 - abs(n1 - n2))
    return dif <= 1


# Rango de normalización de las features de energía (ver analyzer.py) y
# umbral de confianza de key bajo el cual la armonía no es fiable.
MAX_RMS, MAX_TRANS, MAX_BANDA = 0.3, 250.0, 0.15
PESO_RMS, PESO_TRANS, PESO_BANDA = 0.4, 0.35, 0.25
UMBRAL_CONFIANZA = 0.45


def score_energia(t):
    """Score 0-1 ponderado de las 3 features (RMS, transients, banda 2-8k)."""
    return (PESO_RMS * min(t["energy_mean"], MAX_RMS) / MAX_RMS
            + PESO_TRANS * min(t["transients_per_min"], MAX_TRANS) / MAX_TRANS
            + PESO_BANDA * min(t["mid_high_ratio"], MAX_BANDA) / MAX_BANDA)


def niveles_energia(tracks):
    """Clasifica cada track en 1-5 por quintiles del score de energía.

    Es adaptativa a la librería: los quintiles garantizan que los 5 niveles
    se repartan, como ordenan los DJ su biblioteca (relativo, no absoluto).
    """
    orden = sorted(tracks, key=score_energia)
    n = max(len(orden), 1)
    return {t["file"]: min(4, (i * 5) // n) + 1 for i, t in enumerate(orden)}


def key_riesgosa(t):
    """True si la key del track no es fiable: estimadores en desacuerdo o
    confianza de Pearson baja (armonía "no verificada")."""
    return (not t.get("key_agreement", True)
            or t.get("key_confidence", 1.0) < UMBRAL_CONFIANZA)


def objetivo_nivel(i, n, peak, narrativa=False):
    """Nivel de energía deseado para la posición i.

    Modo normal: curva 3 -> 5 -> 3 (abre fuerte, pica en `peak`, cierra).
    Modo narrativo (--narrative): 3 actos con su propia curva — apertura con
    valles suaves, peak con un breather de nivel 3, cierre emocional bajando
    a 2. La tabla se interpola al tamaño del acto (ceil(n/3)).
    """
    if not narrativa:
        pico = max(1, int(peak * n)) - 1
        if i <= pico:
            return 3.0 + 2.0 * (i / max(1, pico))
        return 5.0 - 2.0 * ((i - pico) / max(1, n - 1 - pico))

    ACTOS = ([2, 3, 4, 3, 4, 4, 4, 4, 3, 3],   # I — apertura (breather pos 8)
             [4, 5, 5, 4, 5, 5, 5, 5, 3, 5],   # II — peak (pos 8 = breather)
             [5, 4, 3, 4, 3, 2, 3, 2, 2, 2])   # III — cierre (breather pos 8)
    tam = max(1, math.ceil(n / 3))
    acto = min(i // tam, 2)
    return float(ACTOS[acto][min(i % tam, len(ACTOS[acto]) - 1)])


def asignar_roles(n, narrativa):
    """Rol narrativo de cada posición (solo con --narrative).

    Dentro de cada acto: pos 0-1 opener (más build-up), 2-7 banger (drop
    rápido), 8 breather (valle/outro, necesita acto >= 10 tracks), última
    posición closer (impacto inmediato). Sin narrative: todo banger.
    """
    if not narrativa:
        return ["banger"] * n
    tam = max(1, math.ceil(n / 3))
    roles = []
    for i in range(n):
        pos = i % tam
        if pos <= 1:
            roles.append("opener")
        elif tam >= 10 and pos == 8:
            roles.append("breather")
        elif pos == tam - 1:
            roles.append("closer")
        else:
            roles.append("banger")
    return roles


def _greedy(tracks, max_bpm, peak, por_archivo, niveles, narrativa, generos, prefijo=()):
    """Construcción inicial: abre con el track de nivel más cercano a 3
    (arranque fuerte del modo festival) y elige el candidato compatible que
    mejor se acerca al nivel objetivo de cada posición.

    Si se pasa `prefijo` (stems curados, en orden), esas posiciones quedan
    fijas y el resto se rellena desde ahí con la misma heurística."""
    n = len(tracks)
    if prefijo:
        orden = [s for s in prefijo if s in por_archivo]
    else:
        objetivo0 = objetivo_nivel(0, n, peak, narrativa)
        abridor = min(tracks, key=lambda t: abs(niveles[t["file"]] - objetivo0))["file"]
        orden = [abridor]
    restantes = set(t["file"] for t in tracks) - set(orden)

    for i in range(len(orden), n):
        actual = por_archivo[orden[-1]]
        objetivo = objetivo_nivel(i, n, peak, narrativa)

        # Escalera de relax: la armonía se exige siempre; el límite de BPM se
        # suelta solo si no queda ningún candidato compatible.
        elegido = None
        for limite in (max_bpm, 10, 1e9):
            candidatos = [
                por_archivo[nombre] for nombre in restantes
                if compatibles(actual["camelot"], por_archivo[nombre]["camelot"])
                and abs(por_archivo[nombre]["bpm"] - actual["bpm"]) <= limite
            ]
            if candidatos:
                elegido = min(candidatos,
                              key=lambda t: abs(niveles[t["file"]] - objetivo))
                break
        if elegido is None:
            # Sin candidato armónico: tomar el que menos se desvíe del nivel
            # objetivo para no quedarse bloqueado.
            elegido = por_archivo[min(
                restantes,
                key=lambda n: abs(niveles[por_archivo[n]["file"]] - objetivo))]
        orden.append(elegido["file"])
        restantes.discard(elegido["file"])

    return orden


# Pesos del pulido 2-opt: la armonía manda, después la rampa de BPM
# (el exceso sobre max_bpm se penaliza por unidad), el riesgo de key y la
# variedad (W_VARIEDAD: no repetir el mismo nivel de energía consecutivo).
# W_RISCO/W_VARIEDAD son menores que una violación armónica real.
# Modo narrativo: W_ESCALON (subir >1 nivel de golpe), W_GENERO (misma
# familia consecutiva) y W_ACTOR (carácter vs perfil del acto).
W_ARMONIA, W_BPM, W_ENERGIA, W_RISCO, W_VARIEDAD = 100.0, 8.0, 2.0, 30.0, 20.0
W_ESCALON, W_GENERO, W_ACTOR = 25.0, 18.0, 10.0


def _desajuste_acto(caracter, i, n):
    """0 si el carácter del track encaja con el acto, 1 o 2 si no.

    Acto II (peak) quiere floor/festival; actos I y III prefieren
    chill/balada (rockero es neutro en ambos)."""
    acto = i // max(1, math.ceil(n / 3))
    if acto == 1:
        return 0 if caracter in ("floor", "festival") else (1 if caracter == "rockero" else 2)
    return 0 if caracter in ("chill", "balada") else (1 if caracter == "rockero" else 2)


def _puntuar(orden, max_bpm, peak, por_archivo, niveles, narrativa, generos):
    """Score ponderado de una secuencia: menos es mejor."""
    n = len(orden)
    total = 0.0
    for i in range(n - 1):
        a, b = por_archivo[orden[i]], por_archivo[orden[i + 1]]
        if not compatibles(a["camelot"], b["camelot"]):
            total += W_ARMONIA
        if key_riesgosa(a) or key_riesgosa(b):
            total += W_RISCO
        if niveles[a["file"]] == niveles[b["file"]]:
            total += W_VARIEDAD  # variedad: no 2 niveles iguales seguidos
        total += W_BPM * max(0.0, abs(a["bpm"] - b["bpm"]) - max_bpm)
        total += W_ENERGIA * abs(
            niveles[a["file"]] - objetivo_nivel(i, n, peak, narrativa))
        if narrativa:
            # Subir >1 nivel de golpe suena a accidente (bajar 2 niveles es
            # un breakdown intencional: no se penaliza).
            subida = niveles[b["file"]] - niveles[a["file"]]
            if subida > 1:
                total += W_ESCALON * (subida - 1)
            # Misma familia de género consecutiva (variedad cultural).
            fa = generos.get(a["file"]); fb = generos.get(b["file"])
            if fa and fb and fa["family"] == fb["family"]:
                total += W_GENERO
            # Carácter del track vs perfil del acto.
            ca = generos.get(a["file"])
            if ca:
                total += W_ACTOR * _desajuste_acto(ca["character"], i, n)
    total += W_ENERGIA * abs(
        niveles[por_archivo[orden[-1]]["file"]]
        - objetivo_nivel(n - 1, n, peak, narrativa))
    return total


def _dos_opt(orden, max_bpm, peak, por_archivo, niveles, narrativa, generos, fijos=()):
    """Pulido local: invierte tramos [i, j] de la secuencia mientras el
    score baje (best-improvement hasta converger). El greedy se queda a
    veces en acantilados de tempo; 2-opt los deshace.

    La posición 0 queda fija (el arranque fuerte es parte de la curva del
    modo festival), así que i nunca es 0. Los índices en `fijos` (posiciones
    curadas por el DJ) tampoco se tocan."""
    orden = orden[:]
    fijos = set(fijos)
    mejor = _puntuar(orden, max_bpm, peak, por_archivo, niveles,
                     narrativa, generos)
    while True:
        mejora, mejor_sc = None, mejor
        for i in range(1, len(orden) - 1):
            for j in range(i + 2, len(orden)):
                if any(k in fijos for k in range(i + 1, j + 1)):
                    continue
                candidata = (orden[:i + 1] + orden[i + 1:j + 1][::-1]
                             + orden[j + 1:])
                sc = _puntuar(candidata, max_bpm, peak, por_archivo, niveles,
                              narrativa, generos)
                if sc < mejor_sc:
                    mejora, mejor_sc = candidata, sc
        if mejora is None:
            break
        orden, mejor = mejora, mejor_sc
    return orden


def ordenar(tracks, max_bpm, peak, seed, narrativa=False, generos=None,
            curated=None):
    """Greedy + 2-opt con reinicios aleatorios: devuelve el mejor orden.

    `curated` (opcional): {"order": [stems...], "selected": [stems...]}.
      * "selected" limita la librería a esos tracks.
      * "order" fija las primeras N posiciones (intención del DJ); el resto
        lo completa el motor. Un "order" completo = el set se usa tal cual.
    """
    if not tracks:
        return []
    por_archivo = {t["file"]: t for t in tracks}
    # El curated usa stems (claves de genres.json); aquí se resuelven al
    # nombre de archivo real de la librería. La clave normaliza quitando el
    # prefijo numérico y la extensión, SIN Path.stem (corta en puntos internos
    # como "feat." que sí existen en stems sin extensión).
    def _clave(st):
        s = re.sub(r"^\d+\s*-\s*", "", (st or "").strip())
        s = re.sub(r"\.(flac|mp3|wav)$", "", s, flags=re.I)
        return s.lower()
    def _resolver(stems):
        por_stem = {_clave(f): f for f in por_archivo}
        out = []
        for s in stems or []:
            f = por_stem.get(_clave(s))
            if f:
                out.append(f)
        return out
    if curated and curated.get("selected"):
        sel = set(_resolver(curated["selected"]))
        tracks = [t for t in tracks if t["file"] in sel]
        if not tracks:
            return []
    por_archivo = {t["file"]: t for t in tracks}
    niveles = niveles_energia(tracks)
    generos = generos or {}
    rng = __import__("random").Random(seed)  # reproducible
    nombres = list(por_archivo)
    objetivo0 = objetivo_nivel(0, len(tracks), peak, narrativa)
    abridor = min(tracks, key=lambda t: abs(niveles[t["file"]] - objetivo0))["file"]
    resto = [n for n in nombres if n != abridor]

    prefijo = _resolver((curated or {}).get("order"))
    fijos = set(range(len(prefijo)))

    # Todas las semillas abren con el mejor track del nivel de apertura; el
    # 2-opt deja fija esa posición. Con curación, el prefijo manda.
    semillas = [_greedy(tracks, max_bpm, peak, por_archivo, niveles,
                        narrativa, generos, prefijo=prefijo)]
    if not prefijo:
        semillas += [[abridor] + rng.sample(resto, len(resto)) for _ in range(5)]
    mejor = min(
        (_dos_opt(s, max_bpm, peak, por_archivo, niveles, narrativa, generos,
                  fijos=fijos)
         for s in semillas),
        key=lambda o: _puntuar(o, max_bpm, peak, por_archivo, niveles,
                               narrativa, generos))
    return [por_archivo[f] for f in mejor]


def cmd_catalog(cache_dir):
    """Imprime el catálogo en una línea por track, para curación por LLM.

    Formato: stem | family | character | bpm | camelot | nivel | energy | archivo
    El stem (sin prefijo numérico) es la clave que espera --curated."""
    data = json.loads((cache_dir / "analysis.json").read_text())
    tracks = list(data["songs"].values())
    niveles = niveles_energia(tracks)
    generos = cargar_genres(cache_dir)
    print("# stem | family | character | bpm | camelot | nivel | energy | cancion")
    for t in sorted(tracks, key=lambda x: x["file"]):
        stem = stem_normalizado(t["file"])
        g = generos.get(stem, {})
        print(f"{stem} | {g.get('family', '-'):<16} | {g.get('character', '-'):<8} "
              f"| {t['bpm']:>6.1f} | {t['camelot']:>4} | nv{niveles[t['file']]} "
              f"| {t['energy_mean']:.3f} | {t['file']}")


def cmd_genres(cache_dir, apply=None):
    """Sin --apply: imprime plantilla para los tracks sin entrada en
    genres.json (los rellena el LLM). Con --apply FILE: fusiona el JSON
    devuelto por el LLM en genres.json."""
    data = json.loads((cache_dir / "analysis.json").read_text())
    generos = cargar_genres(cache_dir)
    if apply:
        nuevos = json.loads(Path(apply).read_text(encoding="utf-8"))
        generos.update(nuevos)
        (cache_dir / "genres.json").write_text(
            json.dumps(generos, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"genres.json actualizado: +{len(nuevos)} entradas "
              f"({len(generos)} total).")
        return
    pendientes = [t for t in data["songs"].values()
                  if stem_normalizado(t["file"]) not in generos]
    print("# Rellena family (familia), character (floor|chill|rockero|balada|festival),")
    print("# year, bpm_web (opcional). Devuelve este JSON a --apply.")
    for t in sorted(pendientes, key=lambda x: x["file"]):
        stem = stem_normalizado(t["file"])
        print(f'  "{stem}": {{"family": "RELLENAR", "character": "RELLENAR",'
              f' "bpm_web": null, "confirmed": false}},  # {t["file"]}')


def main():
    parser = argparse.ArgumentParser(
        description="Ordena el setlist festival (armonía + BPM + curva 3-5-3 + variedad)")
    parser.add_argument("comando", nargs="?", choices=["catalog", "genres"],
                        help="catalog: imprime el catálogo; genres: rellena metadatos")
    parser.add_argument("--cache", default="cache", help="carpeta con analysis.json")
    parser.add_argument("--max-bpm", type=float, default=4.0,
                        help="salto BPM máximo entre tracks (set corto = estricto)")
    parser.add_argument("--peak", type=float, default=0.6,
                        help="fracción del set donde está el peak de energía (nivel 5)")
    parser.add_argument("--count", type=int, default=None,
                        help="cuántos tracks samplear de la librería (default: todos)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="semilla del sampling aleatorio (reproducible)")
    parser.add_argument("--narrative", action="store_true",
                        help="modo narrativo: 3 actos + roles + reglas de DJ humano")
    parser.add_argument("--curated", default=None,
                        help="JSON curado: {\"order\":[stems], \"selected\":[stems]}")
    parser.add_argument("--apply", default=None,
                        help="(con 'genres') JSON con las entradas a fusionar")
    args = parser.parse_args()

    cache_dir = Path(args.cache)

    if args.comando == "catalog":
        cmd_catalog(cache_dir)
        return
    if args.comando == "genres":
        cmd_genres(cache_dir, args.apply)
        return

    data = json.loads((cache_dir / "analysis.json").read_text())
    tracks = list(data["songs"].values())

    if args.count and args.count < len(tracks):
        rng = __import__("random").Random(args.random_seed)
        tracks = rng.sample(tracks, args.count)

    generos = {}
    if args.narrative:
        generos_crudos = cargar_genres(cache_dir)
        generos = {t["file"]: g for t in tracks
                   if (g := generos_crudos.get(stem_normalizado(t["file"])))}

    curated = None
    if args.curated:
        curated = json.loads(Path(args.curated).read_text(encoding="utf-8"))

    orden = ordenar(tracks, args.max_bpm, args.peak, args.random_seed,
                    narrativa=args.narrative, generos=generos, curated=curated)
    niveles = niveles_energia(tracks)
    roles = asignar_roles(len(orden), args.narrative)

    transiciones = []
    for prev, sig in zip(orden, orden[1:]):
        transiciones.append({
            "from": prev["file"], "to": sig["file"],
            "bpm_from": prev["bpm"], "bpm_to": sig["bpm"],
            "bpm_delta": round(abs(sig["bpm"] - prev["bpm"]), 1),
            "camelot_from": prev["camelot"], "camelot_to": sig["camelot"],
            "harmonic_compat": compatibles(prev["camelot"], sig["camelot"]),
            "key_risk": key_riesgosa(prev) or key_riesgosa(sig),
            "energy_from": prev["energy_mean"], "energy_to": sig["energy_mean"],
            "energy_level_from": niveles[prev["file"]],
            "energy_level_to": niveles[sig["file"]],
        })

    setlist = {
        "order": [t["file"] for t in orden],
        "transitions": transiciones,
        "energy_curve": [round(t["energy_mean"], 4) for t in orden],
        "energy_levels": [niveles[t["file"]] for t in orden],
        "cue_times": [t["cue_time"] for t in orden],
        "roles": roles,
    }
    out = cache_dir / "setlist.json"
    out.write_text(json.dumps(setlist, ensure_ascii=False, indent=2))

    print(f"{len(orden)} tracks ordenados -> {out}")
    print(f"{'#':>2} {'Canción':<40} {'BPM':>6} {'Cam':>4} {'LVL':>3} {'Rol':>9}")
    for i, t in enumerate(orden, 1):
        print(f"{i:>2} {t['file'][:39]:<40} {t['bpm']:>6.1f} {t['camelot']:>4} "
              f"{niveles[t['file']]:>3} {roles[i - 1]:>9}")
    violaciones = [tr for tr in transiciones if not tr["harmonic_compat"]]
    if violaciones:
        print(f"\n! {len(violaciones)} transiciones sin compatibilidad armónica "
              f"(relax de BPM activado): {[tr['from'] + ' -> ' + tr['to'] for tr in violaciones]}")
    riesgos = [tr for tr in transiciones if tr["key_risk"]]
    if riesgos:
        print(f"! {len(riesgos)} transiciones con key no fiable (penalizadas): "
              f"{[tr['from'] + ' -> ' + tr['to'] for tr in riesgos]}")


if __name__ == "__main__":
    main()
