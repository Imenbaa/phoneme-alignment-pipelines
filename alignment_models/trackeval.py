

"""
trackeval.py — ESTER event tracking task scoring script
Traduction Python du script Perl original de Guillaume Gravier & Sylvain Galliano (2008).
bnd_f1_print
Usage:
    python trackeval.py [options] ref.etf hyp.etf

Format ETF (Event Tracking File) :
    <source> <channel> <start_time> <duration> <type> <subtype> <event> [<score> [<decision>]]
"""

import argparse
import re
import sys
import os
import io
from collections import defaultdict

# ─────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────
RELEASE = "2.4-py"
PATCH   = "0"
DATE    = "2024"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PARSING DES ARGUMENTS EN LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════════════════════
#
# Équivalent du bloc GetOptions() en Perl.
# On utilise argparse, la bibliothèque standard Python.
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Construit et retourne le parseur d'arguments.

    Options reproduites fidèlement depuis le script Perl :
      -l / --list        : fichier contenant la liste des événements
      -e / --event       : événement(s) à évaluer (peut être répété)
      -m / --margin      : tolérance temporelle (défaut 0.25 s)
      -D / --boundary_delta : tolérance de frontiére (défaut 20ms)
      -s / --uem         : fichier UEM (zones de scoring)
      -n / --max-segments: nb max de segments hyp par source/événement
      -t / --subtype     : activer rapports par sous-type
      -r / --error       : rapport de détection (sum, event, source, subtype, combinaisons avec +)
      -b / --segmentation: rapport de segmentation
      -d / --det         : préfixe fichier DET (courbes)
      -a / --align       : afficher alignement ref/hyp
      -o / --output      : fichier de sortie (défaut stdout)
      -v / --verbose     : mode verbeux
      -V / --version     : version
    """
    p = argparse.ArgumentParser(
        prog="trackeval",
        description="ESTER event tracking task scoring script.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    p.add_argument("-l", "--list",     dest="evtfn",   metavar="fn",
                   help="Charger la liste d'événements depuis un fichier.")
    p.add_argument("-e", "--event",    dest="events",  metavar="str", action="append", default=[],
                   help="Événement(s) à scorer (peut être répété ou séparé par virgule).")
    p.add_argument("-m", "--margin",   dest="margin",  metavar="f",  type=float, default=0.25,
                   help="Tolérance autour des frontières de référence (défaut : 0.25 s).")
    p.add_argument("-D", "--boundary_delta",   dest="boundary_delta",  metavar="f",  type=float, default=0.02,
                   help="Tolérance de frontiéres")
    p.add_argument("-s", "--uem",      dest="uemfn",   metavar="fn",
                   help="Fichier UEM définissant les zones de scoring.")
    p.add_argument("-n", "--max-segments", dest="maxseg", metavar="n", type=int, default=0,
                   help="Nombre maximum de segments hypothèse par source/événement.")
    p.add_argument("-t", "--subtype",  dest="subtype", action="store_true", default=False,
                   help="Activer les rapports par sous-type.")
    p.add_argument("-r", "--error",    dest="dout",    metavar="s",  action="append", nargs="?",
                   help="Rapport de détection. Valeurs : sum, event, source, subtype (combinables avec +).")
    p.add_argument("-b", "--segmentation", dest="sout", metavar="s", action="append", nargs="?",
                   help="Rapport de segmentation. Valeurs : sum, event, source (combinables avec +).")
    p.add_argument("-d", "--det",      dest="detfn",   metavar="fn", nargs="?", const="",
                   help="Préfixe pour les fichiers DET. Sans argument : seulement les points singuliers.")
    p.add_argument("-a", "--align",    dest="align",   action="store_true", default=False,
                   help="Afficher l'alignement ref/hyp.")
    p.add_argument("-o", "--output",   dest="outfn",   metavar="fn", default="-",
                   help="Fichier de sortie (défaut : stdout).")
    p.add_argument("-v", "--verbose",  dest="trace",   action="store_true", default=False,
                   help="Mode verbeux.")
    p.add_argument("-V", "--version",  dest="version", action="store_true", default=False,
                   help="Afficher la version et quitter.")
    p.add_argument("reffn", nargs="?", help="Fichier ETF de référence.")
    p.add_argument("hypfn", nargs="?", help="Fichier ETF hypothèse.")
    p.add_argument("--boundary-f1", dest="bnd_f1", action="store_true", default=False,
                   help="Calculer F1 par tolérance de frontière (en plus du F1 par recouvrement).")
    args = p.parse_args()
    return args


def normalize_report_list(raw):
    """
    Normalise la liste des options de rapport (--error / --segmentation).

    En Perl, lorsque l'option est passée sans valeur, l'élément est une
    chaîne vide que grep remplace par "sum". On reproduit ce comportement ici.

    Exemples :
        --error          → raw = [None]  → ["sum"]
        --error=event    → raw = ["event"] → ["event"]
        --error=event+source → ["event+source"]
    """
    result = []
    for item in (raw or []):
        if item is None or item == "":
            result.append("sum")
        else:
            # Support des virgules : --error=event,source → ["event", "source"]
            result.extend(item.split(","))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LECTURE DES FICHIERS ETF
# ══════════════════════════════════════════════════════════════════════════════
#
# Équivalent de sub etfread() en Perl.
# Chaque segment est représenté comme un dict Python (vs hash Perl $seg->{...}).
# ─────────────────────────────────────────────────────────────────────────────

# Regex qui parse une ligne ETF. Reproduit fidèlement la regex Perl :
#   <source> <channel> <start> <dur> <type> <subtype> <event> [<score> [<decision>]]
# Le score accepte : '-', entier, flottant, notation scientifique (e±N)
ETF_RE = re.compile(
    r'^(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)'
    r'(?:\s+(-|-?\d+(?:\.\d*)?(?:e[+\-]\d+)?)(?:\s+(\S+))?)?$',
    re.IGNORECASE
)


def etfread(fn):
    """
    Lit un fichier ETF et retourne une liste de dicts de segments.

    Chaque segment (dict) contient :
      filename, channel, start_time, duration, type, subtype,
      event, score, decision

    Détails importants :
    - Les commentaires commencent par ';' et sont supprimés.
    - Les lignes vides après nettoyage sont ignorées.
    - subtype vaut None si le champ est '-' ou 'na' (insensible casse).
    - score vaut None si '-' ou 'na'.
    - decision vaut 'true' par défaut si le champ est absent.
    """
    segments = []

    with open(fn, "r", encoding="utf-8") as f:
        for lino, line in enumerate(f, start=1):
            # Supprimer les commentaires (';' et tout ce qui suit)
            line = re.sub(r';.*', '', line).strip()
            if not line:
                continue  # Ligne vide → ignorer

            m = ETF_RE.match(line)
            if not m:
                raise ValueError(f"Erreur de format dans {fn} à la ligne {lino} : {line!r}")

            source, channel, start_s, dur_s, typ, subtype, event, score_s, decision = m.groups()

            # Conversion des champs numériques
            start_time = float(start_s)
            duration   = float(dur_s)

            # subtype : None si '-' ou 'na'
            subtype = None if (subtype == "-" or subtype.lower() == "na") else subtype

            # score : None si absent, '-', ou 'na'
            if score_s is None or score_s == "-" or score_s.lower() == "na":
                score = None
            else:
                score = float(score_s)

            # decision : 'true' par défaut si absent
            if decision is None:
                decision = "true"

            seg = {
                "filename":   source,
                "channel":    channel,
                "start_time": start_time,
                "duration":   duration,
                "end_time":   start_time + duration,  # calculé une fois pour toutes
                "type":       typ,
                "subtype":    subtype,
                "event":      event,
                "score":      score,
                "decision":   decision,
            }
            segments.append(seg)

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LECTURE DU FICHIER UEM
# ══════════════════════════════════════════════════════════════════════════════
#
# Équivalent de sub uemread() + sub partition() en Perl.
# ─────────────────────────────────────────────────────────────────────────────

def uemread(fn):
    """
    Lit un fichier UEM (Universal Evaluation Map).

    Format : <source> <channel> <start_time> <end_time>
    Retourne une liste de dicts avec les zones de scoring autorisées.
    """
    regions = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(r';.*', '', line).strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            regions.append({
                "filename":   parts[0],
                "channel":    parts[1],
                "start_time": float(parts[2]),
                "end_time":   float(parts[3]),
            })
    return regions


def partition(etf_segs, uem_regions):
    """
    Découpe les segments ETF selon les zones UEM.

    Pour chaque segment de référence, on calcule l'intersection avec
    chacune des zones UEM correspondant au même fichier source.
    Seules les portions chevauchantes sont conservées.

    Équivalent de sub partition() en Perl.

    Exemple visuel :
      Segment ref  : [--------------------]
      Zone UEM     :         [--------]
      Résultat     :         [--------]
    """
    result = []
    for seg in etf_segs:
        st = seg["start_time"]
        et = seg["end_time"]

        # Filtrer les zones UEM pour ce fichier source, triées par start_time
        buf = sorted(
            [u for u in uem_regions if u["filename"] == seg["filename"]],
            key=lambda u: u["start_time"]
        )

        for uzone in buf:
            if uzone["end_time"] < st:
                continue   # Zone UEM entièrement avant le segment → passer
            if uzone["start_time"] > et:
                break      # Zone UEM entièrement après → plus rien à faire

            # Calcul de l'intersection
            sst = max(st, uzone["start_time"])
            set_ = min(et, uzone["end_time"])

            new_seg = {
                "filename":   seg["filename"],
                "channel":    seg["channel"],
                "start_time": sst,
                "duration":   set_ - sst,
                "end_time":   set_,
                "type":       seg["type"],
                "subtype":    seg["subtype"],
                "event":      seg["event"],
                "score":      seg["score"],
                "decision":   seg["decision"],
            }
            result.append(new_seg)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GESTION DE LA LISTE D'ÉVÉNEMENTS
# ══════════════════════════════════════════════════════════════════════════════

def load_event_list(fn):
    """
    Charge une liste d'événements depuis un fichier texte.
    Chaque ligne non-vide et non-commentée contribue son premier token.
    Équivalent de sub load_event_list() en Perl.
    """
    events = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(r';.*', '', line).strip()
            if not line:
                continue
            token = line.split()[0]
            events.append(token)
    return events


def make_event_list(segments):
    """
    Construit la liste des événements uniques présents dans les segments.
    Équivalent de sub make_event_list() en Perl.
    """
    return list({seg["event"] for seg in segments})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MÉTRIQUES : t2m()
# ══════════════════════════════════════════════════════════════════════════════
#
# Équivalent de sub t2m() (Times to Measures) en Perl.
# C'est la fonction de conversion durées → métriques de performance.
# ─────────────────────────────────────────────────────────────────────────────

def t2m(miss, tar, ins, non):
    """
    Convertit les durées brutes en métriques de performance.

    Paramètres
    ----------
    miss : float  — durée de cible manquée (faux négatif)
    tar  : float  — durée totale de cible présente (référence positive)
    ins  : float  — durée de fausse alarme (faux positif)
    non  : float  — durée totale où la cible est absente (référence négative)

    Retourne
    --------
    fr   : taux de miss     = miss / tar
    fa   : taux de FA       = ins  / non
    e    : error rate       = (miss + ins) / (tar + non)
    r    : recall           = correct / tar
    p    : precision        = correct / detected
    f    : F-mesure         = 2rp / (r+p)

    Gestion des cas limites (divisions par zéro) :
    - tar = 0  → fr = 0, recall = 1 (convention ESTER), precision = 1 si rien détecté
    - non = 0  → fa = 0
    - r+p = 0  → f  = 0
    """
    correct  = tar - miss           # durée correctement détectée
    detected = correct + ins        # durée totale détectée (correct + insertion)

    fr = miss / tar if tar > 0 else 0.0
    fa = ins  / non if non > 0 else 0.0
    e  = (miss + ins) / (tar + non) if (tar + non) > 0 else 0.0

    if tar > 0:
        r = correct / tar
        p = correct / detected if detected > 0 else 0.0
    else:
        # Convention : si aucune cible n'existe, recall = 1
        r = 1.0
        p = correct / detected if detected > 0 else 1.0

    f = (2 * r * p) / (r + p) if (r + p) > 0 else 0.0

    return fr, fa, e, r, p, f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — COMPARAISON ETF : etfcmp()
# ══════════════════════════════════════════════════════════════════════════════
#
# Cœur algorithmique du script. Compare ref et hyp pour un événement/source.
# Équivalent de sub etfcmp() en Perl.
# ─────────────────────────────────────────────────────────────────────────────

def etfcmp(ref_segs, hyp_segs, det, subtype_mode, subs, margin, align, outf):
    """
    Compare deux listes de segments (ref vs hyp) pour un couple (événement, source).

    Paramètres
    ----------
    ref_segs     : liste de segments de référence (triés par start_time)
    hyp_segs     : liste de segments hypothèse (triés par start_time)
    det          : dict global DET (modifié en place)
    subtype_mode : bool — activer l'accumulation par sous-type
    subs         : liste des sous-types valides
    margin       : float — tolérance temporelle en secondes
    align        : bool — afficher l'alignement ref/hyp
    outf         : fichier de sortie

    Retourne
    --------
    miss, tar, ins, non, ebuf

    où ebuf est un dict {subtype: {miss, ins, tar, non}} si subtype_mode.

    Algorithme de sweep (balayage linéaire)
    ----------------------------------------
    Pour chaque segment de référence (après application de la marge) :
      1. Avancer dans hyp jusqu'au premier segment chevauchant le ref
      2. Pour chaque hyp chevauchant :
         a. Si une portion de ref est non couverte avant hyp → MISS
         b. Calculer l'intersection ref ∩ hyp :
            - Si ref=positif et hyp=négatif → MISS sur l'intersection
            - Si ref=négatif et hyp=positif → INSERTION sur l'intersection
      3. Après tous les hyp chevauchants, si reste de ref → MISS

    La table DET est alimentée en parallèle par valeur de score.

    Note sur $eps = 1e-10 :
    Seuil anti-flottant pour ignorer les chevauchements infinitésimaux
    (ex. deux segments qui se touchent exactement en un point).
    """
    EPS = 1e-10

    miss = 0.0
    ins  = 0.0
    tar  = 0.0
    non  = 0.0
    if '*' not in det:
        det['*'] = {}
    # Buffer d'erreurs par sous-type
    ebuf = {}
    if subtype_mode:
        for x in subs:
            ebuf[x] = {"miss": 0.0, "ins": 0.0, "tar": 0.0, "non": 0.0}

    nhyp  = len(hyp_segs)
    hypid = 0   # pointeur courant dans hyp_segs (avance toujours vers l'avant)

    for rseg in ref_segs:

        # ── Application de la marge ──────────────────────────────────────────
        # La marge réduit le segment de référence des deux côtés.
        # Un segment de référence de durée < 2*margin est ignoré.
        rst  = rseg["start_time"] + margin
        ret  = rseg["start_time"] + rseg["duration"] - margin
        dur  = ret - rst
        rdec = rseg["decision"]
        x    = rseg["subtype"]   # sous-type du segment de référence

        if align:
            outf.write(f"ref=({rseg['start_time']:.4f}, {rseg['end_time']:.4f}, {rdec[0]})\n")

        if dur <= 0:
            continue   # Segment trop court après marge → ignorer

        # ── Avance dans hyp jusqu'au premier chevauchement possible ──────────
        # Un hyp est "avant" rst si sa fin est < rst + EPS
        while hypid < nhyp and hyp_segs[hypid]["end_time"] < rst + EPS:
            hypid += 1

        # st = position courante scorée dans le segment de référence
        st = rst

        # ── Traitement de tous les hyp qui chevauchent [rst, ret) ───────────
        tmp_hypid = hypid
        while tmp_hypid < nhyp and hyp_segs[tmp_hypid]["start_time"] < ret - EPS:

            hseg  = hyp_segs[tmp_hypid]
            hst   = hseg["start_time"]
            het   = hseg["end_time"]
            hdec  = hseg["decision"]
            score = hseg["score"]

            # Initialiser les entrées DET pour ce score si nécessaires
            _det_key = score  # peut être None → on le traite comme clé
            if '*' not in det:
                det['*'] = {}
            if _det_key not in det['*']:
                det['*'][_det_key] = {"miss": 0.0, "ins": 0.0}
            if subtype_mode and x is not None:
                if x not in det:
                    det[x] = {}
                if _det_key not in det[x]:
                    det[x][_det_key] = {"miss": 0.0, "ins": 0.0}

            # ── Zone initiale non couverte : [st, hst) ──────────────────────
            # S'il y a un trou entre la position courante et le début du hyp,
            # et que la référence est positive → c'est un MISS
            if hst > st and re.search(r't', rdec, re.IGNORECASE):
                d = hst - st
                if align:
                    outf.write(
                        f"    miss={d:<9.4f}    [{st:10.4f},{hst:10.4f}]"
                        f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n"
                    )
                miss += d
                det['*'].setdefault('offset', 0.0)
                det['*']['offset'] += d
                if subtype_mode and x is not None:
                    ebuf[x]["miss"] += d
                    det.setdefault(x, {})
                    det[x].setdefault('offset', 0.0)
                    det[x]['offset'] += d

            # ── Zone d'intersection ─────────────────────────────────────────
            a = max(hst, st)    # début de l'intersection
            b = min(het, ret)   # fin de l'intersection
            d = b - a           # durée de l'intersection

            if re.search(r't', rdec, re.IGNORECASE):
                # Référence positive
                if not re.search(r't', hdec, re.IGNORECASE):
                    # Hyp négative sur fond positif → MISS
                    if align:
                        outf.write(
                            f"    miss={d:<9.4f}    [{a:10.4f},{b:10.4f}]"
                            f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n"
                        )
                    miss += d
                    if subtype_mode and x is not None:
                        ebuf[x]["miss"] += d
                # Toujours alimenter la table DET (pour la courbe DET)
                det['*'][_det_key]["miss"] += d
                if subtype_mode and x is not None and _det_key is not None:
                    det[x][_det_key]["miss"] += d
            else:
                # Référence négative
                if not re.search(r'f', hdec, re.IGNORECASE):
                    # Hyp positive sur fond négatif → INSERTION (fausse alarme)
                    if align:
                        outf.write(
                            f"    insert={d:<9.4f}    [{a:10.4f},{b:10.4f}]"
                            f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n"
                        )
                    ins += d
                    if subtype_mode and x is not None:
                        ebuf[x]["ins"] += d
                det['*'][_det_key]["ins"] += d
                if subtype_mode and x is not None and _det_key is not None:
                    det[x][_det_key]["ins"] += d

            st = b  # on a scoré jusqu'à b

            # Si le hyp dépasse la fin du ref, il peut servir pour le ref suivant
            if het > ret:
                break
            tmp_hypid += 1

        # ── Reste du segment de référence non couvert ────────────────────────
        if st < ret - EPS and re.search(r't', rdec, re.IGNORECASE):
            d = ret - st
            if align:
                outf.write(f"    miss={d:<9.4f}    [{st:10.4f},{ret:10.4f}]\n")
            miss += d
            det['*'].setdefault('offset', 0.0)
            det['*']['offset'] += d
            if subtype_mode and x is not None:
                ebuf[x]["miss"] += d
                det.setdefault(x, {})
                det[x].setdefault('offset', 0.0)
                det[x]['offset'] += d

        # ── Accumulation tar / non ───────────────────────────────────────────
        if re.search(r't', rdec, re.IGNORECASE):
            tar += dur
            if subtype_mode and x is not None:
                ebuf[x]["tar"] += dur
        else:
            non += dur
            if subtype_mode and x is not None:
                ebuf[x]["non"] += dur

    return miss, tar, ins, non, ebuf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATISTIQUES DE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def bnd_f1_print(stats, events, sources, subs, outf):
    """
    Affiche le tableau F1 par tolérance de frontière, avec décomposition par style.
    """
    xtab = ['*'] + list(subs) if subs else ['*']
    line_len = 74

    for x in xtab:
        label = "global" if x == '*' else x
        outf.write("\n\n")
        outf.write("    " + "-" * line_len + "\n")
        outf.write(f"    | F1 par tolérance de frontière — {label}\n")
        outf.write("    " + "-" * line_len + "\n")
        outf.write(f"    | {'event':<20s} {'source':<30s} | "
                   f"{'TP':>6s} {'FN':>6s} {'FP':>6s} | "
                   f"{'recall':>8s} {'precision':>9s} {'F1':>8s} |\n")
        outf.write("    " + "-" * line_len + "\n")

        sum_tp = sum_fn = sum_fp = 0

        for evt in sorted(events):
            for src in sorted(sources.keys()):
                # CORRECTION : accéder au bon niveau d'imbrication
                s = stats.get(evt, {}).get(src, {}).get(x, {})
                tp = s.get("bnd_tp", 0)
                fn = s.get("bnd_fn", 0)
                fp = s.get("bnd_fp", 0)
                f1 = s.get("bnd_f1", 0.0)
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                outf.write(f"    | {evt:<20s} {src:<30s} | "
                           f"{tp:>6d} {fn:>6d} {fp:>6d} | "
                           f"{100*rec:>8.3f} {100*prec:>9.3f} {f1:>8.4f} |\n")

                sum_tp += tp
                sum_fn += fn
                sum_fp += fp

        outf.write("    " + "=" * line_len + "\n")
        g_rec  = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        g_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        g_f1   = 2*g_rec*g_prec / (g_rec+g_prec) if (g_rec+g_prec) > 0 else 0.0

        outf.write(f"    | {'Summary':<20s} {'':30s} | "
                   f"{sum_tp:>6d} {sum_fn:>6d} {sum_fp:>6d} | "
                   f"{100*g_rec:>8.3f} {100*g_prec:>9.3f} {g_f1:>8.4f} |\n")
        outf.write("    " + "=" * line_len + "\n")
def etfstat(segments):
    """
    Calcule des statistiques de base sur une liste de segments.
    Sépare les segments positifs (decision=~t) et négatifs.

    Retourne (n_pos, dur_pos, n_neg, dur_neg).
    Équivalent de sub etfstat() en Perl.
    """
    n   = [0, 0]
    dur = [0.0, 0.0]
    for seg in segments:
        i = 0 if re.search(r't', seg["decision"], re.IGNORECASE) else 1
        n[i]   += 1
        dur[i] += seg["duration"]
    return n[0], dur[0], n[1], dur[1]


def etfbcmp(ref_segs, hyp_segs, delta):
    """
    Compare les frontières temporelles entre ref et hyp.

    Pour chaque frontière de référence (début ou fin des segments positifs),
    on cherche la frontière hyp la plus proche dans une fenêtre de ±delta.

    Algorithme :
    - Collecte toutes les frontières (start + end) des segments positifs
    - Balayage linéaire avec deux pointeurs : ihyp avance dans @hbounds
    - Chaque frontière hyp ne peut être appariée qu'une fois

    Retourne le nombre de frontières correctement détectées.
    Équivalent de sub etfbcmp() en Perl.

    Note sur le facteur 2 dans recall/precision :
    Chaque segment contribue 2 frontières (début + fin), d'où la division par 2×n.
    """
    rbounds = []
    hbounds = []

    for seg in ref_segs:
        if re.search(r't', seg["decision"], re.IGNORECASE):
            rbounds.append(seg["start_time"])
            rbounds.append(seg["end_time"])

    for seg in hyp_segs:
        if re.search(r't', seg["decision"], re.IGNORECASE):
            hbounds.append(seg["start_time"])
            hbounds.append(seg["end_time"])

    if not rbounds or not hbounds:
        return 0

    nhyp  = len(hbounds)
    ihyp  = 0
    n     = 0   # compteur de frontières correctement appariées

    for rt in rbounds:
        ibest = -1

        # Avancer dans hbounds tant que la frontière hyp est ≤ rt + delta
        while ihyp < nhyp and hbounds[ihyp] <= rt + delta:
            d = abs(rt - hbounds[ihyp])
            if d <= delta and (ibest < 0 or d < abs(rt - hbounds[ibest])):
                ibest = ihyp
            ihyp += 1

        if ibest >= 0:
            n    += 1
            ihyp  = ibest + 1   # cette frontière hyp est consommée

        if ihyp == nhyp:
            break   # Plus de frontières hyp → terminé

    return n

def etfbcmp_f1(ref_segs, hyp_segs, delta):
    """
    Calcule TP, FN, FP par tolérance de frontière.

    Différence avec etfbcmp() :
    - etfbcmp()    → retourne nb de frontières correctes (pour recall/precision séparés)
    - etfbcmp_f1() → retourne (tp, fn, fp) pour calculer F1 directement

    Appariement glouton : on prend toujours la frontière hyp
    la plus proche dans ±delta. Chaque frontière hyp ne peut
    être appariée qu'une seule fois.
    """
    rbounds = []
    for seg in ref_segs:
        if re.search(r't', seg["decision"], re.IGNORECASE):
            rbounds.append(seg["start_time"])
            rbounds.append(seg["end_time"])

    hbounds = []
    for seg in hyp_segs:
        if re.search(r't', seg["decision"], re.IGNORECASE):
            hbounds.append(seg["start_time"])
            hbounds.append(seg["end_time"])

    rbounds = sorted(rbounds)
    hbounds = sorted(hbounds)

    used_hyp = set()
    tp = 0

    for rt in rbounds:
        best_idx  = -1
        best_dist = float('inf')
        for i, ht in enumerate(hbounds):
            if i in used_hyp:
                continue
            d = abs(rt - ht)
            if d <= delta and d < best_dist:
                best_dist = d
                best_idx  = i
        if best_idx >= 0:
            tp += 1
            used_hyp.add(best_idx)

    fn = len(rbounds) - tp
    fp = len(hbounds) - tp

    return tp, fn, fp
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — AGRÉGATION DES ERREURS
# ══════════════════════════════════════════════════════════════════════════════
#
# Équivalent de sub error_sum() et sub bound_sum() en Perl.
# En Perl, '*' est utilisé comme clé "total". On conserve cette convention.
# ─────────────────────────────────────────────────────────────────────────────

def error_sum(err, events, sources, subs, subtype_mode):
    """
    Agrège les erreurs de détection sur l'ensemble des événements et sources.

    Structure de err après agrégation :
      err[evt][src][subtype_ou_*][miss/tar/ins/non]

    Les clés '*' représentent les totaux :
      err['*']['*']['*'] = total global
      err[evt]['*']['*'] = total par événement
      err['*'][src]['*'] = total par source
    """
    key_list = list(subs) + ['*'] if subtype_mode else ['*']

    for k in ("miss", "ins", "tar", "non"):
        for x in key_list:
            err.setdefault('*', {}).setdefault('*', {}).setdefault(x, {})[k] = 0.0
            for src in sources:
                err.setdefault('*', {}).setdefault(src, {}).setdefault(x, {})[k] = 0.0
            for evt in events:
                err.setdefault(evt, {}).setdefault('*', {}).setdefault(x, {})[k] = 0.0

    for evt in events:
        for src in sources:
            for k in ("miss", "ins", "tar", "non"):
                for x in key_list:
                    v = err.get(evt, {}).get(src, {}).get(x, {}).get(k, 0.0)
                    err['*']['*'][x][k]   += v
                    err['*'][src][x][k]   += v
                    err[evt]['*'][x][k]   += v


def bound_sum(stats, events, sources):
    fields = ("nrsegs", "rlength", "nhsegs", "hlength", "nbcorr")

    for evt in events:
        for src in sources:
            src_stats = stats.get(evt, {}).get(src, {})
            for x, xdata in src_stats.items():
                if not isinstance(xdata, dict):
                    continue   # ignorer 'per', 'sub', 'del', 'ins', 'nref'
                for f in fields:
                    v = xdata.get(f, 0.0)
                    # total global
                    stats.setdefault('*', {}).setdefault('*', {}).setdefault(x, {})
                    stats['*']['*'][x][f] = stats['*']['*'][x].get(f, 0.0) + v
                    # total par source
                    stats.setdefault('*', {}).setdefault(src, {}).setdefault(x, {})
                    stats['*'][src][x][f] = stats['*'][src][x].get(f, 0.0) + v
                    # total par événement
                    stats.setdefault(evt, {}).setdefault('*', {}).setdefault(x, {})
                    stats[evt]['*'][x][f] = stats[evt]['*'][x].get(f, 0.0) + v
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — AFFICHAGE DES RAPPORTS DE DÉTECTION
# ══════════════════════════════════════════════════════════════════════════════

def error_by_event(err, events, sources, subs, subtype_mode, who, called_from_print, outf):
    """
    Calcule (et optionnellement affiche) les erreurs par événement/source/sous-type.

    Paramètres
    ----------
    who              : liste de critères d'axe ('so', 'ev', 'sub')
    called_from_print: bool — si True, affiche chaque ligne dans outf

    Retourne un dict %sum avec les totaux pour calcul de la ligne "Average".
    Équivalent de sub error_by_event() en Perl.
    """
    stab = sorted(sources.keys()) if any(w.startswith("so") for w in who) else ['*']
    etab = events               if any(w.startswith("ev") for w in who) else ['*']
    xtab = subs if (subtype_mode and any(w.startswith("sub") for w in who)) else ['*']

    total = dict(miss=0.0, ins=0.0, tar=0.0, non=0.0,
                 fr=0.0, fa=0.0, err=0.0, r=0.0, p=0.0, F=0.0, nb_evt=0)

    for s in stab:
        for e in etab:
            for x in xtab:
                miss = err.get(e, {}).get(s, {}).get(x, {}).get("miss", 0.0)
                tar  = err.get(e, {}).get(s, {}).get(x, {}).get("tar",  0.0)
                ins_ = err.get(e, {}).get(s, {}).get(x, {}).get("ins",  0.0)
                non_ = err.get(e, {}).get(s, {}).get(x, {}).get("non",  0.0)

                fr, fa, er, r, p, f = t2m(miss, tar, ins_, non_)

                if called_from_print:
                    outf.write("    |")
                    if s != '*':
                        outf.write(f" {s:<40s} ")
                    if e != '*':
                        outf.write(f" {e:<30s} ")
                    if x != '*':
                        outf.write(f" {x:<10s} ")
                    outf.write(f" | {tar:9.2f} {non_:9.2f} | {miss:8.2f}  {ins_:8.2f} |")
                    outf.write(f" {100*er:7.3f}  {100*fr:7.3f}  {100*fa:7.3f} |")
                    outf.write(f" {100*r:7.3f}  {100*p:7.3f}  {f:6.4f} |\n")

                total["miss"]   += miss
                total["tar"]    += tar
                total["ins"]    += ins_
                total["non"]    += non_
                total["fr"]     += fr
                total["fa"]     += fa
                total["err"]    += er
                total["r"]      += r
                total["p"]      += p
                total["F"]      += f
                total["nb_evt"] += 1

    return total


def error_print(spec, err, events, sources, subs, subtype_mode, outf):
    """
    Affiche le tableau de détection selon la spécification spec.

    spec est une chaîne comme "event+source", "source", "event+subtype", etc.
    Les tokens reconnus sont : 'source' (→ 'so'), 'event' (→ 'ev'), 'subtype' (→ 'sub').
    Équivalent de sub error_print() en Perl.
    """
    # Extraire les axes depuis la spec (ex. "event+source" → ["ev", "so"])
    who = [tok for tok in re.split(r'\+', spec) if re.match(r'^(so|ev|sub)', tok)]
    if not who:
        return

    # Calcul de la largeur du tableau (reproduit la logique Perl)
    hl = 0
    hl += 42 if any(w.startswith("so") for w in who) else 0
    hl += 32 if any(w.startswith("ev") for w in who) else 0
    hl += 12 if subtype_mode and any(w.startswith("sub") for w in who) else 0
    line_len = hl + 100

    # En-tête
    outf.write("\n\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")
    outf.write("    |")
    if any(w.startswith("so") for w in who):
        outf.write(f" {'source':<40s} ")
    if any(w.startswith("ev") for w in who):
        outf.write(f" {'event':<30s} ")
    if subtype_mode and any(w.startswith("sub") for w in who):
        outf.write(f" {'subtype':<10s} ")
    outf.write(f" | {'tar.':>9s} {'non':>9s} | {'miss':>8s}  {'ins':>8s} | "
               f"{'%err':>7s}  {'%miss':>7s}  {'%fa':>7s} | "
               f"{'%rec':>7s}  {'%prec':>7s}  {'F':>6s} |\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")

    # Corps du tableau
    total = error_by_event(err, events, sources, subs, subtype_mode, who,
                           called_from_print=True, outf=outf)

    # Pied du tableau : Average + Summary
    outf.write("    " + "=" * (line_len + 1) + "\n")
    fmt_lbl = f"    | {{:<{hl}s}}|"

    n = total["nb_evt"]
    if n:
        outf.write(fmt_lbl.format("Average"))
        outf.write(f" {total['tar']/n:9.2f} {total['non']/n:9.2f} | "
                   f"{total['miss']/n:8.2f}  {total['ins']/n:8.2f} |")
        outf.write(f" {100*total['err']/n:7.3f}  {100*total['fr']/n:7.3f}  {100*total['fa']/n:7.3f} |")
        outf.write(f" {100*total['r']/n:7.3f}  {100*total['p']/n:7.3f}  {total['F']/n:6.4f} |\n")

    g_miss = err['*']['*']['*']['miss']
    g_tar  = err['*']['*']['*']['tar']
    g_ins  = err['*']['*']['*']['ins']
    g_non  = err['*']['*']['*']['non']
    fr, fa, e, r, p, f = t2m(g_miss, g_tar, g_ins, g_non)

    outf.write(fmt_lbl.format("Summary"))
    outf.write(f" {g_tar:9.2f} {g_non:9.2f} | {g_miss:8.2f}  {g_ins:8.2f} |")
    outf.write(f" {e:7.3f}  {100*fr:7.3f}  {100*fa:7.3f} |")
    outf.write(f" {100*r:7.3f}  {100*p:7.3f}  {f:6.4f} |\n")
    outf.write("    " + "=" * (line_len + 1) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — AFFICHAGE DES RAPPORTS DE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def bound_print(spec, stats, events, sources, outf):
    """
    Affiche le tableau de statistiques de segmentation (avec support subtype).
    """

    # 🔥 inclure subtype
    who = [tok for tok in re.split(r'\+', spec) if re.match(r'^(so|ev|sub)', tok)]
    if not who:
        return

    stab = sorted(sources.keys()) if any(w.startswith("so") for w in who) else ['*']
    etab = events if any(w.startswith("ev") for w in who) else ['*']
    # Collecter tous les styles réels sur l'ensemble events × sources
    all_styles = set()
    for evt in events:
        for src in sources:
            for k, v in stats.get(evt, {}).get(src, {}).items():
                if isinstance(v, dict) and k != '*':
                    all_styles.add(k)
    
    if any(w.startswith("sub") for w in who) and all_styles:
        xtab = sorted(all_styles)   # ex. ['planned', 'semi-spont', 'spont']
    else:
        xtab = ['*']

    # largeur dynamique
    hl = 0
    hl += 42 if any(w.startswith("so") for w in who) else 0
    hl += 32 if any(w.startswith("ev") for w in who) else 0
    hl += 12 if any(w.startswith("sub") for w in who) else 0
    hl = max(hl - 1, 0)
    line_len = hl + 60

    # ── HEADER ─────────────────────────────────────────
    outf.write("\n\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")
    outf.write("    |" + " " * (hl + 1) +
               "|        ref        |        hyp        |    boundaries    |\n")
    outf.write("    |")

    if any(w.startswith("so") for w in who):
        outf.write(f" {'source':<40s} ")
    if any(w.startswith("ev") for w in who):
        outf.write(f" {'event':<30s} ")
    if any(w.startswith("sub") for w in who):
        outf.write(f" {'subtype':<10s} ")

    outf.write("|  nsegs     length |  nsegs     length |   %rec     %prec |\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")

    sumv = dict(nr=0.0, rad=0.0, nh=0.0, had=0.0, r=0.0, p=0.0)
    n_rows = 0

    # ── LOOP ───────────────────────────────────────────
    for s in stab:
        for e in etab:
            for x in xtab:

                row_stats = stats.get(e, {}).get(s, {}).get(x, {})

                # skip lignes vides
                if not row_stats:
                    continue

                outf.write("    |")

                if s != '*':
                    outf.write(f" {s:<40s} ")
                if e != '*':
                    outf.write(f" {e:<30s} ")
                if any(w.startswith("sub") for w in who):
                    outf.write(f" {x:<10s} ")

                nr = row_stats.get("nrsegs", 0)
                nh = row_stats.get("nhsegs", 0)

                rad = row_stats.get("rlength", 0.0) / nr if nr else 0.0
                had = row_stats.get("hlength", 0.0) / nh if nh else 0.0

                nc = row_stats.get("nbcorr", 0.0)

                r = 100.0 * nc / (2 * nr) if nr else 0.0
                p = 100.0 * nc / (2 * nh) if nh else 0.0

                outf.write(
                    f"| {int(nr):6d}   {rad:8.2f} | {int(nh):6d}   {had:8.2f} | {r:7.3f}  {p:7.3f} |\n"
                )

                sumv["nr"] += nr
                sumv["rad"] += rad
                sumv["nh"] += nh
                sumv["had"] += had
                sumv["r"] += r
                sumv["p"] += p
                n_rows += 1

    # ── AVERAGE ───────────────────────────────────────
    outf.write("    " + "=" * (line_len + 1) + "\n")
    fmt_lbl = f"    | {{:<{hl}s}}|"

    if n_rows:
        outf.write(fmt_lbl.format("Average"))
        outf.write(
            f"   {sumv['nr']/n_rows:7.2f} {sumv['rad']/n_rows:7.2f} |"
            f"   {sumv['nh']/n_rows:7.2f} {sumv['had']/n_rows:7.2f} |"
            f" {sumv['r']/n_rows:7.3f}  {sumv['p']/n_rows:7.3f} |\n"
        )

    # ── GLOBAL SUMMARY (⚠️ reste global, pas par subtype)
    g = stats.get('*', {}).get('*', {}).get('*', {})
    g_nr = g.get("nrsegs", 0)
    g_nh = g.get("nhsegs", 0)

    g_rad = g.get("rlength", 0.0) / g_nr if g_nr else 0.0
    g_had = g.get("hlength", 0.0) / g_nh if g_nh else 0.0

    g_nc = g.get("nbcorr", 0.0)

    g_r = 100.0 * g_nc / (2 * g_nr) if g_nr else 0.0
    g_p = 100.0 * g_nc / (2 * g_nh) if g_nh else 0.0

    outf.write(fmt_lbl.format("Summary"))
    outf.write(
        f" {int(g_nr):6d}   {g_rad:8.2f} | {int(g_nh):6d}   {g_had:8.2f} | {g_r:7.3f}  {g_p:7.3f} |\n"
    )
    outf.write("    " + "=" * (line_len + 1) + "\n")
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — COURBES DET
# ══════════════════════════════════════════════════════════════════════════════

def det_print(ofn, tar, non, det_tab, detfn, trace):
    """
    Calcule et affiche les points DET et les points singuliers.

    La courbe DET (Detection Error Tradeoff) trace FR vs FA pour tous les
    seuils possibles. Elle est générée en balayant les scores.

    Algorithme en deux passes :
    1. Passe descendante (score haut → bas) : accumule FA (insertions)
    2. Passe montante (score bas → haut)    : accumule FR (miss)

    Points singuliers calculés :
    - F   : seuil maximisant le F-measure
    - err : seuil minimisant le taux d'erreur global
    - ter : seuil minimisant FR + FA (Total Error Rate)
    - eer : seuil approximant l'Equal Error Rate (FR ≈ FA)

    Équivalent de sub det_print() en Perl.
    """
    if trace:
        print("computing DET points ...")

    det_file = None
    if detfn is not None and detfn != "":
        det_file = open(ofn, "w", encoding="utf-8")
        det_file.write("# threshold fr fa error recall precision F-measure\n")

    pts = {}   # points singuliers
    ins_acc  = 0.0
    miss_acc = det_tab.pop('offset', 0.0)   # miss inévitables (aucune hyp couvrant cette zone)

    # ── Passe 1 : accumulation FA (score décroissant) ────────────────────────
    # Un score élevé = système plus sûr → en abaissant le seuil, on accepte
    # de plus en plus de segments → FA augmente progressivement.
    valid_keys = [k for k in det_tab if k is not None]
    for th in sorted(valid_keys, reverse=True):
        ins_acc += det_tab[th]["ins"]
        det_tab[th]["ins"] = ins_acc   # mise à jour cumulative

    # ── Passe 2 : accumulation FR + calcul métriques ─────────────────────────
    for th in sorted(valid_keys):
        miss_acc += det_tab[th]["miss"]
        ins_acc   = det_tab[th]["ins"]

        fr, fa, e, r, p, f = t2m(miss_acc, tar, ins_acc, non)

        if det_file:
            det_file.write(f"{th:.6f} {fr:.6f} {fa:.6f} {e:.6f} {r:.6f} {p:.6f} {f:.6f}\n")

        if not pts:
            # Initialisation des 4 points singuliers au premier seuil
            for key in ("F", "err", "ter", "eer"):
                pts[key] = {"th": th, "fa": fa, "fr": fr, "r": r, "p": p}
            pts["F"]["val"]   = f
            pts["err"]["val"] = e
            pts["ter"]["val"] = fr + fa
            pts["eer"]["val"] = (fr + fa) / 2
            pts["eer"]["diff"] = abs(fr - fa)
        else:
            if f > pts["F"]["val"]:
                pts["F"].update(val=f, th=th, fa=fa, fr=fr, r=r, p=p)
            if e < pts["err"]["val"]:
                pts["err"].update(val=e, th=th, fa=fa, fr=fr, r=r, p=p)
            if fr + fa < pts["ter"]["val"]:
                pts["ter"].update(val=fr+fa, th=th, fa=fa, fr=fr, r=r, p=p)
            if abs(fr - fa) < pts["eer"]["diff"]:
                pts["eer"].update(val=(fr+fa)/2, diff=abs(fr-fa), th=th, fa=fa, fr=fr, r=r, p=p)

    if det_file:
        det_file.close()

    return pts
def compute_per_metrics(ref, hyp, events, sources):
    """
    Compute PER-style metrics using Levenshtein alignment.

    Returns:
        {
            "global": {...},
            "by_event": {...},
            "by_source": {...},
            "by_event_source": {...}
        }
    """

    def levenshtein_alignment(ref_seq, hyp_seq):
        """
        Returns counts: sub, del, ins, cor
        """
        n = len(ref_seq)
        m = len(hyp_seq)

        # DP table
        dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
        # (cost, sub, del, ins)

        # init
        for i in range(1, n + 1):
            dp[i][0] = (i, 0, i, 0)
        for j in range(1, m + 1):
            dp[0][j] = (j, 0, 0, j)

        for i in range(1, n + 1):
            for j in range(1, m + 1):

                if ref_seq[i-1] == hyp_seq[j-1]:
                    cost, s, d, ins = dp[i-1][j-1]
                    dp[i][j] = (cost, s, d, ins)
                else:
                    # substitution
                    c1, s1, d1, i1 = dp[i-1][j-1]
                    c1 += 1

                    # deletion
                    c2, s2, d2, i2 = dp[i-1][j]
                    c2 += 1

                    # insertion
                    c3, s3, d3, i3 = dp[i][j-1]
                    c3 += 1

                    best = min(
                        (c1, s1+1, d1, i1),
                        (c2, s2, d2+1, i2),
                        (c3, s3, d3, i3+1),
                        key=lambda x: x[0]
                    )
                    dp[i][j] = best

        cost, sub, dele, ins = dp[n][m]
        cor = n - sub - dele

        return sub, dele, ins, cor

    def extract_seq(segs):
        return [s["event"] for s in sorted(segs, key=lambda x: x["start_time"])]

    results = {
        "global": {"sub":0, "del":0, "ins":0, "cor":0},
        "by_event": {},
        "by_source": {},
        "by_event_source": {}
    }

    for evt in events:
        results["by_event"].setdefault(evt, {"sub":0, "del":0, "ins":0, "cor":0})

        for src in sources:
            results["by_source"].setdefault(src, {"sub":0, "del":0, "ins":0, "cor":0})

            srcref = [s for s in ref if s["event"] == evt and s["filename"] == src]
            srchyp = [s for s in hyp if s["event"] == evt and s["filename"] == src]

            ref_seq = extract_seq(srcref)
            hyp_seq = extract_seq(srchyp)

            sub, dele, ins, cor = levenshtein_alignment(ref_seq, hyp_seq)

            key = (evt, src)
            results["by_event_source"][key] = {
                "sub": sub, "del": dele, "ins": ins, "cor": cor
            }

            # aggregate
            for k, v in [("sub", sub), ("del", dele), ("ins", ins), ("cor", cor)]:
                results["global"][k] += v
                results["by_event"][evt][k] += v
                results["by_source"][src][k] += v

    # compute PER
    def add_per(d):
        n = d["sub"] + d["del"] + d["cor"]
        d["nref"] = n
        d["per"] = (d["sub"] + d["del"] + d["ins"]) / n if n > 0 else 0.0

    add_per(results["global"])
    for d in results["by_event"].values():
        add_per(d)
    for d in results["by_source"].values():
        add_per(d)
    for d in results["by_event_source"].values():
        add_per(d)

    return results
# ══════════════════════════════════════════════════════════════════════════════
# run_trackeval() : API PROGRAMMATIQUE
# ══════════════════════════════════════════════════════════════════════════════

def run_trackeval(reffn, hypfn,
                  margin=0.0,
                  boundary_delta=0.020,
                  subtype=False,
                  events=None,
                  uemfn=None,
                  maxseg=0,
                  bnd_f1=False):
    """
    Version programmatique de trackeval.

    Au lieu d'écrire dans un fichier texte, retourne les résultats
    sous forme de dictionnaires Python directement exploitables.

    Paramètres
    ----------
    reffn   : str   — chemin fichier ETF référence
    hypfn   : str   — chemin fichier ETF hypothèse
    margin  : float — tolérance (0.0 pour alignement phonémique)
    boundary_delta: float -tolérance pour frontiéres
    subtype : bool  — activer les stats par sous-type
    events  : list  — phonèmes à scorer (None = tous)
    uemfn   : str   — fichier UEM (optionnel)
    maxseg  : int   — nb max segments hyp (0 = pas de limite)
    bnd_f1  : bool  — calculer aussi F1 par tolérance de frontière

    Retourne
    --------
    results : dict avec les clés :
        "global"           → métriques globales (tar, non, miss, ins, recall, precision, F1, ...)
        "by_event"         → métriques par phonème
        "by_source"        → métriques par fichier audio
        "by_event_source"  → métriques par (phonème, fichier)

    

    Exemple
    -------
    results = run_trackeval("ref.etf", "hyp.etf", margin=0.0)

    # Métriques globales
    print(f"F1       = {results['global']['F1']:.4f}")
    print(f"recall   = {results['global']['recall']:.4f}")
    print(f"precision= {results['global']['precision']:.4f}")

    # Par phonème
    for ph, m in results['by_event'].items():
        print(f"{ph:5s} F1={m['F1']:.3f}  recall={m['recall']:.3f}")

    
    """

    # ── Chargement ────────────────────────────────────────────────────────────
    ref = etfread(reffn)
    hyp = etfread(hypfn)

    if not events:
        events = make_event_list(ref)

    sources = {}
    for seg in ref:
        sources[seg["filename"]] = sources.get(seg["filename"], 0) + 1

    if uemfn:
        ref = partition(ref, uemread(uemfn))

    subs = []
    if subtype:
        for seg in ref:
            if seg["subtype"] is not None and seg["subtype"] not in subs:
                subs.append(seg["subtype"])

    for seg in hyp:
        if seg["filename"] not in sources:
            raise ValueError(f"No reference for source '{seg['filename']}' in {reffn}")

    # ── Structures internes ───────────────────────────────────────────────────
    err   = {}
    stats = {}
    det   = {}

    # Sortie texte ignorée (on ne veut que le dict)
    dummy = io.StringIO()

    # ── Boucle de scoring ─────────────────────────────────────────────────────
    for evt in events:
        eref = sorted([s for s in ref if s["event"] == evt], key=lambda s: s["start_time"])
        ehyp = sorted([s for s in hyp if s["event"] == evt], key=lambda s: s["start_time"])

        for src in sorted(sources.keys()):
            srcref = [s for s in eref if s["filename"] == src]
            srchyp = [s for s in ehyp if s["filename"] == src]

            if maxseg > 0 and len(srchyp) > maxseg:
                srchyp = srchyp[:maxseg]

            # Erreurs de détection (pour DET + métriques recouvrement)
            miss, tar_t, ins_t, non_t, ebuf = etfcmp(
                srcref, srchyp, det,
                subtype, subs, margin,
                align=False, outf=dummy
            )
            err.setdefault(evt, {}).setdefault(src, {})['*'] = {
                "miss": miss, "tar": tar_t, "ins": ins_t, "non": non_t
            }
            if subtype:
                for x in subs:
                    err[evt][src][x] = {k: ebuf.get(x, {}).get(k, 0.0)
                                        for k in ("miss", "ins", "tar", "non")}

            # Statistiques de segmentation
            srcref_s = sorted(srcref, key=lambda s: s["start_time"])
            srchyp_s = sorted(srchyp, key=lambda s: s["start_time"])
            rn1, rd1, _, _ = etfstat(srcref_s)
            hn1, hd1, _, _ = etfstat(srchyp_s)
            ncb = etfbcmp(srcref_s, srchyp_s, margin)
            stats.setdefault(evt, {}).setdefault(src, {}).setdefault('*', {}).update({
                "nrsegs": rn1, "rlength": rd1,
                "nhsegs": hn1, "hlength": hd1,
                "nbcorr": ncb
            })

            # F1 par frontière (optionnel)
            if bnd_f1:
                tp, fn, fp = etfbcmp_f1(srcref_s, srchyp_s, boundary_delta)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
                stats[evt][src]['*'].update({"bnd_tp": tp, "bnd_fn": fn, "bnd_fp": fp, "bnd_f1": f1})

    # ── Agrégation ────────────────────────────────────────────────────────────
    error_sum(err, events, sources, subs, subtype)
    bound_sum(stats, events, sources)

    # ── Construction du dict de résultats ─────────────────────────────────────
    def make_metrics(e, s, x='*'):
        d    = err.get(e, {}).get(s, {}).get(x, {})
        miss = d.get("miss", 0.0); tar = d.get("tar", 0.0)
        ins  = d.get("ins",  0.0); non = d.get("non", 0.0)
        fr, fa, er, r, p, f = t2m(miss, tar, ins, non)
        out = {
            "tar": tar, "non": non, "miss": miss, "ins": ins,
            "recall": r, "precision": p, "F1": f,
            "miss_rate": fr, "false_alarm_rate": fa, "error_rate": er,
        }
    
        st = stats.get(e, {}).get(s, {}).get(x, {})
        if "bnd_tp" in st:
            tp = st["bnd_tp"]; fn = st["bnd_fn"]; fp = st["bnd_fp"]
            prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            bnd_f1  = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            out["bnd_F1"] = bnd_f1   # ← recalculé depuis TP/FN/FP agrégés
            out["bnd_tp"] = tp
            out["bnd_fn"] = fn
            out["bnd_fp"] = fp
    
        return out

    results = {
        "global": make_metrics('*', '*'),
        "by_event": {evt: make_metrics(evt, '*') for evt in events},
        "by_source": {src: make_metrics('*', src) for src in sources},
        "by_event_source": {
            (evt, src): make_metrics(evt, src)
            for evt in events for src in sources
        },
    }
    per_results = compute_per_metrics(ref, hyp, events, sources)
    results["per"] = per_results
    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — POINT D'ENTRÉE PRINCIPAL : main()
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Arguments ──────────────────────────────────────────────────────────
    args = parse_args()

    if args.version:
        print(f"trackeval version {RELEASE}, patch level {PATCH} ({DATE})")
        sys.exit(0)

    if not args.reffn:
        print("Erreur : fichier ETF de référence non spécifié (--help pour l'aide).")
        sys.exit(1)
    if not args.hypfn:
        print("Erreur : fichier ETF hypothèse non spécifié (--help pour l'aide).")
        sys.exit(1)

    # Normaliser les listes de rapport (None → "sum")
    dout = normalize_report_list(args.dout or [])
    sout = normalize_report_list(args.sout or [])

    # ── 2. Fichier de sortie ───────────────────────────────────────────────────
    if args.outfn == "-":
        outf = sys.stdout
    else:
        outf = open(args.outfn, "w", encoding="utf-8")

    # ── 3. Liste d'événements ─────────────────────────────────────────────────
    if args.trace:
        print("initializing event list ...")
    events = []
    if args.evtfn:
        events.extend(load_event_list(args.evtfn))
    # Aplatir et dédupliquer les événements passés en CLI
    for e_str in args.events:
        events.extend(e_str.split(","))

    # ── 4. Chargement de la référence ─────────────────────────────────────────
    if args.trace:
        print(f"loading reference tracks from file {args.reffn} ...")
    ref = etfread(args.reffn)

    if not events:
        events = make_event_list(ref)

    # Dictionnaire des sources connues (présentes dans la référence)
    sources = {}
    for seg in ref:
        sources[seg["filename"]] = sources.get(seg["filename"], 0) + 1

    # ── 5. Filtrage UEM ───────────────────────────────────────────────────────
    if args.uemfn:
        uem = uemread(args.uemfn)
        ref = partition(ref, uem)

    # ── 6. Construction de la liste des sous-types ────────────────────────────
    subs = []
    if args.subtype:
        for seg in ref:
            if seg["subtype"] is not None and seg["subtype"] not in subs:
                subs.append(seg["subtype"])
    
    # ── 7. Chargement de l'hypothèse ──────────────────────────────────────────
    if args.trace:
        print(f"loading hypothesis tracks from file {args.hypfn} ...")
    hyp = etfread(args.hypfn)
    for seg in hyp:
        if seg["filename"] not in sources:
            raise ValueError(
                f"Pas de piste de référence pour la source '{seg['filename']}' dans {args.reffn}"
            )

    # ── 8. Structures d'accumulation ──────────────────────────────────────────
    # err[evt][src][subtype_ou_*][miss|tar|ins|non]
    # stats[evt][src][nrsegs|rlength|nhsegs|hlength|nbcorr]
    # det[subtype_ou_*][score][miss|ins]
    err   = {}
    stats = {}
    det   = {}

    # ── 9. Boucle principale de scoring ───────────────────────────────────────
    for evt in events:
        if args.trace:
            print(f"scoring {evt} ...")

        # Filtrer et trier ref/hyp pour cet événement
        eref = sorted(
            [s for s in ref if s["event"] == evt],
            key=lambda s: s["start_time"]
        )
        ehyp = sorted(
            [s for s in hyp if s["event"] == evt],
            key=lambda s: s["start_time"]
        )

        for src in sorted(sources.keys()):
            if args.trace:
                print(f"  source={src:<20s}")

            srcref = [s for s in eref if s["filename"] == src]
            srchyp = [s for s in ehyp if s["filename"] == src]

            # Limitation du nombre de segments hyp (--max-segments)
            if args.maxseg > 0 and len(srchyp) > args.maxseg:
                print(f">>>>>> WARNING >>>>> too many segments for event {evt} in file {src} "
                      f"(max={args.maxseg})", file=sys.stderr)
                print(f">>>>>> WARNING >>>>> scoring only first {args.maxseg} segments",
                      file=sys.stderr)
                srchyp = srchyp[:args.maxseg]

            # ── Calcul des erreurs de détection ───────────────────────────────
            if dout or args.detfn is not None:
                miss, tar_t, ins_t, non_t, ebuf = etfcmp(
                    srcref, srchyp, det,
                    args.subtype, subs, args.margin, args.align, outf
                )
                
                err.setdefault(evt, {}).setdefault(src, {}).setdefault('*', {
                    "miss": miss, "tar": tar_t, "ins": ins_t, "non": non_t
                })
                err[evt][src]['*'] = {"miss": miss, "tar": tar_t, "ins": ins_t, "non": non_t}

                if args.subtype:
                    for x in subs:
                        err[evt][src][x] = {
                            k: ebuf.get(x, {}).get(k, 0.0)
                            for k in ("miss", "ins", "tar", "non")
                        }

                if args.trace:
                    fr, fa, e, r, p, f = t2m(miss, tar_t, ins_t, non_t)
                    print(f"       %fr={100*fr:<9.4f}   %fa={100*fa:<9.4f}  "
                          f"%recall={100*r:<9.4f}  %precision={100*p:<9.4f}  f-measure={f:<9.4f}")
            srcref = sorted(srcref, key=lambda s: s["start_time"])
            srchyp = sorted(srchyp, key=lambda s: s["start_time"])
           
            # ── Calcul des statistiques de segmentation ────────────────────────
            if sout:
                # ───────── GLOBAL (*) ─────────
                rn1, rd1, _, _ = etfstat(srcref)
                hn1, hd1, _, _ = etfstat(srchyp)
                ncb = etfbcmp(srcref, srchyp, args.margin)
            
                stats.setdefault(evt, {}).setdefault(src, {}).setdefault('*', {}).update({
                    "nrsegs": rn1,
                    "rlength": rd1,
                    "nhsegs": hn1,
                    "hlength": hd1,
                    "nbcorr": ncb
                })
            
                # ───────── PAR SUBTYPE (STYLE) ─────────
                if args.subtype:
                    for x in subs:
            
                        srcref_sub = [s for s in srcref if s["subtype"] == x]
                        srchyp_sub = [s for s in srchyp if s["subtype"] == x]
            
                        if not srcref_sub and not srchyp_sub:
                            continue
            
                        rn1, rd1, _, _ = etfstat(srcref_sub)
                        hn1, hd1, _, _ = etfstat(srchyp_sub)
                        ncb = etfbcmp(srcref_sub, srchyp_sub, args.margin)
            
                        stats.setdefault(evt, {}).setdefault(src, {}).setdefault(x, {}).update({
                            "nrsegs": rn1,
                            "rlength": rd1,
                            "nhsegs": hn1,
                            "hlength": hd1,
                            "nbcorr": ncb
                        })
                # ── Métrique 2 : tolérance de frontière (nouveau) ───────────
                if args.bnd_f1:

                    # GLOBAL
                    tp, fn, fp = etfbcmp_f1(srcref, srchyp, args.boundary_delta)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                    stats[evt][src].setdefault('*', {}).update({
                        "bnd_tp": tp,
                        "bnd_fn": fn,
                        "bnd_fp": fp,
                        "bnd_f1": f1,
                    })
                
                    # PAR STYLE
                    if args.subtype:
                        for x in subs:
                            srcref_sub = [s for s in srcref if s["subtype"] == x]
                            srchyp_sub = [s for s in srchyp if s["subtype"] == x]
                
                            if not srcref_sub and not srchyp_sub:
                                continue
                
                            tp, fn, fp = etfbcmp_f1(srcref_sub, srchyp_sub, args.boundary_delta)
                            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                            stats[evt][src].setdefault(x, {}).update({
                                "bnd_tp": tp,
                                "bnd_fn": fn,
                                "bnd_fp": fp,
                                "bnd_f1": f1,
                            })
                    
                if args.trace:
                    r_ = 100.0 * ncb / (2 * rn1) if rn1 else 0.0
                    p_ = 100.0 * ncb / (2 * hn1) if hn1 else 0.0
                    print(f"       ref={rn1}/{rd1:.2f}    hyp={hn1}/{hd1:.2f}   "
                          f"bounds={r_:.2f}/{p_:.2f}")

    # ── 10. Agrégation et affichage des erreurs de détection ──────────────────
    if dout or args.detfn is not None:
        error_sum(err, events, sources, subs, args.subtype)
        for spec in dout:
            error_print(spec, err, events, sources, subs, args.subtype, outf)

    # ── 11. Agrégation et affichage des stats de segmentation ─────────────────
    if sout:
        bound_sum(stats, events, sources)
        for spec in sout:
            bound_print(spec, stats, events, sources, outf)
    if args.bnd_f1:
        bnd_f1_print(stats, events, sources, subs, outf)

    # ── 12. Affichage du résumé global ────────────────────────────────────────
    if any("sum" in s for s in dout) or any("sum" in s for s in sout):
        g_miss = err.get('*', {}).get('*', {}).get('*', {}).get('miss', 0.0)
        g_tar  = err.get('*', {}).get('*', {}).get('*', {}).get('tar',  0.0)
        g_ins  = err.get('*', {}).get('*', {}).get('*', {}).get('ins',  0.0)
        g_non  = err.get('*', {}).get('*', {}).get('*', {}).get('non',  0.0)
        fr, fa, e, r, p, f = t2m(g_miss, g_tar, g_ins, g_non)

        if any("sum" in s for s in dout):
            # Score ESTER 2 : moyenne du F-measure par événement
            total2 = error_by_event(
                err, events, sources, subs, args.subtype,
                ["ev"], called_from_print=False, outf=outf
            )
            n = total2["nb_evt"]
            if n:
                outf.write("\nESTER 2 results:\n\n")
                outf.write(f"\t(ESTER 2 official score)\t error_rate = {e:<10.4f}\n")
                outf.write(f"\t(ESTER 2 non-official score)\t mean F-measure = {total2['F']/n:6.4f}\n\n")

            outf.write("ESTER 1 results:\n\n")
            outf.write(f"\ttarget_time = {g_tar:<10.4f}\n")
            outf.write(f"\tnon_target_time = {g_non:<10.4f}\n")
            outf.write(f"\tmiss_time = {g_miss:<10.4f}\n")
            outf.write(f"\tinsertion_time = {g_ins:<10.4f}\n")
            outf.write(f"\terror_rate = {e:<10.4f}\n")
            outf.write(f"\tmiss_rate = {fr:<10.4f}\n")
            outf.write(f"\tfalse_alarm_rate = {fa:<10.4f}\n")
            outf.write(f"\trecall = {r:<10.4f}\n")
            outf.write(f"\tprecision = {p:<10.4f}\n")
            outf.write(f"\tF-measure = {f:<10.4f}\n")

        if any("sum" in s for s in sout):
            nr  = stats.get('*', {}).get('*', {}).get('nrsegs',  0)
            nh  = stats.get('*', {}).get('*', {}).get('nhsegs',  0)
            nc  = stats.get('*', {}).get('*', {}).get('nbcorr',  0.0)
            rad = stats['*']['*']['rlength'] / nr if nr else 0.0
            had = stats['*']['*']['hlength'] / nh if nh else 0.0
            r_  = 100.0 * nc / (2 * nr) if nr else 0.0
            p_  = 100.0 * nc / (2 * nh) if nh else 0.0
            outf.write(f"num_ref_segs = {nr}\n")
            outf.write(f"avg_ref_length = {rad:.4f}\n")
            outf.write(f"num_hyp_segs = {nh}\n")
            outf.write(f"avg_hyp_length = {had:.4f}\n")
            outf.write(f"bound_recall = {r_:.4f}\n")
            outf.write(f"bound_precision = {p_:.4f}\n")

    # ── 13. Courbes DET ───────────────────────────────────────────────────────
    if args.detfn is not None:
        pts = det_print(
            f"{args.detfn}.all.det" if args.detfn else "",
            g_tar, g_non, det.get('*', {}),
            args.detfn, args.trace
        )
        def fmt_pt(label, key):
            pt = pts.get(key, {})
            outf.write(
                f"{label} = {pt.get('val',0):.5f} "
                f"[th={pt.get('th',0):.5f} "
                f"%fr={100*pt.get('fr',0):.3f} "
                f"%fa={100*pt.get('fa',0):.3f} "
                f"recall={pt.get('r',0):.2f} "
                f"precision={pt.get('p',0):.2f}]\n"
            )
        fmt_pt("max_F_measure",             "F")
        fmt_pt("min_error_rate",            "err")
        fmt_pt("min_half_total_error_rate", "ter")
        fmt_pt("min_equal_error_rate",      "eer")

        for x in subs:
            pts_x = det_print(
                f"{args.detfn}.{x}.det",
                err.get('*', {}).get('*', {}).get(x, {}).get('tar', 0.0),
                err.get('*', {}).get('*', {}).get(x, {}).get('non', 0.0),
                det.get(x, {}), args.detfn, args.trace
            )
            for label, key in [("max_F_measure", "F"), ("min_error_rate", "err"),
                                ("min_half_total_error_rate", "ter"), ("min_equal_error_rate", "eer")]:
                pt = pts_x.get(key, {})
                outf.write(
                    f"{label}({x}) = {pt.get('val',0):.5f} "
                    f"[th={pt.get('th',0):.5f} "
                    f"%fr={100*pt.get('fr',0):.3f} "
                    f"%fa={100*pt.get('fa',0):.3f} "
                    f"recall={pt.get('r',0):.2f} "
                    f"precision={pt.get('p',0):.2f}]\n"
                )

   
            
    # ── 14. Fermeture du fichier de sortie ────────────────────────────────────
    if outf is not sys.stdout:
        outf.close()


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()