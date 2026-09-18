#!/usr/bin/env python3
"""
trackeval.py — ESTER event tracking task scoring script
Traduction Python du script Perl original de Guillaume Gravier & Sylvain Galliano (2008).

Usage CLI :
    python trackeval.py [options] ref.etf hyp.etf

Usage programmatique :
    from trackeval import run_trackeval, plot_det_curve

    results, det = run_trackeval("ref.etf", "hyp.etf", margin=0.0)
    plot_det_curve(det['*'], results['global']['tar'], results['global']['non'])
"""

import argparse
import re
import sys
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
    p.add_argument("-l", "--list",         dest="evtfn",   metavar="fn")
    p.add_argument("-e", "--event",        dest="events",  metavar="str", action="append", default=[])
    p.add_argument("-m", "--margin",       dest="margin",  metavar="f",  type=float, default=0.25)
    p.add_argument("-s", "--uem",          dest="uemfn",   metavar="fn")
    p.add_argument("-n", "--max-segments", dest="maxseg",  metavar="n",  type=int,   default=0)
    p.add_argument("-t", "--subtype",      dest="subtype", action="store_true", default=False)
    p.add_argument("-r", "--error",        dest="dout",    metavar="s",  action="append", nargs="?")
    p.add_argument("-b", "--segmentation", dest="sout",    metavar="s",  action="append", nargs="?")
    p.add_argument("-d", "--det",          dest="detfn",   metavar="fn", nargs="?", const="")
    p.add_argument("-a", "--align",        dest="align",   action="store_true", default=False)
    p.add_argument("-o", "--output",       dest="outfn",   metavar="fn", default="-")
    p.add_argument("-v", "--verbose",      dest="trace",   action="store_true", default=False)
    p.add_argument("-V", "--version",      dest="version", action="store_true", default=False)
    p.add_argument("--boundary-f1",        dest="bnd_f1",  action="store_true", default=False)
    p.add_argument("reffn", nargs="?")
    p.add_argument("hypfn",  nargs="?")
    return p.parse_args()


def normalize_report_list(raw):
    result = []
    for item in (raw or []):
        if item is None or item == "":
            result.append("sum")
        else:
            result.extend(item.split(","))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LECTURE DES FICHIERS ETF
# ══════════════════════════════════════════════════════════════════════════════

ETF_RE = re.compile(
    r'^(\S+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)'
    r'(?:\s+(-|-?\d+(?:\.\d*)?(?:e[+\-]\d+)?)(?:\s+(\S+))?)?$',
    re.IGNORECASE
)


def etfread(fn):
    segments = []
    with open(fn, "r", encoding="utf-8") as f:
        for lino, line in enumerate(f, start=1):
            line = re.sub(r';.*', '', line).strip()
            if not line:
                continue
            m = ETF_RE.match(line)
            if not m:
                raise ValueError(f"Format error in {fn} at line {lino}: {line!r}")
            source, channel, start_s, dur_s, typ, subtype, event, score_s, decision = m.groups()
            start_time = float(start_s)
            duration   = float(dur_s)
            subtype    = None if (subtype == "-" or subtype.lower() == "na") else subtype
            if score_s is None or score_s == "-" or score_s.lower() == "na":
                score = None
            else:
                score = float(score_s)
            if decision is None:
                decision = "true"
            segments.append({
                "filename":   source,
                "channel":    channel,
                "start_time": start_time,
                "duration":   duration,
                "end_time":   start_time + duration,
                "type":       typ,
                "subtype":    subtype,
                "event":      event,
                "score":      score,
                "decision":   decision,
            })
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UEM
# ══════════════════════════════════════════════════════════════════════════════

def uemread(fn):
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
    result = []
    for seg in etf_segs:
        st = seg["start_time"]
        et = seg["end_time"]
        buf = sorted(
            [u for u in uem_regions if u["filename"] == seg["filename"]],
            key=lambda u: u["start_time"]
        )
        for uzone in buf:
            if uzone["end_time"] < st:
                continue
            if uzone["start_time"] > et:
                break
            sst  = max(st, uzone["start_time"])
            set_ = min(et, uzone["end_time"])
            result.append({**seg, "start_time": sst, "duration": set_ - sst, "end_time": set_})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LISTE D'ÉVÉNEMENTS
# ══════════════════════════════════════════════════════════════════════════════

def load_event_list(fn):
    events = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(r';.*', '', line).strip()
            if not line:
                continue
            events.append(line.split()[0])
    return events


def make_event_list(segments):
    return list({seg["event"] for seg in segments})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MÉTRIQUES : t2m()
# ══════════════════════════════════════════════════════════════════════════════

def t2m(miss, tar, ins, non):
    """Times to Measures — convertit durées brutes en métriques."""
    correct  = tar - miss
    detected = correct + ins
    fr = miss / tar if tar > 0 else 0.0
    fa = ins  / non if non > 0 else 0.0
    e  = (miss + ins) / (tar + non) if (tar + non) > 0 else 0.0
    if tar > 0:
        r = correct / tar
        p = correct / detected if detected > 0 else 0.0
    else:
        r = 1.0
        p = correct / detected if detected > 0 else 1.0
    f = (2 * r * p) / (r + p) if (r + p) > 0 else 0.0
    return fr, fa, e, r, p, f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — COMPARAISON ETF : etfcmp()
# ══════════════════════════════════════════════════════════════════════════════

def etfcmp(ref_segs, hyp_segs, det, subtype_mode, subs, margin, align, outf):
    EPS  = 1e-10
    miss = ins = tar = non = 0.0
    if '*' not in det:
        det['*'] = {}
    ebuf = {}
    if subtype_mode:
        for x in subs:
            ebuf[x] = {"miss": 0.0, "ins": 0.0, "tar": 0.0, "non": 0.0}

    nhyp  = len(hyp_segs)
    hypid = 0

    for rseg in ref_segs:
        rst  = rseg["start_time"] + margin
        ret  = rseg["start_time"] + rseg["duration"] - margin
        dur  = ret - rst
        rdec = rseg["decision"]
        x    = rseg["subtype"]

        if align:
            outf.write(f"ref=({rseg['start_time']:.4f}, {rseg['end_time']:.4f}, {rdec[0]})\n")

        if dur <= 0:
            continue

        while hypid < nhyp and hyp_segs[hypid]["end_time"] < rst + EPS:
            hypid += 1

        st = rst
        tmp_hypid = hypid

        while tmp_hypid < nhyp and hyp_segs[tmp_hypid]["start_time"] < ret - EPS:
            hseg  = hyp_segs[tmp_hypid]
            hst   = hseg["start_time"]
            het   = hseg["end_time"]
            hdec  = hseg["decision"]
            score = hseg["score"]

            _det_key = score
            if _det_key not in det['*']:
                det['*'][_det_key] = {"miss": 0.0, "ins": 0.0}
            if subtype_mode and x is not None:
                if x not in det:
                    det[x] = {}
                if _det_key not in det[x]:
                    det[x][_det_key] = {"miss": 0.0, "ins": 0.0}

            # Zone initiale non couverte → MISS
            if hst > st and re.search(r't', rdec, re.IGNORECASE):
                d = hst - st
                if align:
                    outf.write(f"    miss={d:<9.4f}    [{st:10.4f},{hst:10.4f}]"
                               f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n")
                miss += d
                det['*'].setdefault('offset', 0.0)
                det['*']['offset'] += d
                if subtype_mode and x is not None:
                    ebuf[x]["miss"] += d
                    det.setdefault(x, {}).setdefault('offset', 0.0)
                    det[x]['offset'] += d

            # Intersection
            a = max(hst, st)
            b = min(het, ret)
            d = b - a

            if re.search(r't', rdec, re.IGNORECASE):
                if not re.search(r't', hdec, re.IGNORECASE):
                    if align:
                        outf.write(f"    miss={d:<9.4f}    [{a:10.4f},{b:10.4f}]"
                                   f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n")
                    miss += d
                    if subtype_mode and x is not None:
                        ebuf[x]["miss"] += d
                det['*'][_det_key]["miss"] += d
                if subtype_mode and x is not None and _det_key is not None:
                    det[x][_det_key]["miss"] += d
            else:
                if not re.search(r'f', hdec, re.IGNORECASE):
                    if align:
                        outf.write(f"    insert={d:<9.4f}    [{a:10.4f},{b:10.4f}]"
                                   f"      hyp=({hst:.4f}, {het:.4f}, {hdec[0]})\n")
                    ins += d
                    if subtype_mode and x is not None:
                        ebuf[x]["ins"] += d
                det['*'][_det_key]["ins"] += d
                if subtype_mode and x is not None and _det_key is not None:
                    det[x][_det_key]["ins"] += d

            st = b
            if het > ret:
                break
            tmp_hypid += 1

        # Reste non couvert → MISS
        if st < ret - EPS and re.search(r't', rdec, re.IGNORECASE):
            d = ret - st
            if align:
                outf.write(f"    miss={d:<9.4f}    [{st:10.4f},{ret:10.4f}]\n")
            miss += d
            det['*'].setdefault('offset', 0.0)
            det['*']['offset'] += d
            if subtype_mode and x is not None:
                ebuf[x]["miss"] += d
                det.setdefault(x, {}).setdefault('offset', 0.0)
                det[x]['offset'] += d

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

def etfstat(segments):
    n   = [0, 0]
    dur = [0.0, 0.0]
    for seg in segments:
        i = 0 if re.search(r't', seg["decision"], re.IGNORECASE) else 1
        n[i]   += 1
        dur[i] += seg["duration"]
    return n[0], dur[0], n[1], dur[1]


def etfbcmp(ref_segs, hyp_segs, delta):
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
    n     = 0
    for rt in rbounds:
        ibest = -1
        while ihyp < nhyp and hbounds[ihyp] <= rt + delta:
            d = abs(rt - hbounds[ihyp])
            if d <= delta and (ibest < 0 or d < abs(rt - hbounds[ibest])):
                ibest = ihyp
            ihyp += 1
        if ibest >= 0:
            n    += 1
            ihyp  = ibest + 1
        if ihyp == nhyp:
            break
    return n


def etfbcmp_f1(ref_segs, hyp_segs, delta):
    rbounds = sorted(
        [t for seg in ref_segs
         if re.search(r't', seg["decision"], re.IGNORECASE)
         for t in (seg["start_time"], seg["end_time"])]
    )
    hbounds = sorted(
        [t for seg in hyp_segs
         if re.search(r't', seg["decision"], re.IGNORECASE)
         for t in (seg["start_time"], seg["end_time"])]
    )
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

def error_sum(err, events, sources, subs, subtype_mode):
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
                    err['*']['*'][x][k] += v
                    err['*'][src][x][k] += v
                    err[evt]['*'][x][k] += v


def bound_sum(stats, events, sources):
    fields     = ("nrsegs", "rlength", "nhsegs", "hlength", "nbcorr")
    bnd_fields = ("bnd_tp", "bnd_fn", "bnd_fp")   # ← ajout

    all_fields = fields + bnd_fields

    for evt in events:
        for src in sources:
            for x, xdata in stats.get(evt, {}).get(src, {}).items():
                if not isinstance(xdata, dict):
                    continue
                for f in all_fields:          # ← tous les champs
                    v = xdata.get(f, 0.0)
                    stats.setdefault('*', {}).setdefault('*', {}).setdefault(x, {})
                    stats['*']['*'][x][f] = stats['*']['*'][x].get(f, 0.0) + v
                    stats.setdefault('*', {}).setdefault(src, {}).setdefault(x, {})
                    stats['*'][src][x][f] = stats['*'][src][x].get(f, 0.0) + v
                    stats.setdefault(evt, {}).setdefault('*', {}).setdefault(x, {})
                    stats[evt]['*'][x][f] = stats[evt]['*'][x].get(f, 0.0) + v

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — AFFICHAGE RAPPORTS DE DÉTECTION
# ══════════════════════════════════════════════════════════════════════════════

def error_by_event(err, events, sources, subs, subtype_mode, who, called_from_print, outf):
    stab  = sorted(sources.keys()) if any(w.startswith("so") for w in who) else ['*']
    etab  = events                 if any(w.startswith("ev") for w in who) else ['*']
    xtab  = subs if (subtype_mode and any(w.startswith("sub") for w in who)) else ['*']
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
                    if s != '*': outf.write(f" {s:<40s} ")
                    if e != '*': outf.write(f" {e:<30s} ")
                    if x != '*': outf.write(f" {x:<10s} ")
                    outf.write(f" | {tar:9.2f} {non_:9.2f} | {miss:8.2f}  {ins_:8.2f} |")
                    outf.write(f" {100*er:7.3f}  {100*fr:7.3f}  {100*fa:7.3f} |")
                    outf.write(f" {100*r:7.3f}  {100*p:7.3f}  {f:6.4f} |\n")
                total["miss"]   += miss; total["tar"]  += tar
                total["ins"]    += ins_; total["non"]  += non_
                total["fr"]     += fr;   total["fa"]   += fa
                total["err"]    += er;   total["r"]    += r
                total["p"]      += p;    total["F"]    += f
                total["nb_evt"] += 1
    return total


def error_print(spec, err, events, sources, subs, subtype_mode, outf):
    who = [tok for tok in re.split(r'\+', spec) if re.match(r'^(so|ev|sub)', tok)]
    if not who:
        return
    hl  = (42 if any(w.startswith("so") for w in who) else 0) + \
          (32 if any(w.startswith("ev") for w in who) else 0) + \
          (12 if subtype_mode and any(w.startswith("sub") for w in who) else 0)
    line_len = hl + 100
    outf.write("\n\n    " + "-" * (line_len + 1) + "\n    |")
    if any(w.startswith("so")  for w in who): outf.write(f" {'source':<40s} ")
    if any(w.startswith("ev")  for w in who): outf.write(f" {'event':<30s} ")
    if subtype_mode and any(w.startswith("sub") for w in who): outf.write(f" {'subtype':<10s} ")
    outf.write(f" | {'tar.':>9s} {'non':>9s} | {'miss':>8s}  {'ins':>8s} | "
               f"{'%err':>7s}  {'%miss':>7s}  {'%fa':>7s} | "
               f"{'%rec':>7s}  {'%prec':>7s}  {'F':>6s} |\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")
    total = error_by_event(err, events, sources, subs, subtype_mode, who,
                           called_from_print=True, outf=outf)
    outf.write("    " + "=" * (line_len + 1) + "\n")
    fmt_lbl = f"    | {{:<{hl}s}}|"
    n = total["nb_evt"]
    if n:
        outf.write(fmt_lbl.format("Average"))
        outf.write(f" {total['tar']/n:9.2f} {total['non']/n:9.2f} | "
                   f"{total['miss']/n:8.2f}  {total['ins']/n:8.2f} |"
                   f" {100*total['err']/n:7.3f}  {100*total['fr']/n:7.3f}  {100*total['fa']/n:7.3f} |"
                   f" {100*total['r']/n:7.3f}  {100*total['p']/n:7.3f}  {total['F']/n:6.4f} |\n")
    g_miss = err['*']['*']['*']['miss']; g_tar = err['*']['*']['*']['tar']
    g_ins  = err['*']['*']['*']['ins'];  g_non = err['*']['*']['*']['non']
    fr, fa, e, r, p, f = t2m(g_miss, g_tar, g_ins, g_non)
    outf.write(fmt_lbl.format("Summary"))
    outf.write(f" {g_tar:9.2f} {g_non:9.2f} | {g_miss:8.2f}  {g_ins:8.2f} |"
               f" {e:7.3f}  {100*fr:7.3f}  {100*fa:7.3f} |"
               f" {100*r:7.3f}  {100*p:7.3f}  {f:6.4f} |\n")
    outf.write("    " + "=" * (line_len + 1) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — AFFICHAGE RAPPORTS DE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def bound_print(spec, stats, events, sources, outf):
    who = [tok for tok in re.split(r'\+', spec) if re.match(r'^(so|ev|sub)', tok)]
    if not who:
        return
    stab = sorted(sources.keys()) if any(w.startswith("so") for w in who) else ['*']
    etab = events                  if any(w.startswith("ev") for w in who) else ['*']
    all_styles = {k for evt in events for src in sources
                  for k, v in stats.get(evt, {}).get(src, {}).items()
                  if isinstance(v, dict) and k != '*'}
    xtab = sorted(all_styles) if any(w.startswith("sub") for w in who) and all_styles else ['*']
    hl   = max((42 if any(w.startswith("so") for w in who) else 0) +
               (32 if any(w.startswith("ev") for w in who) else 0) +
               (12 if any(w.startswith("sub") for w in who) else 0) - 1, 0)
    line_len = hl + 60
    outf.write("\n\n    " + "-" * (line_len + 1) + "\n")
    outf.write("    |" + " " * (hl + 1) +
               "|        ref        |        hyp        |    boundaries    |\n    |")
    if any(w.startswith("so")  for w in who): outf.write(f" {'source':<40s} ")
    if any(w.startswith("ev")  for w in who): outf.write(f" {'event':<30s} ")
    if any(w.startswith("sub") for w in who): outf.write(f" {'subtype':<10s} ")
    outf.write("|  nsegs     length |  nsegs     length |   %rec     %prec |\n")
    outf.write("    " + "-" * (line_len + 1) + "\n")
    sumv   = dict(nr=0.0, rad=0.0, nh=0.0, had=0.0, r=0.0, p=0.0)
    n_rows = 0
    for s in stab:
        for e in etab:
            for x in xtab:
                row = stats.get(e, {}).get(s, {}).get(x, {})
                if not row:
                    continue
                outf.write("    |")
                if s != '*': outf.write(f" {s:<40s} ")
                if e != '*': outf.write(f" {e:<30s} ")
                if any(w.startswith("sub") for w in who): outf.write(f" {x:<10s} ")
                nr  = row.get("nrsegs", 0); nh  = row.get("nhsegs", 0)
                rad = row.get("rlength", 0.0) / nr if nr else 0.0
                had = row.get("hlength", 0.0) / nh if nh else 0.0
                nc  = row.get("nbcorr", 0.0)
                r   = 100.0 * nc / (2 * nr) if nr else 0.0
                p   = 100.0 * nc / (2 * nh) if nh else 0.0
                outf.write(f"| {int(nr):6d}   {rad:8.2f} | {int(nh):6d}   {had:8.2f} | {r:7.3f}  {p:7.3f} |\n")
                sumv["nr"] += nr; sumv["rad"] += rad
                sumv["nh"] += nh; sumv["had"] += had
                sumv["r"]  += r;  sumv["p"]   += p
                n_rows += 1
    outf.write("    " + "=" * (line_len + 1) + "\n")
    fmt_lbl = f"    | {{:<{hl}s}}|"
    if n_rows:
        outf.write(fmt_lbl.format("Average"))
        outf.write(f"   {sumv['nr']/n_rows:7.2f} {sumv['rad']/n_rows:7.2f} |"
                   f"   {sumv['nh']/n_rows:7.2f} {sumv['had']/n_rows:7.2f} |"
                   f" {sumv['r']/n_rows:7.3f}  {sumv['p']/n_rows:7.3f} |\n")
    g   = stats.get('*', {}).get('*', {}).get('*', {})
    g_nr = g.get("nrsegs", 0); g_nh = g.get("nhsegs", 0)
    g_r  = 100.0 * g.get("nbcorr", 0.0) / (2 * g_nr) if g_nr else 0.0
    g_p  = 100.0 * g.get("nbcorr", 0.0) / (2 * g_nh) if g_nh else 0.0
    outf.write(fmt_lbl.format("Summary"))
    outf.write(f" {int(g_nr):6d}   {g.get('rlength',0)/g_nr if g_nr else 0:8.2f} |"
               f" {int(g_nh):6d}   {g.get('hlength',0)/g_nh if g_nh else 0:8.2f} |"
               f" {g_r:7.3f}  {g_p:7.3f} |\n")
    outf.write("    " + "=" * (line_len + 1) + "\n")


def bnd_f1_print(stats, events, sources, subs, outf):
    xtab     = ['*'] + list(subs) if subs else ['*']
    line_len = 74
    for x in xtab:
        label = "global" if x == '*' else x
        outf.write(f"\n\n    " + "-" * line_len + f"\n    | F1 par tolérance de frontière — {label}\n")
        outf.write("    " + "-" * line_len + "\n")
        outf.write(f"    | {'event':<20s} {'source':<30s} | {'TP':>6s} {'FN':>6s} {'FP':>6s} |"
                   f" {'recall':>8s} {'precision':>9s} {'F1':>8s} |\n")
        outf.write("    " + "-" * line_len + "\n")
        sum_tp = sum_fn = sum_fp = 0
        for evt in sorted(events):
            for src in sorted(sources.keys()):
                s  = stats.get(evt, {}).get(src, {}).get(x, {})
                tp = s.get("bnd_tp", 0); fn = s.get("bnd_fn", 0); fp = s.get("bnd_fp", 0)
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1   = 2*rec*prec / (rec+prec) if (rec+prec) > 0 else 0.0
                outf.write(f"    | {evt:<20s} {src:<30s} | {tp:>6d} {fn:>6d} {fp:>6d} |"
                           f" {100*rec:>8.3f} {100*prec:>9.3f} {f1:>8.4f} |\n")
                sum_tp += tp; sum_fn += fn; sum_fp += fp
        outf.write("    " + "=" * line_len + "\n")
        g_r = sum_tp/(sum_tp+sum_fn) if (sum_tp+sum_fn) > 0 else 0.0
        g_p = sum_tp/(sum_tp+sum_fp) if (sum_tp+sum_fp) > 0 else 0.0
        g_f = 2*g_r*g_p/(g_r+g_p)   if (g_r+g_p) > 0 else 0.0
        outf.write(f"    | {'Summary':<20s} {'':30s} | {sum_tp:>6d} {sum_fn:>6d} {sum_fp:>6d} |"
                   f" {100*g_r:>8.3f} {100*g_p:>9.3f} {g_f:>8.4f} |\n")
        outf.write("    " + "=" * line_len + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — COURBES DET : det_print() + plot_det_curve()
# ══════════════════════════════════════════════════════════════════════════════

def det_print(ofn, tar, non, det_tab, detfn, trace):
    """Calcule les points DET et retourne les points singuliers (F, err, ter, eer)."""
    if trace:
        print("computing DET points ...")

    det_file = None
    if detfn is not None and detfn != "":
        det_file = open(ofn, "w", encoding="utf-8")
        det_file.write("# threshold fr fa error recall precision F-measure\n")

    pts      = {}
    ins_acc  = 0.0
    miss_acc = det_tab.pop('offset', 0.0)

    valid_keys = [k for k in det_tab if k is not None]

    # Passe 1 — accumulation FA (score décroissant)
    for th in sorted(valid_keys, reverse=True):
        ins_acc           += det_tab[th]["ins"]
        det_tab[th]["ins"] = ins_acc

    # Passe 2 — accumulation FR + métriques
    for th in sorted(valid_keys):
        miss_acc += det_tab[th]["miss"]
        ins_acc   = det_tab[th]["ins"]
        fr, fa, e, r, p, f = t2m(miss_acc, tar, ins_acc, non)

        if det_file:
            det_file.write(f"{th:.6f} {fr:.6f} {fa:.6f} {e:.6f} {r:.6f} {p:.6f} {f:.6f}\n")

        if not pts:
            for key in ("F", "err", "ter", "eer"):
                pts[key] = {"th": th, "fa": fa, "fr": fr, "r": r, "p": p}
            pts["F"]["val"]    = f
            pts["err"]["val"]  = e
            pts["ter"]["val"]  = fr + fa
            pts["eer"]["val"]  = (fr + fa) / 2
            pts["eer"]["diff"] = abs(fr - fa)
        else:
            if f         > pts["F"]["val"]:   pts["F"].update(val=f,        th=th, fa=fa, fr=fr, r=r, p=p)
            if e         < pts["err"]["val"]: pts["err"].update(val=e,       th=th, fa=fa, fr=fr, r=r, p=p)
            if fr+fa     < pts["ter"]["val"]: pts["ter"].update(val=fr+fa,   th=th, fa=fa, fr=fr, r=r, p=p)
            if abs(fr-fa)< pts["eer"]["diff"]:
                pts["eer"].update(val=(fr+fa)/2, diff=abs(fr-fa), th=th, fa=fa, fr=fr, r=r, p=p)

    if det_file:
        det_file.close()

    return pts


# ══════════════════════════════════════════════════════════════════════════════
# NOUVELLE FONCTION : plot_det_curve()
# ══════════════════════════════════════════════════════════════════════════════

def plot_det_curve(det_tabs, tar_vals, non_vals, labels=None,
                   colors=None, title="DET Curve",
                   save_path=None, show=True):
    """
    Trace une ou plusieurs courbes DET sur le même graphe.

    La courbe DET est le standard de l'évaluation des systèmes de détection.
    Elle trace le taux de miss (FR) en fonction du taux de fausse alarme (FA)
    pour tous les seuils de décision possibles.
    Les deux axes sont sur une échelle normale (normal deviate scale) ce qui
    donne une droite pour un détecteur gaussien, et facilite la comparaison
    entre systèmes.

    Paramètres
    ----------
    det_tabs : dict ou list de dict
        Table(s) DET. Chaque table = det['*'] produit par run_trackeval().
        Format : {score: {"miss": float, "ins": float}, "offset": float}
        Peut être un seul dict (un système) ou une liste de dicts (plusieurs).

    tar_vals : float ou list de float
        Durée(s) totale(s) de cible. Correspond à results['global']['tar'].

    non_vals : float ou list de float
        Durée(s) totale(s) de non-cible. Correspond à results['global']['non'].

    labels : list de str, optionnel
        Noms des systèmes pour la légende. Ex. ["CTC", "MFA_w2v", "MFA_asr"].

    colors : list de str, optionnel
        Couleurs des courbes. Défaut : palette matplotlib.

    title : str
        Titre du graphe.

    save_path : str, optionnel
        Chemin de sauvegarde (ex. "det_curve.pdf"). None = pas de sauvegarde.

    show : bool
        Afficher le graphe interactivement (plt.show()).

    Retourne
    --------
    fig, ax : objets matplotlib pour personnalisation ultérieure.

    Exemple
    -------
    # Un seul système
    results, det = run_trackeval("ref.etf", "hyp.etf", margin=0.0)
    plot_det_curve(det['*'], results['global']['tar'], results['global']['non'],
                   labels=["MonSystème"], save_path="det.pdf")

    # Plusieurs systèmes
    systems = {
        "CTC":     run_trackeval("ref.etf", "hyp_ctc.etf"),
        "MFA_w2v": run_trackeval("ref.etf", "hyp_mfa.etf"),
    }
    plot_det_curve(
        det_tabs  = [v[1]['*'] for v in systems.values()],
        tar_vals  = [v[0]['global']['tar'] for v in systems.values()],
        non_vals  = [v[0]['global']['non'] for v in systems.values()],
        labels    = list(systems.keys()),
        save_path = "det_comparison.pdf"
    )
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import norm as sp_norm
    except ImportError:
        raise ImportError("matplotlib et scipy sont requis pour plot_det_curve().\n"
                          "Installez-les : pip install matplotlib scipy")

    # ── Normalisation des entrées (un seul système ou plusieurs) ─────────────
    if isinstance(det_tabs, dict):
        det_tabs = [det_tabs]
        tar_vals = [tar_vals]
        non_vals = [non_vals]

    n_sys = len(det_tabs)
    if labels is None:
        labels = [f"system {i+1}" for i in range(n_sys)]
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors     = [c['color'] for c in list(prop_cycle)[:n_sys]]

    # ── Paramètres des axes en % (lisibles) ──────────────────────────────────
    ticks_pct = [0.5, 1, 2, 5, 10, 20, 40]
    ticks_nd  = [sp_norm.ppf(t / 100) for t in ticks_pct]
    tick_lbl  = [f"{t}%" for t in ticks_pct]
    EPS       = 1e-6   # anti-ppf(0) ou ppf(1)

    fig, ax = plt.subplots(figsize=(6, 6))

    # ── Diagonale EER (FR = FA) ───────────────────────────────────────────────
    lims = [sp_norm.ppf(0.005), sp_norm.ppf(0.45)]
    ax.plot(lims, lims, color='gray', linewidth=0.8,
            linestyle='--', alpha=0.5, label='EER line')

    # ── Tracé de chaque système ───────────────────────────────────────────────
    for det_tab, tar, non, label, color in zip(det_tabs, tar_vals, non_vals, labels, colors):

        # Copie locale pour ne pas modifier l'original
        tab      = {k: dict(v) for k, v in det_tab.items()
                    if k != "offset" and k is not None}
        miss_off = det_tab.get("offset", 0.0)

        # Passe 1 — accumulation FA (décroissant)
        ins_acc = 0.0
        for th in sorted(tab.keys(), reverse=True):
            ins_acc      += tab[th]["ins"]
            tab[th]["ins"] = ins_acc

        # Passe 2 — accumulation FR + collecte points
        fr_list = []; fa_list = []
        f1_list = []; er_list = []; th_list = []
        miss_acc = miss_off

        for th in sorted(tab.keys()):
            miss_acc += tab[th]["miss"]
            ins_curr  = tab[th]["ins"]
            fr, fa, e, r, p, f = t2m(miss_acc, tar, ins_curr, non)
            fr_list.append(fr); fa_list.append(fa)
            f1_list.append(f);  er_list.append(e); th_list.append(th)

        if not fr_list:
            continue

        fr_arr = [sp_norm.ppf(max(min(v, 1-EPS), EPS)) for v in fr_list]
        fa_arr = [sp_norm.ppf(max(min(v, 1-EPS), EPS)) for v in fa_list]

        # Courbe principale
        ax.plot(fa_arr, fr_arr, label=label, color=color, linewidth=1.8)

        # ── Points singuliers ─────────────────────────────────────────────────

        # EER — point où FR ≈ FA
        eer_idx = min(range(len(fr_list)), key=lambda i: abs(fr_list[i] - fa_list[i]))
        eer_val = (fr_list[eer_idx] + fa_list[eer_idx]) / 2

        # Max F1
        f1_idx  = max(range(len(f1_list)), key=lambda i: f1_list[i])

        # Min Error Rate
        er_idx  = min(range(len(er_list)), key=lambda i: er_list[i])

        def _nd(v):
            return sp_norm.ppf(max(min(v, 1-EPS), EPS))

        ax.plot(_nd(fa_list[eer_idx]), _nd(fr_list[eer_idx]),
                marker='o', color=color, markersize=8, markeredgecolor='white',
                label=f"EER={100*eer_val:.1f}% ({label})")

        ax.plot(_nd(fa_list[f1_idx]), _nd(fr_list[f1_idx]),
                marker='s', color=color, markersize=8, markeredgecolor='white',
                label=f"maxF1={f1_list[f1_idx]:.3f} ({label})")

        ax.plot(_nd(fa_list[er_idx]), _nd(fr_list[er_idx]),
                marker='^', color=color, markersize=8, markeredgecolor='white',
                label=f"minErr={100*er_list[er_idx]:.1f}% ({label})")

    # ── Formatage des axes ────────────────────────────────────────────────────
    ax.set_xticks(ticks_nd); ax.set_xticklabels(tick_lbl, fontsize=9)
    ax.set_yticks(ticks_nd); ax.set_yticklabels(tick_lbl, fontsize=9)
    ax.set_xlabel("False Alarm Rate (%)", fontsize=11)
    ax.set_ylabel("Miss Rate (%)",        fontsize=11)
    ax.set_title(title,                   fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(fontsize=7, loc='upper right')

    # Limites de la zone d'affichage
    lo = sp_norm.ppf(0.003)
    hi = sp_norm.ppf(0.60)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DET curve saved → {save_path}")

    if show:
        plt.show()

    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — run_trackeval() : API PROGRAMMATIQUE
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

    det : dict
        Table DET brute. Utiliser det['*'] pour plot_det_curve().
        Format : {score: {"miss": float, "ins": float}, "offset": float}

    Exemple
    -------
    results, det = run_trackeval("ref.etf", "hyp.etf", margin=0.0)

    # Métriques globales
    print(f"F1       = {results['global']['F1']:.4f}")
    print(f"recall   = {results['global']['recall']:.4f}")
    print(f"precision= {results['global']['precision']:.4f}")

    # Par phonème
    for ph, m in results['by_event'].items():
        print(f"{ph:5s} F1={m['F1']:.3f}  recall={m['recall']:.3f}")

    # Courbe DET
    plot_det_curve(det['*'], results['global']['tar'], results['global']['non'])
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

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — POINT D'ENTRÉE CLI : main()
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.version:
        print(f"trackeval version {RELEASE}, patch level {PATCH} ({DATE})")
        sys.exit(0)

    if not args.reffn:
        print("Erreur : fichier ETF de référence non spécifié."); sys.exit(1)
    if not args.hypfn:
        print("Erreur : fichier ETF hypothèse non spécifié.");    sys.exit(1)

    dout = normalize_report_list(args.dout or [])
    sout = normalize_report_list(args.sout or [])

    outf = sys.stdout if args.outfn == "-" else open(args.outfn, "w", encoding="utf-8")

    # ── Chargement ────────────────────────────────────────────────────────────
    if args.trace: print("initializing event list ...")
    events = []
    if args.evtfn:
        events.extend(load_event_list(args.evtfn))
    for e_str in args.events:
        events.extend(e_str.split(","))

    if args.trace: print(f"loading reference tracks from {args.reffn} ...")
    ref = etfread(args.reffn)
    if not events:
        events = make_event_list(ref)

    sources = {}
    for seg in ref:
        sources[seg["filename"]] = sources.get(seg["filename"], 0) + 1

    if args.uemfn:
        ref = partition(ref, uemread(args.uemfn))

    subs = []
    if args.subtype:
        for seg in ref:
            if seg["subtype"] is not None and seg["subtype"] not in subs:
                subs.append(seg["subtype"])

    if args.trace: print(f"loading hypothesis tracks from {args.hypfn} ...")
    hyp = etfread(args.hypfn)
    for seg in hyp:
        if seg["filename"] not in sources:
            raise ValueError(f"No reference for source '{seg['filename']}' in {args.reffn}")

    # ── Structures internes ───────────────────────────────────────────────────
    err   = {}
    stats = {}
    det   = {}

    # ── Boucle principale ─────────────────────────────────────────────────────
    for evt in events:
        if args.trace: print(f"scoring {evt} ...")
        eref = sorted([s for s in ref if s["event"] == evt], key=lambda s: s["start_time"])
        ehyp = sorted([s for s in hyp if s["event"] == evt], key=lambda s: s["start_time"])

        for src in sorted(sources.keys()):
            if args.trace: print(f"  source={src:<20s}")
            srcref = [s for s in eref if s["filename"] == src]
            srchyp = [s for s in ehyp if s["filename"] == src]

            if args.maxseg > 0 and len(srchyp) > args.maxseg:
                print(f"WARNING: too many segments for {evt}/{src} (max={args.maxseg})", file=sys.stderr)
                srchyp = srchyp[:args.maxseg]

            if dout or args.detfn is not None:
                miss, tar_t, ins_t, non_t, ebuf = etfcmp(
                    srcref, srchyp, det, args.subtype, subs, args.margin, args.align, outf)
                err.setdefault(evt, {}).setdefault(src, {})['*'] = {
                    "miss": miss, "tar": tar_t, "ins": ins_t, "non": non_t}
                if args.subtype:
                    for x in subs:
                        err[evt][src][x] = {k: ebuf.get(x, {}).get(k, 0.0)
                                            for k in ("miss", "ins", "tar", "non")}
                if args.trace:
                    fr, fa, e, r, p, f = t2m(miss, tar_t, ins_t, non_t)
                    print(f"       %fr={100*fr:<9.4f}  %fa={100*fa:<9.4f}  "
                          f"%rec={100*r:<9.4f}  %prec={100*p:<9.4f}  F={f:<9.4f}")

            srcref_s = sorted(srcref, key=lambda s: s["start_time"])
            srchyp_s = sorted(srchyp, key=lambda s: s["start_time"])

            if sout:
                rn1, rd1, _, _ = etfstat(srcref_s)
                hn1, hd1, _, _ = etfstat(srchyp_s)
                ncb = etfbcmp(srcref_s, srchyp_s, args.margin)
                stats.setdefault(evt, {}).setdefault(src, {}).setdefault('*', {}).update({
                    "nrsegs": rn1, "rlength": rd1, "nhsegs": hn1, "hlength": hd1, "nbcorr": ncb})
                if args.subtype:
                    for x in subs:
                        sr = [s for s in srcref_s if s["subtype"] == x]
                        sh = [s for s in srchyp_s if s["subtype"] == x]
                        if not sr and not sh: continue
                        rn1x, rd1x, _, _ = etfstat(sr)
                        hn1x, hd1x, _, _ = etfstat(sh)
                        ncbx = etfbcmp(sr, sh, args.margin)
                        stats[evt][src].setdefault(x, {}).update({
                            "nrsegs": rn1x, "rlength": rd1x,
                            "nhsegs": hn1x, "hlength": hd1x, "nbcorr": ncbx})
                if args.bnd_f1:
                    tp, fn, fp = etfbcmp_f1(srcref_s, srchyp_s, args.boundary_delta)
                    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
                    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
                    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
                    stats[evt][src]['*'].update({"bnd_tp": tp, "bnd_fn": fn, "bnd_fp": fp, "bnd_f1": f1})
                    if args.subtype:
                        for x in subs:
                            sr = [s for s in srcref_s if s["subtype"] == x]
                            sh = [s for s in srchyp_s if s["subtype"] == x]
                            if not sr and not sh: continue
                            tp, fn, fp = etfbcmp_f1(sr, sh, args.boundary_delta)
                            prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
                            rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
                            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
                            stats[evt][src].setdefault(x, {}).update(
                                {"bnd_tp": tp, "bnd_fn": fn, "bnd_fp": fp, "bnd_f1": f1})
                if args.trace:
                    r_ = 100.0 * ncb / (2 * rn1) if rn1 else 0.0
                    p_ = 100.0 * ncb / (2 * hn1) if hn1 else 0.0
                    print(f"       ref={rn1}/{rd1:.2f}  hyp={hn1}/{hd1:.2f}  bounds={r_:.2f}/{p_:.2f}")

    # ── Agrégation + affichage ────────────────────────────────────────────────
    if dout or args.detfn is not None:
        error_sum(err, events, sources, subs, args.subtype)
        for spec in dout:
            error_print(spec, err, events, sources, subs, args.subtype, outf)

    if sout:
        bound_sum(stats, events, sources)
        for spec in sout:
            bound_print(spec, stats, events, sources, outf)

    if args.bnd_f1:
        bnd_f1_print(stats, events, sources, subs, outf)

    # ── Résumé global ─────────────────────────────────────────────────────────
    if any("sum" in s for s in dout) or any("sum" in s for s in sout):
        g_miss = err.get('*',{}).get('*',{}).get('*',{}).get('miss', 0.0)
        g_tar  = err.get('*',{}).get('*',{}).get('*',{}).get('tar',  0.0)
        g_ins  = err.get('*',{}).get('*',{}).get('*',{}).get('ins',  0.0)
        g_non  = err.get('*',{}).get('*',{}).get('*',{}).get('non',  0.0)
        fr, fa, e, r, p, f = t2m(g_miss, g_tar, g_ins, g_non)

        if any("sum" in s for s in dout):
            total2 = error_by_event(err, events, sources, subs, args.subtype,
                                    ["ev"], called_from_print=False, outf=outf)
            n = total2["nb_evt"]
            if n:
                outf.write(f"\nESTER 2 results:\n\n"
                           f"\t(official)     error_rate   = {e:<10.4f}\n"
                           f"\t(non-official) mean F-measure = {total2['F']/n:6.4f}\n\n")
            outf.write(f"ESTER 1 results:\n\n"
                       f"\ttarget_time       = {g_tar:<10.4f}\n"
                       f"\tnon_target_time   = {g_non:<10.4f}\n"
                       f"\tmiss_time         = {g_miss:<10.4f}\n"
                       f"\tinsertion_time    = {g_ins:<10.4f}\n"
                       f"\terror_rate        = {e:<10.4f}\n"
                       f"\tmiss_rate         = {fr:<10.4f}\n"
                       f"\tfalse_alarm_rate  = {fa:<10.4f}\n"
                       f"\trecall            = {r:<10.4f}\n"
                       f"\tprecision         = {p:<10.4f}\n"
                       f"\tF-measure         = {f:<10.4f}\n")

        if any("sum" in s for s in sout):
            g  = stats.get('*',{}).get('*',{}).get('*',{})
            nr = g.get("nrsegs", 0); nh = g.get("nhsegs", 0); nc = g.get("nbcorr", 0.0)
            outf.write(f"num_ref_segs      = {nr}\n"
                       f"avg_ref_length    = {g.get('rlength',0)/nr if nr else 0:.4f}\n"
                       f"num_hyp_segs      = {nh}\n"
                       f"avg_hyp_length    = {g.get('hlength',0)/nh if nh else 0:.4f}\n"
                       f"bound_recall      = {100*nc/(2*nr) if nr else 0:.4f}\n"
                       f"bound_precision   = {100*nc/(2*nh) if nh else 0:.4f}\n")

    # ── DET ───────────────────────────────────────────────────────────────────
    if args.detfn is not None:
        g_tar = err.get('*',{}).get('*',{}).get('*',{}).get('tar', 0.0)
        g_non = err.get('*',{}).get('*',{}).get('*',{}).get('non', 0.0)
        pts = det_print(f"{args.detfn}.all.det" if args.detfn else "",
                        g_tar, g_non, det.get('*', {}), args.detfn, args.trace)
        for lbl, key in [("max_F_measure","F"),("min_error_rate","err"),
                         ("min_half_total_error_rate","ter"),("min_equal_error_rate","eer")]:
            pt = pts.get(key, {})
            outf.write(f"{lbl} = {pt.get('val',0):.5f} [th={pt.get('th',0):.5f} "
                       f"%fr={100*pt.get('fr',0):.3f} %fa={100*pt.get('fa',0):.3f} "
                       f"recall={pt.get('r',0):.2f} precision={pt.get('p',0):.2f}]\n")

        # Courbe DET automatique si --det est utilisé
        plot_det_curve(
            det_tabs  = det.get('*', {}),
            tar_vals  = g_tar,
            non_vals  = g_non,
            labels    = [args.hypfn],
            title     = "DET Curve",
            save_path = f"{args.detfn}.det.pdf" if args.detfn else None,
            show      = False
        )

    if outf is not sys.stdout:
        outf.close()


if __name__ == "__main__":
    main()
