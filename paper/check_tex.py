"""
check_tex.py — static consistency checks for main.tex + references.bib.

Not a LaTeX compiler. It catches the class of problems that would otherwise
only surface at build time: unbalanced braces or environments, undefined or
unused \\label/\\ref, undefined or unused \\cite keys, duplicate labels,
missing figure files, and mismatched tabular column counts.

Usage:  python check_tex.py
Exit code 0 = no problems found.
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")
BIB = os.path.join(HERE, "references.bib")
FIGDIR = os.path.join(HERE, "figures")

problems = []
notes = []

src = open(TEX, encoding="utf-8").read()
bib = open(BIB, encoding="utf-8").read()

# Strip comments (a % not preceded by a backslash) before structural checks.
stripped = re.sub(r"(?<!\\)%.*", "", src)

# ── 1. brace balance ───────────────────────────────────────────────────────
depth = 0
for i, ch in enumerate(stripped):
    if ch == "{" and (i == 0 or stripped[i - 1] != "\\"):
        depth += 1
    elif ch == "}" and (i == 0 or stripped[i - 1] != "\\"):
        depth -= 1
        if depth < 0:
            problems.append("brace underflow near offset %d" % i)
            break
if depth != 0:
    problems.append("unbalanced braces: final depth %d" % depth)

# ── 2. environment matching ────────────────────────────────────────────────
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", stripped):
    kind, env = m.group(1), m.group(2)
    if kind == "begin":
        stack.append(env)
    else:
        if not stack:
            problems.append("\\end{%s} with no matching \\begin" % env)
        elif stack[-1] != env:
            problems.append("\\end{%s} closes \\begin{%s}" % (env, stack[-1]))
            stack.pop()
        else:
            stack.pop()
if stack:
    problems.append("unclosed environments: %s" % ", ".join(stack))

# ── 3. labels and references ───────────────────────────────────────────────
labels = re.findall(r"\\label\{([^}]+)\}", stripped)
dupes = [k for k, v in Counter(labels).items() if v > 1]
for d in dupes:
    problems.append("duplicate \\label{%s}" % d)

refs = set()
for cmd in ("ref", "eqref", "autoref"):
    refs.update(re.findall(r"\\%s\{([^}]+)\}" % cmd, stripped))
for r in sorted(refs - set(labels)):
    problems.append("undefined reference: \\ref{%s}" % r)
for l in sorted(set(labels) - refs):
    notes.append("label defined but never referenced: %s" % l)

# ── 4. citations ───────────────────────────────────────────────────────────
cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", stripped):
    cited.update(k.strip() for k in m.group(1).split(","))
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
for c in sorted(cited - bibkeys):
    problems.append("cited key missing from references.bib: %s" % c)
for b in sorted(bibkeys - cited):
    problems.append("bib entry never cited (would not appear in the list): %s" % b)

dupe_bib = [k for k, v in Counter(re.findall(r"@\w+\{([^,]+),", bib)).items() if v > 1]
for d in dupe_bib:
    problems.append("duplicate bib key: %s" % d)

# ── 5. figure files ────────────────────────────────────────────────────────
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", stripped):
    name = m.group(1)
    if not os.path.exists(os.path.join(FIGDIR, name)):
        problems.append("missing figure file: figures/%s" % name)

# ── 6. tabular column counts ───────────────────────────────────────────────
def col_count(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"p\{[^}]*\}", "p", spec)
    return sum(1 for c in spec if c in "lcrp")

def brace_group(text, start):
    """Return (content, index_after) for the balanced {...} group at `start`."""
    assert text[start] == "{"
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced group")


for m in re.finditer(r"\\begin\{tabular\}", stripped):
    spec, after = brace_group(stripped, m.end())
    end = stripped.index(r"\end{tabular}", after)
    ncols = col_count(spec)
    body = stripped[after:end]
    body = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule(\([lr]+\))?(\{[^}]*\})?)", "", body)
    for line in body.split(r"\\"):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if r"\multicolumn" in line:
            continue
        cells = len(re.findall(r"(?<!\\)&", line)) + 1
        if cells != ncols:
            problems.append(
                "tabular row has %d cells, spec declares %d: %.60s..." % (cells, ncols, line))

# ── report ─────────────────────────────────────────────────────────────────
print("labels: %d   refs: %d   cites: %d   bib entries: %d"
      % (len(labels), len(refs), len(cited), len(bibkeys)))
if notes:
    print("\nNOTES (not errors):")
    for n in notes:
        print("  -", n)
if problems:
    print("\nPROBLEMS (%d):" % len(problems))
    for p in problems:
        print("  !", p)
    sys.exit(1)
print("\nNo structural problems found.")
