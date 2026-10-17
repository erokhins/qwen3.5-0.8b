import re, subprocess, pathlib

# Does a line-form cite still point at content its citing doc quotes?
#
# Why: citation_rot_check.py compares the WORKING TREE against HEAD at the cited line. With a clean tree that
# comparison is trivially equal, so after a commit that shifts lines it reports "0 rotted" while the cites are
# in fact pointing somewhere else. This script asks the question the other one cannot: for every `path:NNN`
# cite, does the doc's own quoted fragment still appear near line NNN of the file as it now is?
import sys
root = pathlib.Path('.')
cite = re.compile(r'`?([A-Za-z0-9_./\-]+\.(?:md|py|cpp|h|hpp|sh|tsv|json)):(\d{1,5})(?:-(\d+))?`?')
QUOTES = ('“', '”', '"', '`')
bad, checked = [], 0
targets = ['progress', 'bench', 'specification', 'devices', 'engine', 'prototype', 'README.md']
docs = [p for d in targets for p in root.glob(d + '/**/*.md')] + [root / 'README.md']
for doc in sorted(set(p for p in docs if p.exists())):
    try:
        lines = doc.read_text(errors='replace').splitlines()
    except Exception:
        continue
    for i, line in enumerate(lines, 1):
        for m in cite.finditer(line):
            path, ln = m.group(1), int(m.group(2))
            f = root / path
            if not f.exists():
                continue
            try:
                flines = f.read_text(errors='replace').splitlines()
            except Exception:
                continue
            if ln > len(flines):
                bad.append(f'{doc}:{i} -> {path}:{ln} BEYOND EOF ({len(flines)} lines)')
                checked += 1
                continue
            checked += 1
            # the doc's quoted phrase, if it carries one on this line
            q = re.search(r'[“"]([^”"]{12,})[”"]', line)
            if not q:
                continue
            frag = q.group(1).strip(' `')
            window = '\n'.join(flines[max(0, ln - 3):ln + 2])
            key = frag.split()[0] if frag.split() else frag
            if frag.lower() in window.lower() or key.lower() in window.lower():
                continue
            bad.append(f'{doc}:{i} -> {path}:{ln} QUOTE NOT NEAR: {frag[:52]!r}')
print(f'checked {checked} line-form cites against the text at their coordinates')
print(f'{len(bad)} suspect(s):')
for b in bad[:60]:
    print(' ', b)
