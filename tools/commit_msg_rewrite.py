#!/usr/bin/env python3
"""Rewrite every commit message into header / blank line / wrapped body — the mechanical pass.

Why this exists: the repo's messages were written as one-line walls. The part of the fix
that can be done by machine must be done by machine, so that what is left for a human is
only the part that needs judgement (a header that cannot be derived at all).

Method (CPU only, no device, no build, no timing):

  1. Read messages out of the commit objects (`git cat-file --batch`), not through
     `git log --format=%B` — a log format applies whitespace cleanup on the way past, and
     a rewrite tool must not work from a lossy copy of its own input.
  2. The subject is the first PARAGRAPH, not the first line, because that is what git calls
     a subject. Five messages here hard-wrap their subject across 3-6 lines (d475b8d9,
     256a28d3, 995fd6a7, d7d5664a, 77487e8a); read line-by-line and a 1,671-char wall
     looks like a tidy 99-char header with a body under it. The paragraph is joined with
     single spaces before it is measured or split.
  3. Classify, write `<sha>.msg` per commit plus `manifest.json`, then prove no word moved.

Classification, first rule that matches wins:

  KEEP        subject <= 72 chars AND a body exists AND the header is not THIN. Header
              verbatim, body re-flowed.
  SPLIT       subject > 72 chars. Split at the first of ` -- `, ` — `, `;`, `,` (that
              priority) whose left clause is a usable header of <= 72 chars. The left
              clause is the header; everything after it becomes the body's first
              paragraph, then the original body follows. An area prefix (`bench:`,
              `engine:`, `tools:`, `prototype/<op>:`, `docs:` — here: a first token ending
              in `:`) must survive in the header, so a split point at or inside the prefix
              is not a split point. A candidate header is NOT usable (D4) if it ends with
              an unclosed `(` or `[`, or with a bare function word (and, or, but, of, in,
              to, the, ... — see FUNCTION_WORDS), or with `:`; such a split point is
              skipped and the search continues. A header that merely ends with a glued
              `,` or `;` is accepted: mildly ugly, and cheaper than 133 human rewrites.
  NEEDS-BODY  subject fits in a header but the message has no body at all (D5). The header
              is the original subject VERBATIM — the fix for this class is a body, not a
              header — and the body is the literal token `TODO-BODY`, so the file cannot
              be applied by accident.
  THIN        the derived header is legal but says nothing (D6): THIN_MAX chars or fewer,
              which per D4's re-check includes a glued `,`/`;` cut that short. Examples:
              `plan: D-5`, `results: P5`, `gdn_gated_norm: §9`. Shape is the same as HARD —
              `TODO-HEADER` in the header, the whole original subject carried in the body.
  HARD        nothing yields a <= 72 header that passes D4. The header line becomes the
              literal token `TODO-HEADER` and the original subject is kept as the body's
              first paragraph, so the file cannot be applied by accident and nothing is
              invented in its place.

KEEP and SPLIT are `ready`. NEEDS-BODY, THIN and HARD are the authoring queue.

Formatting is paragraph-level, not line-level (iteration 2; iteration 1 folded one line at
a time and left 832 stub lines under 20 characters):

  * Protected lines are verbatim AND are paragraph separators: ``` fence interiors,
    markdown table rows (`|`), list items (`-`/`*`/`+`/`1.`/`1)`), a line whose content
    has no space in it (nothing to break on), and a line whose interior holds a run of 2+
    spaces (R2: an aligned column block — a tool listing, a key/value ladder — where the
    alignment is the content and reflowing it would destroy what it shows).
  * Consecutive unprotected prose lines are ONE paragraph: join them, then fold to 76
    columns at existing spaces. The paragraph's leading indentation is repeated on every
    output line. So a list item is never joined with its neighbour, nothing is ever joined
    inside a fence, and a hand-wrapped subject or body paragraph comes out as real prose.
  * A prose paragraph longer than WALL_MAX is split into paragraphs at sentence boundaries
    (R3: `.`, `!`, `?` always, and `;` whenever paren/bracket depth is 0), preferring a
    boundary that opens an enumerator (`§`, `Q1`/`Q2`, `1)`, `- `) when one exists within
    the next sentence. No resulting paragraph is under MIN_PARA chars unless it is the last
    one. Only spaces are replaced by newlines, so the words cannot change.
  * A standalone `—`, `--`, `,` or `;` is a delimiter, not a word (R1). The split keeps it
    in the body — nothing is ever deleted — but the word proof does not grade it, so the
    delimiter that opens a body paragraph cannot make a SPLIT fail on its own.
  * Blank lines survive, so real paragraph breaks survive. Spelling, punctuation, `§`,
    `->`, `<=`, numbers and paths are never touched.

Losslessness is the contract, so two consequences that are easy to get wrong:

  * A delimiter is only consumable if it is not a word. ` -- ` and ` — ` stand alone
    surrounded by spaces, and under R1 a standalone punctuation token is not a word — so
    the split still keeps them (nothing is deleted) and they open the body's first
    paragraph ("-- the rest of the old subject"), but they are outside the proof. `;` and
    `,` are glued to the word before them, so they stay at the END of the header
    ("... speccheck/;") and travel inside a word that the proof does count. A candidate
    split whose two halves do not re-tokenize to the original word multiset is rejected —
    that is what catches the pathological cases (a `,` with no space after it, a `;`
    inside a path) instead of silently cutting a word in two.
  * Re-filling and wall-splitting move whitespace only. The word proof is therefore a
    multiset comparison over `str.split()`, which is blind to exactly that.

Only tokens containing an alphanumeric character are words (R1): the proof grades the
author's content, not the punctuation the splitter had to leave behind. A file whose body
opens with a dangling `—` therefore proves on its words, not on that dash.

The word proof is per class, because an authored header legitimately rewords:

  * KEEP / SPLIT — the file's words are EXACTLY the original message's words (multiset).
  * HARD / THIN — the BODY must contain every word of the original message (multiset
    superset). The header line is free text and is excluded from the check, because it is
    the line a human writes; the body carries the whole original subject, so the proof is
    still a real one. `TODO-HEADER` is subtracted from the body side.
  * NEEDS-BODY — the header must stay the original subject verbatim (lint enforces it, and
    that is what makes the class provable), every content word of the original SUBJECT must
    still appear in the file, and nothing but the subject may sit above the blank line that
    closes it. The new body may add whatever it needs. A body of one subject line — or one
    that swallowed the subject it was written under — is the whole failure mode this class
    exists for.

The 78-column lint applies to prose only (D1): a protected line is exempt, because a table
row, a list item or a single long token cannot be shortened without lying. `--verify`
therefore exits 2 (lint) separately from 1 (word lost or file missing), and exit 2 after
the re-fill means one thing only: a prose line with a single word too long to break.

Usage:
  python3 tools/commit_msg_rewrite.py --emit OUTDIR           # candidates + manifest.json
  python3 tools/commit_msg_rewrite.py --verify OUTDIR         # word proof + lint
  python3 tools/commit_msg_rewrite.py --apply OUTDIR          # REWRITES HISTORY, guarded
  python3 tools/commit_msg_rewrite.py --verify-history OLD NEW [CANDIDATES_DIR]
  python3 tools/commit_msg_rewrite.py --selftest              # no repo state needed
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER_MAX = 72          # what a header may be at most
WRAP = 76                # prose is folded to this
LINT_MAX = 78            # nothing longer than this unless it is protected
WALL_MAX = 240           # a prose paragraph longer than this is split into paragraphs
MIN_PARA = 40            # a wall sub-paragraph shorter than this is a stub, not a paragraph
THIN_MAX = 39            # a complete header this short says nothing (D6)
TODO_HEADER = "TODO-HEADER"
TODO_BODY = "TODO-BODY"
BACKUP_REF = "refs/heads/backup-pre-msgrewrite"

KEEP, SPLIT, NEEDS_BODY, THIN, HARD = "KEEP", "SPLIT", "NEEDS-BODY", "THIN", "HARD"
CLASSES = (KEEP, SPLIT, NEEDS_BODY, THIN, HARD)
# The authoring queue: a human writes the header (NEEDS-BODY: the body) before --apply.
AUTHORED = (HARD, THIN, NEEDS_BODY)

# Priority order, per the spec. `;` and `,` are glued to the word before them; the other
# two are standalone tokens. split_claims() proves which is which by re-tokenizing.
DELIMS = (" -- ", " \u2014 ", ";", ",")
GLUED_DELIMS = (";", ",")

AREA_RE = re.compile(r"^[^\s]+:")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
URL_RE = re.compile(r"https?://")
# R1: a token with no alphanumeric character in it is a delimiter, not a word, so the word
# proof does not count it. `§` alone is not alphanumeric either, and nothing in this
# history uses a bare `§` as its own token; a `§` inside a word keeps that word counted.
PUNCT_ONLY_RE = re.compile(r"^[^0-9A-Za-z\u00a7]+$")
# R2: an interior run of 2+ spaces is an aligned column block (a tool listing, a
# key/value ladder). The alignment is the content, so the line is never joined or refolded.
COLS_RE = re.compile(r"\S[ \t]{2,}\S")
FENCE = "```"

# D4: a header that stops on a function word, an unclosed opener or a colon is a truncated
# sentence, not a header. Matched lowercased against the last word, punctuation stripped.
FUNCTION_WORDS = frozenset(
    "and or but of in to the that with for on by as at from is are was were a an not its it"
    .split())

# D3/R3: sentence boundaries. `.`, `!`, `?` always end a sentence. `;` ends one only at
# paren/bracket depth 0, so a semicolon inside a parenthetical list stays inside its
# sentence, while one between two independent claims becomes a real boundary.
SENTENCE_END = ".!?"
BRACKET_OPEN = "(["
BRACKET_CLOSE = ")]"
GAP_RE = re.compile(r" +")
ENUM_START_RE = re.compile(r"^(?:\u00a7|\d+[.)]\s|\(\d+\)\s|Q\d+\b|[-*]\s)")


def git(*args: str, stdin: str | None = None) -> bytes:
    """The one thing this tool shells out to."""
    proc = subprocess.run(
        ["git", *args], cwd=REPO,
        input=(stdin.encode() if stdin is not None else None),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def rev_list(ref: str) -> list[str]:
    """Commits reachable from ref, oldest first — the order every report and diff uses."""
    return git("rev-list", "--reverse", ref).decode().split()


def _batch(shas: list[str]) -> list[bytes]:
    """Raw commit objects, oldest first, one `git cat-file --batch` for the whole set.

    Length and separator are both honoured, so a message containing a line shaped like a
    batch header cannot desynchronise the parse the way a line-based read would.
    """
    blob = git("cat-file", "--batch", stdin="\n".join(shas) + "\n")
    out, pos = [], 0
    for sha in shas:
        eol = blob.index(b"\n", pos)
        bits = blob[pos:eol].split(b" ")
        if len(bits) != 3 or bits[0].decode() != sha or bits[1] != b"commit":
            raise SystemExit(f"cat-file --batch desynchronized at {sha}: got {bits!r}")
        size = int(bits[2])
        out.append(blob[eol + 1: eol + 1 + size])
        pos = eol + 1 + size + 1
    return out


def read_messages(shas: list[str]) -> dict[str, str]:
    """Byte-exact message per commit, taken from the object body."""
    return {
        sha: obj[obj.find(b"\n\n") + 2:].decode("utf-8") if b"\n\n" in obj else ""
        for sha, obj in zip(shas, _batch(shas))
    }


def read_headers(shas: list[str]) -> dict[str, dict[str, object]]:
    """tree / parent count / author line / committer line, exactly as stored."""
    out: dict[str, dict[str, object]] = {}
    for sha, obj in zip(shas, _batch(shas)):
        head = obj.decode("utf-8").split("\n\n", 1)[0].split("\n")
        rec: dict[str, object] = {"tree": None, "parents": 0, "author": None, "committer": None}
        for line in head:
            if line.startswith("tree "):
                rec["tree"] = line[5:]
            elif line.startswith("parent "):
                rec["parents"] = int(rec["parents"]) + 1
            elif line.startswith("author "):
                rec["author"] = line[7:]
            elif line.startswith("committer "):
                rec["committer"] = line[10:]
        out[sha] = rec
    return out


def split_person(line: str | None) -> tuple[str, str, str]:
    """"Name <mail> 1739000000 +0100" -> (name, email, date)."""
    if not line:
        return ("", "", "")
    m = re.match(r"^(.*?) <(.*?)> (\d+ [+-]\d{4})$", line)
    if not m:
        return (line, "", "")
    return (m.group(1), m.group(2), m.group(3))


# ------------------------------------------------------------------ formatting


def split_subject_body(text: str) -> tuple[list[str], list[str], int]:
    """(subject paragraph lines, body lines, subject_end).

    Subject = up to the first blank line. `subject_end` is the subject/body boundary: the
    index, in the message's own line list, at which the subject stops and the author's prose
    region begins (the returned `body` starts there, with the separator blanks dropped).
    It is threaded through rewrite() and recorded in the manifest so an authored NEEDS-BODY
    body can be written BELOW that line and only there: a body written above it is glued
    onto the subject, git joins the two into one paragraph, and the file is no longer the
    shape the class promises. word_diff refuses such a file whoever wrote it.
    """
    lines = text.split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    subject: list[str] = []
    idx = 0
    while idx < len(lines) and lines[idx].strip() != "":
        subject.append(lines[idx])
        idx += 1
    body = lines[idx:]
    subject_end = len(subject)
    while body and body[0].strip() == "":
        body.pop(0)
    return subject, body, subject_end


def content_words(text: str) -> Counter:
    """The word multiset the proof grades: tokens with an alphanumeric character in them.

    R1: punctuation is not a word. A standalone `—`, `--`, `,` or `;` is the delimiter a
    split stopped at, not author content, so it cannot make a rewrite fail on its own. No
    proof grades, not the punctuation the splitter had to leave behind. `§` counts as
    content, because a section marker is exactly what a reader looks for.

    R1b: case is not a word either. The splitter capitalizes the clause it promotes to the
    head of a body (`— the wave-width checks …` becomes `The wave-width checks …`), and
    `The` vs `the` is one word wearing a different hat; the letters are still graded, so a
    real substitution still fails. Relaxing this only ever turns a FAIL into a PASS, which
    is why it can be introduced after a set was already proved case-sensitively.
    """
    return Counter(w.lower() for w in text.split() if not PUNCT_ONLY_RE.match(w))


def protected_line(line: str, in_fence: bool) -> bool:
    """Verbatim, exempt from the column limit, and a paragraph separator (D1).

    One spelling for both halves of the rule on purpose: a line the rewriter may not touch
    is exactly a line the linter may not complain about. A line with no space in it has
    nothing to break on, so breaking the limit there would demand a lie.

    The table-row lane requires the row to CLOSE with `|`, because prose here contains
    line-initial `|ref|/max, ...` fragments (23b6395213) that are not tables and must not
    become unsplittable walls. No commit message in this range carries a real table row.

    The column-block lane (R2) is a line whose INTERIOR holds a run of 2+ spaces: a tool
    listing or a key/value ladder whose alignment is the content (b53570cdf3's testdata
    table, e0bb0f0941's bandwidth rows). Joining or refolding such a line destroys the
    alignment, so it is kept as written — alignment beats reflow.
    """
    stripped = line.strip()
    table_row = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3
    return (in_fence or table_row or bool(BULLET_RE.match(line))
            or " " not in stripped or bool(URL_RE.search(stripped))
            or bool(COLS_RE.search(stripped)))


def fold_text(indent: str, text: str, width: int) -> list[str]:
    """Break one paragraph at existing spaces. `indent` is repeated on every output line,
    because dropping it would promote a nested bullet continuation to column 0 and
    re-draw the list the author wrote."""
    words = text.split()
    if not words:
        return [indent.rstrip()] if indent else [""]
    out: list[str] = []
    cur = ""
    for w in words:
        cand = (indent + w) if not cur else cur + " " + w
        if cur and len(cand) > width:
            out.append(cur)
            cur = indent + w
        else:
            cur = cand
    out.append(cur)
    return out


def _starts_enum(sentence: str) -> bool:
    """Does this sentence open with an enumerator (`§3`, `Q2`, `1)`, `- `)?"""
    return bool(ENUM_START_RE.match(sentence))


def split_sentences(text: str) -> list[str]:
    """The sentence pieces of a flat paragraph (D3 + R3).

    `.`, `!`, `?` always end a sentence. `;` ends one only at paren/bracket depth 0, so a
    semicolon inside a parenthetical list ("(read 240 GB/s; copy 210)") stays inside its
    sentence, while one between two independent claims is a real boundary — which matters
    here because 142 of this history's prose walls have no `.!?` anywhere, and 67 of those
    are semicolon prose that a full-stop-only rule leaves as one 300-char block. A boundary
    needs a following space and some text after it; the pieces carry no separator spaces, so
    `" ".join(pieces) == text` holds exactly when the paragraph's whitespace is flat, which
    is what split_wall tests.
    """
    pieces: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in BRACKET_OPEN:
            depth += 1
        elif ch in BRACKET_CLOSE:
            depth = max(0, depth - 1)
        elif ch in SENTENCE_END or (ch == ";" and depth == 0):
            gap = GAP_RE.match(text, i + 1)
            if gap and gap.end() < len(text):
                pieces.append(text[start:i + 1])
                start = gap.end()
    pieces.append(text[start:])
    return pieces


def split_wall(text: str) -> list[str]:
    """D3: a prose paragraph over WALL_MAX becomes several paragraphs.

    Breaks only at sentence boundaries (see split_sentences for what counts as one, R3),
    and only where the boundary is worth it: given a choice, a boundary that opens an
    enumerator (`§3`, `Q2`, `1)`, `- `) beats the plain one before it, because then the
    reader sees the section marker at the start of a paragraph instead of buried mid-line.
    A paragraph shorter than MIN_PARA is never emitted except as the last one, so a
    two-word sentence cannot become its own block. Only spaces are replaced, so the word
    multiset is untouched by construction — and `" ".join(returned) == text` is asserted,
    because a wall that does not rejoin is a splitter bug, not a shape difference.
    """
    if len(text) <= WALL_MAX:
        return [text]
    sentences = split_sentences(text)
    if len(sentences) < 2 or " ".join(sentences) != text:
        return [text]                 # no boundary to cut on, or whitespace is not flat
    groups: list[str] = []
    cur = sentences[0]
    idx = 1
    close_after = False
    while idx < len(sentences):
        if close_after:                       # the enumerator boundary was just passed
            groups.append(cur)
            cur = sentences[idx]
            idx += 1
            close_after = False
            continue
        nxt = sentences[idx]
        fits = len(cur) + 1 + len(nxt) <= WALL_MAX
        stub = len(cur) < MIN_PARA
        prefer = (not fits and not stub and not _starts_enum(nxt)
                  and idx + 1 < len(sentences) and _starts_enum(sentences[idx + 1]))
        if fits or stub or prefer:
            cur = cur + " " + nxt
            idx += 1
            close_after = prefer
            continue
        groups.append(cur)
        cur = nxt
        idx += 1
    groups.append(cur)
    if " ".join(groups) != text:
        raise SystemExit("split_wall lost or moved a word — refusing to emit")
    return groups


def paragraph_lines(lines: list[str]) -> list[str]:
    """One prose paragraph (consecutive unprotected lines) -> re-filled, wall-split lines.

    Whitespace inside a paragraph is flattened before it is re-folded, because `fold_text`
    rejoins words with single spaces anyway and a sentence ending in `. ` (two spaces, which
    is how half this history was written) would otherwise hide its own boundary from D3.
    Only spaces and newlines move; no word can change as a result.
    """
    indent = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
    text = " ".join(" ".join(line.strip() for line in lines).split())
    out: list[str] = []
    for chunk in split_wall(text):
        if out:
            out.append("")                    # a paragraph break the wall split created
        out.extend(fold_text(indent, chunk, WRAP))
    return out


def rewrap(lines: list[str]) -> list[str]:
    """Re-flow a body. Protected lines are verbatim separators; consecutive prose lines are
    one paragraph and are joined before they are folded (D2).

    The threshold for touching a prose paragraph is the paragraph, not the line: a body
    hand-wrapped at 40 columns used to come out as 40-column stubs, which is what made 832
    lines of the candidate set unreadable. Blank lines and protected lines are the only
    thing that separates two paragraphs, so an author's real paragraph breaks survive.
    """
    out: list[str] = []
    para: list[str] = []
    in_fence = False

    def flush() -> None:
        if para:
            out.extend(paragraph_lines(para))
            para.clear()

    for raw in lines:
        line = raw.rstrip()                       # trailing whitespace is never content
        stripped = line.strip()
        if stripped.startswith(FENCE):
            flush()
            out.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(line)
            continue
        if stripped == "":
            flush()
            out.append("")
            continue
        if protected_line(line, False):
            flush()
            out.append(line)
            continue
        para.append(line)
    flush()
    while out and out[-1] == "":
        out.pop()
    return out


def header_shape_reject(head: str) -> str | None:
    """D4: why this candidate header is a truncated sentence rather than a header.

    The glued `,`/`;` ending is deliberately NOT rejected here: 133 of the 272 SPLIT headers
    end that way, they read as mildly ugly, and rewriting 133 of them by hand is a worse
    bill than the comma. What D4's re-check does catch is a glued cut that is ALSO thin, and
    that is demoted to THIN by rewrite() — the length rule, not the punctuation, is what
    makes a header say nothing.
    """
    if head[-1] in "([":
        return "ends with an unclosed " + head[-1]
    if head.endswith(":"):
        return "ends with a colon"
    words = head.split()
    last = re.sub(r"[^0-9A-Za-z\u00a7]+$", "", words[-1]) if words else ""
    if last.lower() in FUNCTION_WORDS:
        return f"ends with the function word {last!r}"
    return None


def tidy_rest(rest: str, delim: str) -> str:
    """Start the remainder like a sentence, without touching a single word.

    A non-glued split hands its delimiter back to the body, so 78 SPLIT bodies opened
    `— the wave-width checks hold the scan identical …` — a fragment that reads like a
    truncation of the header sitting above it. Two edits are allowed and nothing else: drop
    the delimiter the splitter actually consumed (a standalone punctuation token is not a
    word, by R1, so the proof cannot notice), and capitalize the first bare lowercase word
    (case is not a word, by R1b). Scoping the strip to the consumed delimiter is what keeps
    a markdown bullet at the top of a body from being eaten, and the identifier guard — the
    capitalize fires only on a plain word followed by whitespace or `,`/`;`, never on
    `engine::kStopToken` or `gate.py` — keeps code spelled as the author spelled it.
    """
    d = delim.strip()
    while rest.startswith(d) and rest[len(d):len(d) + 1].isspace():
        rest = rest[len(d):].strip()
    m = re.match(r"([a-z][a-z]+)(?=[\s,;]|$)", rest)
    if m and "_" not in m.group(1):
        rest = m.group(1)[:1].upper() + m.group(1)[1:] + rest[len(m.group(1)):]
    return rest


def split_claims(subject: str) -> dict:
    """First delimiter (by priority) whose left clause is a usable header.

    Always returns a dict: {"head", "rest", "rejected"}, with head=None when nothing is
    usable. `rejected` lists every candidate that looked like a split point and was refused,
    so a HARD can say why rather than just shrug.

    The delimiter balance is proved over content words only (R1), so a standalone `—` or
    `--` cannot be the reason a usable split is refused. Bracket depth is NOT a filter here:
    R3 confines the depth rule to split_wall, and refusing a `,` inside a parenthetical —
    the same depth test — would cost 7 more commits a human has to write a header for, for
    the sake of the 9 SPLIT headers that currently carry an unclosed `(`.
    """
    total = content_words(subject)
    area_end = 0
    m = AREA_RE.match(subject)
    if m:
        area_end = m.end()
    rejected: list[str] = []
    for d in DELIMS:
        start = 0
        while True:
            i = subject.find(d, start)
            if i < 0:
                break
            start = i + 1
            if d in GLUED_DELIMS:
                head, rest = subject[:i + 1].rstrip(), subject[i + 1:].strip()
            else:
                head, rest = subject[:i].rstrip(), tidy_rest(subject[i:].strip(), d)
            if not head or not rest or len(head) > HEADER_MAX:
                continue
            # The header must still name the area: the split has to land past the prefix.
            if len(head) <= area_end:
                continue
            if not re.search(r"[A-Za-z]", head):
                continue
            # A delimiter may only be consumed if it is not a word of its own.
            if content_words(head) + content_words(rest) != total:
                continue
            why = header_shape_reject(head)
            if why is not None:
                rejected.append(f"{head!r} {why}")
                continue
            return {"head": head, "rest": rest, "rejected": rejected}
    return {"head": None, "rest": "", "rejected": rejected}


def rewrite(text: str) -> dict:
    """Classify one message and produce its candidate rewrite.

    The class decides what a human still owes: KEEP/SPLIT need nothing, HARD/THIN owe a
    header (the body already carries the whole original message), NEEDS-BODY owes a body
    (the header is already the original subject).
    """
    subject_lines, body, subject_end = split_subject_body(text)
    subject = " ".join(l.strip() for l in subject_lines if l.strip())
    has_body = any(l.strip() for l in body)

    if len(subject) <= HEADER_MAX and not has_body:
        # D5: the subject is fine, the emptiness is the defect. Fixing the header here
        # would be rearranging the one part that is not broken.
        cls, header, src = NEEDS_BODY, subject, [TODO_BODY]
        note = "subject fits; there is no body at all — the fix is a body, not a header"
    elif len(subject) <= HEADER_MAX:
        cls, header, src, note = KEEP, subject, list(body), ""
    else:
        claim = split_claims(subject)
        if claim["head"] is not None:
            cls, header, src = SPLIT, claim["head"], [claim["rest"], ""] + list(body)
            note = ""
        else:
            cls, header, src = HARD, TODO_HEADER, [subject, ""] + list(body)
            note = ("no split point yields a usable header"
                    + (": " + "; ".join(claim["rejected"]) if claim["rejected"] else ""))

    if cls in (KEEP, SPLIT) and len(header) <= THIN_MAX:
        # D6: a complete header that says nothing is not ready — not `plan: D-5`, and not a
        # glued `,`/`;` cut under the limit either (D4's re-check: a trailing `,` under 40
        # chars is a thin header, not merely an ugly one). Same shape as HARD: the sentinel
        # keeps --apply out, and the body carries the stub so nothing is lost.
        note = f"header says nothing ({header!r}, {len(header)} chars)"
        cls, src = THIN, [subject, ""] + (list(body) if cls == KEEP else src[2:])
        header = TODO_HEADER

    return {
        "class": cls,
        "text": "\n".join([header, ""] + rewrap(src)) + "\n",
        "header": header,
        "subject": subject,
        "subject_len": len(subject),
        # Where the author's own prose starts inside `body` (R4 plumbing): the emitted
        # NEEDS-BODY body goes below the blank line that closes the subject, and a body
        # written above that boundary is git subject text, not a body. lint and word_diff
        # both judge an authored body against this line, never against a re-derived one.
        "subject_end": subject_end,
        "has_body": has_body,
        "note": note,
    }


# ------------------------------------------------------------------ lint


def lint(text: str, cls: str) -> list[str]:
    """Structural checks. Every one of them can fail; --selftest proves each.

    The column limit is a PROSE rule (D1): protected lines are exempt for the same reason
    they are left verbatim — a table row, a list item or a single long token cannot be
    shortened without lying about its content.

    A NEEDS-BODY file must keep the subject/body boundary readable: git's subject is the
    first paragraph, so line 2 has to be the blank line that closes it and the authored body
    starts on line 3. Body text pushed above that boundary is not a body, it is subject text
    git will join and truncate — caught here by line 2 and in word_diff by the subject carry.
    """
    problems: list[str] = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or not lines[0].strip():
        return ["header line is empty"]
    if len(lines[0]) > HEADER_MAX:
        problems.append(f"header is {len(lines[0])} chars, limit {HEADER_MAX}")
    if len(lines) < 2 or lines[1] != "":
        problems.append("line 2 is not blank")
    if not any(l.strip() for l in lines[2:]):
        problems.append("body is empty")
    in_fence = False
    for n, line in enumerate(lines, start=1):
        was_in_fence = in_fence
        if line.strip().startswith(FENCE):
            in_fence = not in_fence
        if line != line.rstrip():
            problems.append(f"line {n}: trailing whitespace")
        if len(line) > LINT_MAX and not protected_line(line, was_in_fence):
            problems.append(f"line {n}: {len(line)} chars, limit {LINT_MAX}")
    if TODO_HEADER in text and cls not in (HARD, THIN):
        problems.append(f"{TODO_HEADER} left in a file marked READY ({cls})")
    if TODO_BODY in text and cls != NEEDS_BODY:
        problems.append(f"{TODO_BODY} left in a file marked {cls}")
    return problems


def _drop_words(counter: Counter, token: str) -> None:
    if counter.get(token):
        counter[token] -= 1
        if counter[token] <= 0:
            del counter[token]


def word_diff(original: str, rewritten: str, cls: str | None) -> list[str]:
    """The losslessness check, per class.

    Only content words are counted (R1): a standalone `—`, `--`, `,` or `;` is a delimiter
    the splitter left behind, not author text, so it cannot be what makes a file fail.

    KEEP / SPLIT   exact multiset equality over the whole file.
    HARD / THIN    the BODY must carry every original word; the header line is the line a
                   human writes and is excluded, so `TODO-HEADER` and any authored wording
                   are free. A dropped number in the body still fails.
    NEEDS-BODY     three rules, because an authored body may replace the placeholder line
                   wholesale and even refill a hand-wrapped subject: the header must still
                   be the original subject verbatim; every content word of the original
                   SUBJECT must appear in the file (a body that swallowed the subject's
                   subject matter fails here, not by luck); and nothing but the subject may
                   sit above the blank line that closes the subject, because git reads that
                   region as the subject and joins it. The file as a whole must still carry
                   every original word — the new body may add whatever it needs.
    """
    before = content_words(original)
    lines = rewritten.split("\n")
    if cls in (NEEDS_BODY, HARD, THIN):
        if cls == NEEDS_BODY:
            after = content_words(rewritten)
            _drop_words(after, TODO_BODY)
            out = [f"lost {n}x {w!r}" for w, n in sorted((before - after).items())]
            subject_lines, _, subject_end = split_subject_body(original)
            want = " ".join(l.strip() for l in subject_lines if l.strip())
            if not lines or lines[0].strip() != want:
                out.append("header is not the original subject verbatim")
            # The subject/body boundary of the file itself: everything before its first
            # blank line is git's subject, so it may hold the subject and nothing else.
            probe = lines[:]
            while probe and probe[-1] == "":
                probe.pop()
            blank = next((i for i, l in enumerate(probe) if l.strip() == ""), len(probe))
            strays = content_words(" ".join(probe[:blank])) - content_words(
                " ".join(subject_lines[:subject_end]))
            if strays:
                out.append("body text above the subject/body boundary: "
                           + " ".join(sorted(strays)))
            return out
        body_counter = content_words("\n".join(lines[1:]))
        _drop_words(body_counter, TODO_HEADER)
        return [f"lost {n}x {w!r}" for w, n in sorted((before - body_counter).items())]
    after = content_words(rewritten)
    out = [f"lost {n}x {w!r}" for w, n in sorted((before - after).items())]
    out += [f"invented {n}x {w!r}" for w, n in sorted((after - before).items())]
    return out

# ------------------------------------------------------------------ commands


def load_manifest(indir: str) -> dict:
    path = os.path.join(indir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cmd_emit(outdir: str) -> int:
    shas = rev_list("HEAD")
    msgs = read_messages(shas)
    os.makedirs(outdir, exist_ok=True)
    counts: Counter = Counter()
    rows = []
    for sha in shas:
        r = rewrite(msgs[sha])
        counts[r["class"]] += 1
        with open(os.path.join(outdir, f"{sha}.msg"), "w", encoding="utf-8") as fh:
            fh.write(r["text"])
        rows.append({
            "sha": sha,
            "class": r["class"],
            "ready": r["class"] not in AUTHORED,
            "file": f"{sha}.msg",
            "header": r["header"],
            "header_len": len(r["header"]),
            "subject_len": r["subject_len"],
            # Recorded so an authoring driver writes a NEEDS-BODY body below the boundary
            # instead of into it; word_diff refuses a file that breaks it either way.
            "subject_end": r["subject_end"],
            "had_body": r["has_body"],
            "note": r["note"],
            "lint": lint(r["text"], r["class"]),
        })
    manifest = {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "count": len(shas),
        "counts": {k: counts.get(k, 0) for k in CLASSES},
        "commits": rows,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    print(f"EMITTED {len(shas)} message files into {outdir}")
    print("  " + "  ".join(f"{k}={counts.get(k, 0)}" for k in CLASSES))
    print(f"  authoring queue (oldest first): "
          f"{sum(1 for row in rows if row['class'] in AUTHORED)} commits "
          f"({'+'.join(k for k in AUTHORED)})")
    return 0


def cmd_verify(indir: str) -> int:
    """Recompute the word proof per commit and lint the result. Exit 1 / 2 / 0.

    The class comes from the manifest, because the word contract is per class: an authored
    header may reword, a mechanically derived one may not.

    A manifest record may name a `ref`: the commit whose message is the reference for the
    word proof. Without it the file's own sha is used, which is right the first time and
    wrong for every rewrite after one — after a history rewrite the file is named for the
    REWRITTEN commit, and grading a rewrite against a rewrite asks whether the second
    rewrite preserved the first, which is not the contract. The contract is always against
    the author's original text, so a re-apply carries the original sha forward in `ref`.
    """
    shas = rev_list("HEAD")
    manifest = load_manifest(indir)
    classes = {c["sha"]: c["class"] for c in manifest.get("commits", [])}
    refs = {c["sha"]: c.get("ref") or c["sha"] for c in manifest.get("commits", [])}
    msgs = read_messages(sorted(set(shas) | {r for r in refs.values() if r}))
    if manifest and len(classes) != len(shas):
        print(f"LINT manifest lists {len(classes)} commits, HEAD reaches {len(shas)}")
    word_fails = lint_fails = 0
    authored = pending = 0
    for sha in shas:
        path = os.path.join(indir, f"{sha}.msg")
        if not os.path.exists(path):
            print(f"FAIL {sha[:12]} missing file {sha}.msg")
            word_fails += 1
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        cls = classes.get(sha)
        problems = []
        if cls is None:
            problems.append("not listed in manifest.json")
        elif cls not in CLASSES:
            problems.append(f"manifest class {cls!r} is not one of {'/'.join(CLASSES)}")
        else:
            ref = refs.get(sha, sha)
            if ref not in msgs:
                problems.append(f"word-proof reference {ref[:12]} is not reachable")
            else:
                problems += word_diff(msgs[ref], text, cls)
            if cls in AUTHORED:
                # A queue file that no longer carries a sentinel has been authored, so the
                # class contract above is what decided it. Counted either way: a
                # half-authored batch has to show in the summary line.
                if TODO_HEADER in text or TODO_BODY in text:
                    pending += 1
                else:
                    authored += 1
        if problems:
            print(f"FAIL {sha[:12]} [{cls}] " + "; ".join(problems))
            word_fails += 1
        problems = lint(text, cls or HARD)
        if problems:
            print(f"LINT {sha[:12]} [{cls}] " + "; ".join(problems))
            lint_fails += 1
    print(f"VERIFY files={len(shas)} word_failures={word_fails} lint_failures={lint_fails}")
    print(f"  authoring queue: {authored} authored, {pending} still carrying a sentinel")
    if word_fails:
        print("exit 1: a word was lost or invented, or a file is missing — do not apply")
        return 1
    if lint_fails:
        print(f"exit 2: shape only — prose over {LINT_MAX} that re-filling could"
              f"not shorten (one word longer than the limit). Protected lines — fences,"
              f"table rows, list items, spaceless tokens — are exempt and verbatim.")
        return 2
    return 0


def cmd_apply(indir: str) -> int:
    """REWRITES HISTORY. Guards run before anything is touched."""
    indir = os.path.abspath(indir)
    if not os.path.isdir(indir):
        raise SystemExit(f"--apply: {indir} is not a directory")
    if git("status", "--porcelain").strip():
        raise SystemExit("--apply: working tree is dirty; commit or stash first")
    shas = rev_list("HEAD")
    missing, todos = [], []
    for sha in shas:
        path = os.path.join(indir, f"{sha}.msg")
        if not os.path.exists(path):
            missing.append(sha)
            continue
        with open(path, encoding="utf-8") as fh:
            if TODO_HEADER in fh.read():
                todos.append(sha)
    if missing:
        raise SystemExit(f"--apply: {len(missing)} message files missing, e.g. {missing[0]}")
    if todos:
        raise SystemExit(
            f"--apply: {len(todos)} files still carry {TODO_HEADER}, e.g. {todos[0]}"
        )
    # `git show-ref --verify` answers "does this ref exist" with its exit status, so this
    # one call goes to subprocess.run directly: git()'s contract is to die on non-zero.
    if subprocess.run(["git", "show-ref", "--verify", "--quiet", BACKUP_REF], cwd=REPO).returncode != 0:
        git("update-ref", BACKUP_REF, git("rev-parse", "HEAD").decode().strip())
        print(f"created {BACKUP_REF} at HEAD")
    else:
        print(f"{BACKUP_REF} already exists at "
              f"{git('rev-parse', BACKUP_REF).decode().strip()}")
    filt = f'cat "{indir}"/"$GIT_COMMIT".msg'
    out = subprocess.run(
        ["git", "filter-branch", "-f", "--msg-filter", filt, "HEAD"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "FILTER_BRANCH_SQUELCH_WARNING": "1"},
    )
    sys.stdout.write(out.stdout.decode("utf-8", "replace"))
    if out.returncode != 0:
        raise SystemExit(f"--apply: git filter-branch exited {out.returncode}")
    print(f"applied {len(shas)} rewritten messages; HEAD is now "
          f"{git('rev-parse', 'HEAD').decode().strip()}")
    return 0


def cmd_verify_history(oldref: str, newref: str, indir: str | None = None) -> int:
    """The gate: prove nothing but the messages changed.

    Without INDIR every commit is judged by the strict rule (exact word multiset), which
    is the right default because a false alarm is cheaper than a false pass: it will
    report every authored-class commit (HARD / THIN / NEEDS-BODY, whose header is free
    text by contract) as "invented" words. Pass the candidates directory to get the
    per-commit class from its manifest and a verdict that can actually reach PASS.
    """
    old, new = rev_list(oldref), rev_list(newref)
    if len(old) != len(new):
        print(f"FAIL commit count {len(old)} ({oldref}) != {len(new)} ({newref})")
        return 1
    classes: dict[str, str] = {}
    if indir:
        classes = {r["sha"]: r["class"] for r in load_manifest(indir)["commits"]}
    oh, nh = read_headers(old), read_headers(new)
    om, nm = read_messages(old), read_messages(new)
    fails = 0
    for i, (o, n) in enumerate(zip(old, new), start=1):
        bad: list[str] = []
        for key in ("tree", "parents"):
            if oh[o][key] != nh[n][key]:
                bad.append(f"{key} {oh[o][key]} != {nh[n][key]}")
        oa, oe, odate = split_person(oh[o]["author"])
        na, ne, ndate = split_person(nh[n]["author"])
        if (oa, oe) != (na, ne):
            bad.append(f"author {oa} <{oe}> != {na} <{ne}>")
        if odate != ndate:
            bad.append(f"author date {odate} != {ndate}")
        ocd = split_person(oh[o]["committer"])[2]
        ncd = split_person(nh[n]["committer"])[2]
        if ocd != ncd:
            bad.append(f"committer date {ocd} != {ncd}")
        # The messages are EXPECTED to differ — that is the whole change. What may not
        # differ is their word multiset, so that is the only message comparison here.
        # A `TODO-HEADER` left in the new side shows up as an invented word.
        diffs = word_diff(om[o], nm[n], classes.get(o, "KEEP"))
        if diffs:
            bad.append("message words changed: " + "; ".join(diffs))
        if bad:
            print(f"FAIL {i}/{len(old)} {o[:12]} -> {n[:12]} " + "; ".join(bad))
            fails += 1
    if fails:
        print(f"VERIFY-HISTORY FAIL ({fails} of {len(old)} commits differ beyond the message)")
        return 1
    print(f"VERIFY-HISTORY PASS ({len(old)} commits: tree, parent count, author name+email, "
          f"author date and committer date identical in every one; no word lost or invented "
          f"between {oldref} and {newref}"
          + (f", judged by the classes in {indir}/manifest.json" if indir else ")")
          )
    return 0

# ------------------------------------------------------------------ selftest


def _long_clause(n_words: int, start: str = "engine: ") -> str:
    """A clause of about n_words words, with no delimiter character anywhere in it."""
    filler = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    words = [start.strip()] + [filler[i % len(filler)] + "x" * (i // len(filler))
                               for i in range(n_words)]
    return start + " ".join(words[len(words) - n_words - 1:])


def selftest() -> int:
    """Synthetic messages, one assertion per rule, and every rule proven able to fail.

    Runs with no repository state: nothing here rewrites history and nothing writes
    outside a temp directory.
    """
    checks: list[tuple[str, object, object]] = []

    def check(name: str, got, want) -> None:
        checks.append((name, got, want))

    # ---- KEEP: subject + body, long enough to be a header. Header verbatim, body re-wrapped.
    keep = (
        "engine: fix the norm divisor so the wave matches the pinned dumps\n\n"
        "The sum was taken over 8 lanes, want 128.\n"
        "A second paragraph here.\n"
    )
    r = rewrite(keep)
    check("KEEP class", r["class"], "KEEP")
    check("KEEP header verbatim",
          r["header"], "engine: fix the norm divisor so the wave matches the pinned dumps")
    check("KEEP body kept", r["text"].count("want 128."), 1)
    check("KEEP lossless", word_diff(keep, r["text"], "KEEP"), [])
    check("KEEP lint clean", lint(r["text"], "KEEP"), [])

    # ---- D2: consecutive prose lines are ONE paragraph, joined then re-folded.
    long_subj = "engine: a subject long enough to be a header and not a stub"
    d2_in = long_subj + "\n\nalpha beta\ngamma delta\necho foxtrot\n"
    check("consecutive prose lines are one paragraph",
          rewrite(d2_in)["text"].split("\n")[2:-1], ["alpha beta gamma delta echo foxtrot"])
    check("re-fill neither invents nor loses a word", word_diff(d2_in, rewrite(d2_in)["text"], "KEEP"), [])

    # ---- D2b: a non-glued split must not leave the delimiter as the body's first token.
    d2b = ("engine: the wave-width checks hold the scan identical on both sides — both "
           "were uninitialized LDS on any body whose tail is short")
    r_d2b = rewrite(d2b)
    check("split body opens on a sentence, not a dash",
          r_d2b["text"].split("\n")[2].startswith("Both were uninitialized"), True)
    check("split body still loses no word", word_diff(d2b, r_d2b["text"], r_d2b["class"]), [])
    check("tidy leaves an identifier alone", tidy_rest("— engine::kStopToken ends the row", "—"),
          "engine::kStopToken ends the row")
    check("tidy does not eat a markdown bullet", tidy_rest("-- item one", "—"), "-- item one")
    check("tidy strips only the delimiter it consumed", tidy_rest("— `gate.py` is the check", "—"),
          "`gate.py` is the check")

    # ---- paragraph separators are honoured: blank line, list item, fence, spaceless line.
    check("a blank line separates paragraphs",
          rewrite(long_subj + "\n\nalpha beta\n\ngamma delta\n")["text"].split("\n")[2:-1],
          ["alpha beta", "", "gamma delta"])
    check("a list item is never joined with its neighbour",
          rewrite(long_subj + "\n\n- one\ntwo\n- three\n")["text"].split("\n")[2:-1],
          ["- one", "two", "- three"])
    check("nothing is joined inside a fence",
          rewrite(long_subj + "\n\n```\nalpha\nbeta\n```\n")["text"].split("\n")[2:-1],
          ["```", "alpha", "beta", "```"])
    check("a spaceless line separates paragraphs and stays verbatim",
          rewrite(long_subj + "\n\nalpha\n" + "t" * 90 + "\nbeta\n")["text"].split("\n")[2:-1],
          ["alpha", "t" * 90, "beta"])

    # ---- R2: an aligned column block is protected — never joined, never refolded,
    # kept as written, because the alignment is the content (b53570cdf3, e0bb0f0941).
    cols_body = ("  sink(tok)      - called per emitted token, so the REPL streams\n"
                 "  resume_from=   - the caller may prefill itself before resuming\n")
    long_col = "  iGPU read" + " " * 30 + "240 GB/s best at 1 GiB " + "(measured) " * 8
    check("R2: an interior run of 2+ spaces makes a line protected",
          protected_line("  sink(tok)      - per emitted token", False), True)
    check("R2: single-spaced prose is not swept into the new family",
          protected_line("sink(tok) - called per emitted token", False), False)
    r_cols = rewrite(long_subj + "\n\n" + cols_body + "\n" + long_col + "\n")
    check("R2: column lines are never joined with their neighbour",
          [l for l in r_cols["text"].split("\n") if l.startswith("  sink(tok)")],
          ["  sink(tok)      - called per emitted token, so the REPL streams"])
    check("R2: a column block keeps its alignment byte for byte",
          "  resume_from=   - the caller may prefill itself before resuming"
          in r_cols["text"].split("\n"), True)
    check("R2: a long column line is lint-exempt, like every protected family",
          [p for p in lint(r_cols["text"], r_cols["class"]) if "chars, limit" in p], [])
    check("R2 column block is lossless",
          word_diff(cols_body + long_col,
                    "\n".join(r_cols["text"].split("\n")[2:]), "KEEP"), [])

    # ---- an over-long nested line keeps its indentation on every folded line.
    nested_body = "  " + "cont " * 30 + "tail"
    r = rewrite("engine: fix the wave across the whole lane group\n\n" + nested_body + "\n")
    folded = r["text"].split("\n\n")[1].strip("\n").split("\n")
    check("nested fold keeps the indent on every line",
          [l for l in folded if not l.startswith("  ")], [])
    check("nested fold wrapped inside the limit",
          [l for l in folded if len(l) > WRAP], [])
    check("nested fold lossless", word_diff(nested_body, "\n".join(folded), "KEEP"), [])

    # ---- a paragraph already inside the limit keeps the author's words in one line.
    author_break = long_subj + "\n\n" + ("a " * 38).strip() + "\n"
    check("a legal-length paragraph comes back as one line",
          rewrite(author_break)["text"].split("\n")[2], ("a " * 38).strip())

    # ---- a 1,500-char one-line subject: splits at the ` -- `, header <= 72, lossless.
    big = ("engine: gate the wave path so a lane row cannot race its neighbour -- "
           + "word " * 300 + "end")
    big_msg = big + "\n"
    check("1500-char subject is > 1400 chars", len(big) > 1400, True)
    r = rewrite(big_msg)
    check("1500-char class", r["class"], "SPLIT")
    check("1500-char header fits", len(r["header"]) <= HEADER_MAX, True)
    check("1500-char header keeps the prefix", r["header"].startswith("engine:"), True)
    check("1500-char body opens with the delimiter word",
          r["text"].split("\n")[2].startswith("-- word"), True)
    check("1500-char every line wrapped",
          [len(l) for l in r["text"].split("\n") if len(l) > WRAP], [])
    check("1500-char lossless", word_diff(big_msg, r["text"], "SPLIT"), [])
    check("1500-char lint clean", lint(r["text"], "SPLIT"), [])
    # A wall with no sentence boundary stays one paragraph: D3 may only cut at a boundary.
    check("a wall with no sentence boundary is not split into paragraphs",
          [l for l in r["text"].rstrip("\n").split("\n") if l == ""], [""])

    # ---- first clause itself 90 chars, no other delimiter -> HARD, never applied.
    hard = _long_clause(18) + " -- " + "tail words here"
    check("90-char first clause length", len(_long_clause(18)) > 72, True)
    hard_msg = hard + "\n\nAn original body line.\n"
    r = rewrite(hard_msg)
    check("90-char first clause class", r["class"], "HARD")
    check("HARD header is the sentinel", r["text"].split("\n")[0], TODO_HEADER)
    hard_paras = [p for p in r["text"].split("\n", 2)[2].split("\n\n") if p.strip()]
    check("HARD keeps the whole subject as the body's first paragraph",
          " ".join(hard_paras[0].split("\n")), hard)
    check("HARD keeps the original body", "An original body line." in r["text"], True)
    check("HARD lossless modulo the sentinel", word_diff(hard_msg, r["text"], "HARD"), [])
    check("HARD lint clean while unready", lint(r["text"], "HARD"), [])

    # ---- a table and a fenced block come back unreflowed, long lines and all.
    fence_line = "--gdn-scan collapse " + "x" * 90
    table_row = "| route " + "x" * 60 + " | verdict " + "y" * 40 + " |"
    md = (
        "bench: the census table " + "z" * 60 + " and more words past the limit\n\n"
        + ("intro prose keeps going with ordinary words that are each short enough "
           "to fold at a space, so nothing has to be broken mid-word here at all.\n")
        + "```\n" + fence_line + "\n```\n\n"
        + "| head | verdict |\n| " + "q" * 90 + " | 1 |\n" + table_row + "\n\n"
        + "Closing prose.\n"
    )
    r = rewrite(md)
    out = r["text"]
    check("fence line unreflowed", fence_line in out.split("\n"), True)
    check("table row unreflowed", table_row in out.split("\n"), True)
    check("table long row unreflowed",
          "| " + "q" * 90 + " | 1 |" in out.split("\n"), True)
    check("fence/table long lines are lint-exempt",
          [p for p in lint(out, r["class"]) if "chars, limit" in p], [])
    check("prose line over the limit was folded",
          [l for l in out.split("\n") if l.startswith("intro prose") and len(l) > WRAP], [])
    check("markdown case lossless", word_diff(md, out, r["class"]), [])

    # ---- a pipe-initial PROSE line is not a table row. Found for real: 23b6395213 wraps
    # mid-sentence onto "|ref|/max, ...", and treating that as a table row would make a
    # 2,500-char wall unsplittable and lint-exempt.
    pipe_prose = ("engine: price the row by row and print " + "|ref|/max " * 14 + "end\n")
    r = rewrite(pipe_prose)
    check("a pipe-initial prose line is not protected as a table row",
          [l for l in r["text"].split("\n") if l.startswith("|ref|/max ") and len(l) > WRAP], [])
    check("a real table row still is protected",
          protected_line("| head | verdict |", False), True)
    check("a pipe-initial prose line is not protected",
          protected_line("|ref|/max, both sides top-1 room and the verdict here", False), False)

    # ---- trailing whitespace is stripped, and it cannot survive into a rewrite.
    tw = long_subj + "\n\nfirst line has one \nsecond  \n"
    r = rewrite(tw)
    check("trailing whitespace stripped",
          [l for l in r["text"].split("\n") if l != l.rstrip()], [])
    check("trailing whitespace lint clean", lint(r["text"], r["class"]), [])
    check("trailing whitespace lossless", word_diff(tw, r["text"], r["class"]), [])

    # ---- delimiter priority: ` -- ` beats `;`, and `;` beats `,`.
    p1 = ("tools: add the gate to the run protocol so two sweeps cannot collide -- "
          + "it ships red; fix the parser, then run it and log the count\n")
    check("priority case is long enough to need a split", len(p1.strip()) > HEADER_MAX, True)
    check("` -- ` beats `;`", rewrite(p1)["header"],
          "tools: add the gate to the run protocol so two sweeps cannot collide")
    p2 = ("tools: add the gate to the run protocol so two sweeps cannot collide; "
          + "fix the parser, then run it all now " + "u" * 40 + "\n")
    check("`;` beats `,`", rewrite(p2)["header"],
          "tools: add the gate to the run protocol so two sweeps cannot collide;")
    p3 = ("tools: add the gate to the run protocol so two sweeps cannot collide, "
          + "then run it all " + "v" * 40 + "\n")
    check("a glued `,` stays at the end of the header",
          rewrite(p3)["header"],
          "tools: add the gate to the run protocol so two sweeps cannot collide,")
    check("glued comma is not invented as a word",
          word_diff(p3, rewrite(p3)["text"], rewrite(p3)["class"]), [])
    check("a long glued cut stays SPLIT (ugly but legal, per D4)", rewrite(p3)["class"], SPLIT)
    thin_glue = ("results: P5, with the body of the message running on past the split point "
                 + "and saying considerably more than that\n")
    check("a glued cut under THIN_MAX is THIN, not merely ugly (D4 re-check)",
          rewrite(thin_glue)["class"], THIN)

    # ---- an area prefix must survive: a delimiter inside the prefix is not a split point.
    pre = "proto,type: the first comma is inside the area word so it cannot head a line\n"
    r = rewrite(pre)
    check("split inside the area prefix is refused", r["class"], "HARD")
    check("refusal did not eat the prefix", r["text"].split("\n")[2].startswith("proto,type:"),
          True)

    # ---- the negative controls: every lint rule must be able to fail.
    check("lint catches a long header",
          lint("x" * 80 + "\n\nbody\n", "KEEP"), ["header is 80 chars, limit 72"])
    check("lint catches an empty header", lint("\n\nbody\n", "KEEP"), ["header line is empty"])
    check("lint catches a missing blank line 2",
          lint("engine: ok\nbody at once\n", "KEEP"),
          ["line 2 is not blank", "body is empty"])
    check("lint catches an empty body",
          lint("engine: ok\n\n", "KEEP"), ["body is empty"])
    check("lint catches a long prose line",
          lint("engine: ok\n\n" + ("a b " * 30).strip() + "\n", "KEEP")[0],
          "line 3: 119 chars, limit 78")
    check("lint catches trailing whitespace",
          lint("engine: ok\n\nbody text \n", "KEEP"), ["line 3: trailing whitespace"])
    check("lint catches the sentinel in a READY file",
          lint(TODO_HEADER + "\n\nbody\n", "SPLIT"),
          [f"{TODO_HEADER} left in a file marked READY (SPLIT)"])
    check("lint lets the sentinel through while HARD",
          lint(TODO_HEADER + "\n\nbody\n", "HARD"), [])
    check("lint lets a URL and a spaceless token through",
          lint("engine: ok\n\nsee https://x/" + "y" * 80 + "\n" + "t" * 90 + "\n", "KEEP"),
          [])

    # ---- the word proof must be able to fail, in both directions.
    src = "engine: a subject long enough to be a header on its own\n\none two three four\n"
    lost = rewrite(src)
    check("word proof catches a dropped word",
          word_diff(src, lost["text"].replace("three", ""), "KEEP"), ["lost 1x 'three'"])
    check("word proof catches an invented word",
          word_diff(src, lost["text"] + " five\n", "KEEP"), ["invented 1x 'five'"])
    check("word proof is silent on a clean rewrite", word_diff(src, lost["text"], "KEEP"), [])

    # ---- R1: punctuation is not a word. The delimiter a split leaves at the head of the
    # body proves on nothing — and nothing is deleted to make that true.
    dash_src = ("engine: gate the row so no lane strays past its neighbour -- the rest of "
                "the message runs on past the split point\n")
    r_dash = rewrite(dash_src)
    check("R1 case is a SPLIT", r_dash["class"], SPLIT)
    check("R1 deletes nothing: the delimiter still opens the body",
          r_dash["text"].split("\n")[2].startswith("-- the"), True)
    check("R1: a standalone delimiter is not a word, so it cannot fail a SPLIT",
          word_diff(dash_src, r_dash["text"].replace("-- the", "the"), "SPLIT"), [])
    check("R1 is not a licence to lose a word: a real word still fails",
          word_diff(dash_src, r_dash["text"].replace("gate", ""), "SPLIT"),
          ["lost 1x 'gate'"])
    em_dash = ("engine: pin the four state cells in the wave table \u2014 the tail clause "
               "runs long enough to need a split point at all\n")
    r_em = rewrite(em_dash)
    check("R1: a standalone em dash is not a word either",
          word_diff(em_dash, r_em["text"].replace("\u2014", ""), "SPLIT"), [])
    check("R1: a glued `,` is counted inside the word it is glued to",
          word_diff(p3, rewrite(p3)["text"].replace("collide,", "collide"), "SPLIT"),
          ["lost 1x 'collide,'", "invented 1x 'collide'"])
    check("R1 keeps the delimiter out of the split proof too (no candidate refused for it)",
          rewrite(dash_src)["header"],
          "engine: gate the row so no lane strays past its neighbour")

    # ---- D3: a wall paragraph becomes several, cut at sentence boundaries.
    sents = ["Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi.",
             "Second sentence is long enough to matter on its own, so the split is real.",
             "Third one here, and it is the one that has to start the next paragraph.",
             "Q3 carries the verdict, which is why this boundary is the one worth taking.",
             "The fifth sentence closes the wall and is allowed to be short."]
    wall = " ".join(sents)
    check("wall case is over WALL_MAX", len(wall) > WALL_MAX, True)
    chunks = split_wall(wall)
    check("wall split into several paragraphs", len(chunks) > 1, True)
    check("wall rejoins exactly", " ".join(chunks), wall)
    check("no wall paragraph under MIN_PARA except the last",
          [c for c in chunks[:-1] if len(c) < MIN_PARA], [])
    check("every closed wall paragraph ends on a sentence end",
          [c[-1] for c in chunks[:-1] if c[-1] not in ".!?;"], [])
    check("the enumerator boundary was taken", any(c.startswith("Q3") for c in chunks[1:]), True)
    # The enumerator rule on its own: two candidate boundaries, the plain one first.
    enum_in = ("x" * 200 + ". plain sentence goes here to push past the limit for a bit. "
               + "Q7 says what actually changed, and this one is the boundary we want.")
    check("the enumerator boundary is preferred over the plain one before it",
          split_wall(enum_in)[1].startswith("Q7"), True)
    # A stub sentence must not become a paragraph of its own.
    stub_in = "y" * 250 + ". No. " + "z" * 60
    check("a two-word sentence is not left as its own paragraph",
          [c for c in split_wall(stub_in) if len(c) < MIN_PARA], [])
    wall_msg = long_subj + "\n\n" + wall + "\n"
    r_wall = rewrite(wall_msg)
    check("wall re-fill keeps every word", word_diff(wall_msg, r_wall["text"], "KEEP"), [])
    check("wall re-fill produced blank-line separated paragraphs",
          r_wall["text"].count("\n\n") > 1, True)
    check("wall re-fill lint clean", lint(r_wall["text"], "KEEP"), [])

    # ---- R3: `;` at paren/bracket depth 0 is a sentence boundary; one inside a bracket is
    # not. Most long walls in this history are semicolon prose with no full stop at all.
    semi_wall = ("alpha " * 50 + "first claim; " + "beta " * 50 + "second claim; "
                 + "gamma " * 50 + "third claim")
    check("R3 case is a wall with no `.!?` anywhere", len(semi_wall) > WALL_MAX
          and not any(c in semi_wall for c in SENTENCE_END), True)
    check("R3: a depth-0 semicolon splits a wall that had no sentence boundary",
          len(split_wall(semi_wall)) > 1, True)
    semi_chunks = split_wall(semi_wall)
    check("R3 rejoins exactly", " ".join(semi_chunks), semi_wall)
    check("R3: every closed paragraph ends on a sentence end",
          [c[-1] for c in semi_chunks[:-1] if c[-1] not in ".!?;"], [])
    paren_wall = ("alpha " * 60 + "(read 240 GB/s; copy 210 GB/s; state 18 MiB) "
                  + "beta " * 60 + "tail")
    check("R3: a semicolon inside parentheses is not a boundary",
          split_wall(paren_wall), [paren_wall])
    check("R3 is not vacuous: the same wall without the parens does split",
          len(split_wall(paren_wall.replace("(", "").replace(")", ""))) > 1, True)
    check("R3 stays inside split_wall: a `;` still ends a SPLIT header",
          rewrite(p2)["header"].endswith("collide;"), True)

    # ---- D4: a header that stops mid-sentence is refused, and the rule can fail.
    d4_cases = {
        "unclosed paren": (
            "engine: sweep notes in ( -- the raw rows live in the ledger file anyway today\n",
            "engine: sweep notes in the ledger file from Q9 -- the raw rows are all here too\n"),
        "function word": (
            "engine: where the extra milliseconds actually are -- and the launch path is why\n",
            "engine: where the extra milliseconds actually live -- and the launch path is why\n"),
        "colon": (
            "engine: notes below: -- the ledger row for Q9 and why the whole thing moved here\n",
            "engine: notes below here for the record today -- the ledger row for Q9 moved\n"),
    }
    for name, (bad, good) in d4_cases.items():
        r_bad = rewrite(bad + "\nA body so the case is not NEEDS-BODY.\n")
        check(f"D4 rejects a header ending with an {name}", r_bad["class"], HARD)
        check(f"D4 says which shape it rejected for {name}",
              any(w in r_bad["note"] for w in ("unclosed", "function word", "colon")), True)
        r_good = rewrite(good + "\nA body so the case is not NEEDS-BODY.\n")
        check(f"D4 is not vacuous for {name}", r_good["class"], SPLIT)
        check(f"D4 kept a usable header for {name}", len(r_good["header"]) <= HEADER_MAX, True)
    check("D4 function word list is the one the ruling names",
          "its" in FUNCTION_WORDS and "ledger" not in FUNCTION_WORDS, True)

    # ---- D5: a message that is only a subject is NEEDS-BODY, not a broken header.
    nb = "spec: spell out the end-of-text token name in 02 and A-10\n"
    r = rewrite(nb)
    check("NEEDS-BODY class", r["class"], NEEDS_BODY)
    check("NEEDS-BODY header is the original subject", r["header"], r["subject"])
    check("NEEDS-BODY body is the sentinel", r["text"].split("\n")[2], TODO_BODY)
    check("NEEDS-BODY lint clean", lint(r["text"], NEEDS_BODY), [])
    check("NEEDS-BODY word proof clean as emitted", word_diff(nb, r["text"], NEEDS_BODY), [])
    check("NEEDS-BODY says what is owed", "no body at all" in r["note"], True)
    authored_nb = ("spec: spell out the end-of-text token name in 02 and A-10\n\n"
                   "The stop list held only the end-of-turn id, so 248044 was named by "
                   "number in 02 and A-10. Both files now spell the name.\n")
    check("NEEDS-BODY accepts an authored body", word_diff(nb, authored_nb, NEEDS_BODY), [])
    check("NEEDS-BODY rejects a rewritten header",
          "header is not the original subject verbatim" in
          word_diff(nb, "spec: renamed the token\n\nbody now exists\n", NEEDS_BODY), True)
    check("lint refuses the body sentinel outside NEEDS-BODY",
          lint("engine: a header of a decent length here\n\n" + TODO_BODY + "\n", "KEEP"),
          [f"{TODO_BODY} left in a file marked KEEP"])

    # ---- R4: an authored NEEDS-BODY body replaces the placeholder wholesale and may even
    # refill a hand-wrapped subject, but it may not swallow the subject or starve it.
    nb_wrap = "spec: spell out the end-of-text token name\nin 02 and A-10 as text\n"
    r_nbwrap = rewrite(nb_wrap)
    check("NEEDS-BODY joins a hand-wrapped subject into the one header line",
          r_nbwrap["text"].split("\n")[0],
          "spec: spell out the end-of-text token name in 02 and A-10 as text")
    check("NEEDS-BODY accepts a body written below the subject/body boundary",
          word_diff(nb_wrap, r_nbwrap["text"].replace(
              TODO_BODY, "Both files name the id in literal text now."), NEEDS_BODY), [])
    nb_boundary_diff = word_diff(
        nb_wrap, "spec: spell out the end-of-text token name\n"
                 "in 02 and A-10 as text\nThe stop list held only 248046.\n",
        NEEDS_BODY)
    check("NEEDS-BODY rejects a body written above the subject/body boundary",
          any("body text above the subject/body boundary" in d for d in nb_boundary_diff),
          True)
    nb_starved = word_diff(nb, "spec: spell out the token name\n\n"
                               "The stop list held only the end-of-turn id, so both files "
                               "now name it in literal text.\n", NEEDS_BODY)
    check("NEEDS-BODY fails when the subject's own words go missing from the file",
          "lost 1x '02'" in nb_starved and "lost 1x 'A-10'" in nb_starved, True)
    check("and it names the header that stopped being the subject",
          "header is not the original subject verbatim" in nb_starved, True)

    # ---- D6: a header that says nothing is THIN, and the body carries the stub.
    for thin_subject in ("plan: D-5", "results: P5", "gdn_gated_norm: \u00a79"):
        msg = thin_subject + "\n\nThere is a body here, mentioning 42 things.\n"
        r = rewrite(msg)
        check(f"THIN class for {thin_subject!r}", r["class"], THIN)
        check(f"THIN carries the stub in the body for {thin_subject!r}",
              r["text"].split("\n")[2], thin_subject)
        check(f"THIN word proof for {thin_subject!r}", word_diff(msg, r["text"], THIN), [])
        check(f"THIN is not ready for {thin_subject!r}", r["class"] in AUTHORED, True)
    r_thin = rewrite("results: P5\n\nThere is a body here, mentioning 42 things.\n")
    check("THIN header is the sentinel, not the stub",
          r_thin["text"].split("\n")[0], TODO_HEADER)
    check("lint lets the sentinel through while THIN", lint(r_thin["text"], THIN), [])
    check("lint catches the sentinel in a READY file even for THIN text",
          lint(r_thin["text"], "KEEP"), [f"{TODO_HEADER} left in a file marked READY (KEEP)"])
    check("a THIN note names the header it rejected",
          "plan: D-5" in rewrite("plan: D-5\n\nbody text here\n")["note"], True)

    # ---- the per-class word contract, in the direction an author needs.
    hard_src = _long_clause(18) + " -- " + "tail words here\n\nShips at 42 lanes.\n"
    hard_r = rewrite(hard_src)
    authored_hard = hard_r["text"].replace(
        TODO_HEADER, "engine: gate the wave at the row level instead of the kernel level")
    check("an authored header may add words (HARD)",
          word_diff(hard_src, authored_hard, HARD), [])
    check("a body that drops a number still fails (HARD)",
          word_diff(hard_src, authored_hard.replace("42", ""), HARD), ["lost 1x '42'"])
    check("an authored header may add words (THIN)",
          word_diff("results: P5\n\nThere is a body here, mentioning 42 things.\n",
                    r_thin["text"].replace(TODO_HEADER, "engine: results.md P5 verdict"), THIN),
          [])
    check("a body that drops a number still fails (THIN)",
          word_diff("results: P5\n\nThere is a body here, mentioning 42 things.\n",
                    r_thin["text"].replace("42", ""), THIN), ["lost 1x '42'"])
    check("KEEP may not add a word, only SPLIT/KEEP exact equality forbids it",
          word_diff(src, lost["text"] + "\nzorch\n", "KEEP"), ["invented 1x 'zorch'"])

    # ---- re-running the tool on its own output is a no-op (the SPLIT/KEEP half).
    again = rewrite(rewrite(p2)["text"])
    check("rewrite is idempotent", rewrite(again["text"])["text"], again["text"])
    check("rewritten output needs no further split", again["class"], "KEEP")

    # ---- end-to-end over the real history, into a temp directory, read-only.
    # Its own chatter is captured so a green selftest does not print red-looking lines.
    tmp = tempfile.mkdtemp(prefix="cr-selftest-")
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_emit = cmd_emit(tmp)
            n = len(rev_list("HEAD"))
            files = [f for f in os.listdir(tmp) if f.endswith(".msg")]
            rc_verify = cmd_verify(tmp)
        check("emit wrote one file per commit", (rc_emit, len(files)), (0, n))
        check("verify accepted the emitted set at exit 0", rc_verify, 0)
        check("round-trip lost no word",
              [l for l in buf.getvalue().split("\n") if l.startswith("FAIL")], [])
        # A tampered file must be caught, by name, and must fail the run. The victim is
        # picked by class: only a KEEP/SPLIT file is judged by exact equality, so only one
        # can witness the invention rule. A NEEDS-BODY file gets its own tamper below.
        manifest = load_manifest(tmp)
        ready = [c["file"] for c in manifest["commits"] if c["class"] in (KEEP, SPLIT)]
        check("the emitted set has a READY file to tamper with", bool(ready), True)
        victim = sorted(ready)[0]
        with open(os.path.join(tmp, victim), encoding="utf-8") as fh:
            tampered = fh.read()
        with open(os.path.join(tmp, victim), "w", encoding="utf-8") as fh:
            fh.write(tampered.replace("\n\n", "\n\nzorchfrumpin\n\n", 1))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_tampered = cmd_verify(tmp)
        check("verify names a tampered file", rc_tampered, 1)
        check("the tamper is the only complaint",
              [l for l in buf.getvalue().split("\n") if l.startswith("FAIL")],
              [l for l in buf.getvalue().split("\n") if l.startswith("FAIL")][:1])
        with open(os.path.join(tmp, victim), "w", encoding="utf-8") as fh:
            fh.write(tampered)
        with open(os.path.join(tmp, victim), "w", encoding="utf-8") as fh:
            fh.write(tampered)
        queue = [c["file"] for c in manifest["commits"] if c["class"] == NEEDS_BODY]
        if queue:
            # The NEEDS-BODY contract is "the header stays the original subject": rewriting
            # the header of such a file must be caught even though its body may grow.
            with open(os.path.join(tmp, queue[0]), encoding="utf-8") as fh:
                nb_text = fh.read()
            with open(os.path.join(tmp, queue[0]), "w", encoding="utf-8") as fh:
                fh.write("spec: a header somebody else wrote\n\n" + nb_text.split("\n", 2)[2])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_nb = cmd_verify(tmp)
            check("verify catches a rewritten NEEDS-BODY header", rc_nb, 1)
            check("and names it as the subject rule",
                  "not the original subject" in buf.getvalue(), True)
            with open(os.path.join(tmp, queue[0]), "w", encoding="utf-8") as fh:
                fh.write(nb_text)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_restored = cmd_verify(tmp)
        check("the restored set verifies clean again", rc_restored, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = 0
    for name, got, want in checks:
        if got != want:
            failed += 1
            print(f"SELFTEST FAIL {name}: got {got!r} want {want!r}")
    print(f"SELFTEST {'PASS' if not failed else 'FAIL'} ({len(checks)} checks, {failed} failed)")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="commit_msg_rewrite.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", metavar="OUTDIR", help="write <sha>.msg + manifest.json")
    group.add_argument("--verify", metavar="INDIR", help="word proof + lint over a written set")
    group.add_argument("--apply", metavar="INDIR",
                       help="REWRITE HISTORY with git filter-branch --msg-filter")
    group.add_argument("--verify-history", metavar=("OLDREF", "NEWREF", "CANDIDATES"), nargs="+",
                       help="prove only the messages changed between two refs; pass the "
                            "candidates dir to judge each commit by its own class")
    group.add_argument("--selftest", action="store_true", help="synthetic rule cases")
    args = ap.parse_args(argv)
    if args.emit:
        return cmd_emit(args.emit)
    if args.verify:
        return cmd_verify(args.verify)
    if args.apply:
        return cmd_apply(args.apply)
    if args.verify_history:
        vh = args.verify_history
        if len(vh) not in (2, 3):
            ap.error("--verify-history takes OLDREF NEWREF [CANDIDATES]")
        return cmd_verify_history(*vh)
    return selftest()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
