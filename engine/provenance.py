#!/usr/bin/env python3
"""Provenance digests for measurements that are pinned to a file's BEHAVIOUR, not its bytes.

Why this exists (2026-10-21). `engine/gate.py:allowance_inputs()` hashed `speccheck/specimpl.py` as
whole bytes, and a commit that moved documents out of `bench/` touched one COMMENT line in that file.
Result: five fixtures' cached route allowances became "stale" and `engine/gate.py` FAILED on
`dec300f`, `dec64f`, `dec64t`, `mt32f`, `wide520f` — red on the shipping gate for a change that could
not have moved a single number. The proof was cheap and decisive: the bytes differ (`e5acb50c…` ->
`cf28b17f…`) while the code-only AST digests are IDENTICAL (`a0d032ed…` both sides).

The gate's own docstring already states the right rule for the measurer — "a doc-comment edit to the
measurer is not a change to the measurement" — and applies it by EXCLUDING `route_price.py` from the
hash. This module applies the same rule to the reference itself, so the check fires on what it is for:
a change to the reference's arithmetic.

What the digest is: SHA-256 over `ast.dump()` of the parsed module with docstrings removed. Comments are
invisible to the parser; docstrings are stripped because they are strings the interpreter never uses in
the arithmetic. Names and literals ARE included, so a constant change, a reordered expression, a renamed
function or a changed default all move the digest. Two consequences worth stating out loud:

  * it is stable across reformatting, comment and docstring edits, and line moves — that is the point;
  * it depends on the running Python's AST shape (dumps are broadly stable but not a frozen ABI). If a
    future interpreter upgrade moves every digest at once, re-pin in ONE commit with a band-history
    line per fixture, exactly as this one's migration did, and never by repricing under it.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
from typing import Union

ROOT = pathlib.Path(__file__).resolve().parents[1]

_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr):
                val = getattr(body[0], "value", None)
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    body.pop(0)


def code_digest(path: Union[str, pathlib.Path]) -> str:
    """SHA-256 of the parsed code of `path`, docstrings stripped: behaviour, not bytes."""
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    _strip_docstrings(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def bytes_digest(path: Union[str, pathlib.Path]) -> str:
    """The old whole-file hash, kept so a migration can name what it left behind."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def specimpl_digest() -> str:
    """The digest a cached route allowance must match. One definition, used by the checker AND the
    pricer, because two spellings of a provenance rule is how the two sides stop agreeing."""
    return code_digest(ROOT / "speccheck" / "specimpl.py")
