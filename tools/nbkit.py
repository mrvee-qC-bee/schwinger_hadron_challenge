"""
Tiny toolkit for building the participant and solution notebooks from ONE cell list.

Cell helpers
------------
    md(text, tags=())          -> markdown cell dict
    code(src, tags=())         -> code cell dict

Conventions inside code cells (see organizer/SPEC.md section B)
----------------------------------------------------------------
    # PROMPT: ...                 kept in both versions
    # BEGIN ANSWER ... # END ANSWER
                                  solution lines; in the participant build the body is replaced by a
                                  single "# YOUR CODE HERE" line (same indentation as the BEGIN marker),
                                  except lines ending with "# KEEP", which are kept verbatim.
    name = # <description>        QDC-style blank: written in the SOLUTION as
                                  "name = <expr>  # SOL" and rendered for participants as
                                  "name = # <description>" by the pair convention below.

    Pair convention for one-liners (optional, keeps QDC look-and-feel):
        # PARTICIPANT: qc_isa = # transpiled circuit, of type QuantumCircuit
        qc_isa = pm.run(qc)   # SOL
      -> participant build shows the PARTICIPANT line (without the marker) and drops the SOL line;
      -> solution build drops the PARTICIPANT line and keeps the SOL line (without the marker).

Tags
----
    'solution-only'     cell dropped from the participant notebook
    'participant-only'  cell dropped from the solution notebook
    'hardware'          informational; cell must be guarded by RUN_ON_HARDWARE

Build
-----
    build_notebook(cells, participant: bool) -> nbformat.NotebookNode
    write_notebook(nb, path)
"""
from __future__ import annotations

import re
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

_BEGIN = re.compile(r"^(\s*)# BEGIN ANSWER\b")
_END = re.compile(r"^\s*# END ANSWER\b")
_KEEP = re.compile(r"#\s*KEEP\s*$")
_PART = re.compile(r"^(\s*)# PARTICIPANT:\s?(.*)$")
_SOL = re.compile(r"^(.*?)\s*#\s*SOL\s*$")


def md(text: str, tags=()) -> dict:
    return {"cell_type": "markdown", "source": text.strip("\n") + "\n", "tags": list(tags)}


def code(src: str, tags=()) -> dict:
    return {"cell_type": "code", "source": src.strip("\n") + "\n", "tags": list(tags)}


def strip_answers(src: str) -> str:
    """Participant view of a solution cell source."""
    out = []
    inside = False
    indent = ""
    for line in src.splitlines():
        m = _BEGIN.match(line)
        if m:
            inside = True
            indent = m.group(1)
            out.append(line)
            out.append(f"{indent}# YOUR CODE HERE")
            continue
        if _END.match(line):
            inside = False
            out.append(line)
            continue
        if inside:
            if _KEEP.search(line):
                out.append(_KEEP.sub("", line).rstrip())
            continue
        mp = _PART.match(line)
        if mp:
            out.append(f"{mp.group(1)}{mp.group(2)}")
            continue
        if _SOL.match(line):
            continue
        if _KEEP.search(line):
            out.append(_KEEP.sub("", line).rstrip())
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def solution_view(src: str) -> str:
    """Solution view: drop PARTICIPANT lines, strip SOL/KEEP markers."""
    out = []
    for line in src.splitlines():
        if _PART.match(line):
            continue
        ms = _SOL.match(line)
        if ms:
            out.append(ms.group(1))
            continue
        if _KEEP.search(line):
            out.append(_KEEP.sub("", line).rstrip())
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def build_notebook(cells: list[dict], participant: bool) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for c in cells:
        tags = set(c.get("tags", []))
        if participant and "solution-only" in tags:
            continue
        if (not participant) and "participant-only" in tags:
            continue
        if c["cell_type"] == "markdown":
            cell = new_markdown_cell(c["source"])
        else:
            src = strip_answers(c["source"]) if participant else solution_view(c["source"])
            cell = new_code_cell(src)
        if tags:
            cell.metadata["tags"] = sorted(tags)
        nb.cells.append(cell)
    return nb


def write_notebook(nb: nbformat.NotebookNode, path: str) -> None:
    nbformat.validate(nb)
    with open(path, "w") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":  # tiny self-test
    src = """def f(L):  # KEEP
    # PROMPT: fill in
    # BEGIN ANSWER
    x = 1
    y = 2  # KEEP
    # END ANSWER
    return x
# PARTICIPANT: qc_isa = # transpiled circuit
qc_isa = pm.run(qc)  # SOL
"""
    print("--- participant ---"); print(strip_answers(src))
    print("--- solution ---"); print(solution_view(src))
