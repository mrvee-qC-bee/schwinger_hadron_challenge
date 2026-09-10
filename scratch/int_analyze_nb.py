"""Analyse the executed solution notebook: errors, per-cell timing, grader verdict lines."""
import nbformat, sys, re, datetime as dt
nb = nbformat.read("schwinger_hadron_solution.ipynb", as_version=4)
errs, rows, verdicts = [], [], []
def ts(s): return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            errs.append((i, o.get("ename"), o.get("evalue", "")[:300]))
        txt = o.get("text", "") if o.get("output_type") == "stream" else ""
        for m in re.finditer(r"^\[(ex[\d.]+|B\d)\] .*$", txt, re.M):
            verdicts.append(m.group(0)[:160])
    ex = c.metadata.get("execution", {})
    if "shell.execute_reply" in ex and "iopub.execute_input" in ex:
        d = (ts(ex["shell.execute_reply"]) - ts(ex["iopub.execute_input"])).total_seconds()
        first = c.source.strip().splitlines()[0][:70] if c.source.strip() else ""
        rows.append((d, i, first))
print("ERRORS:", len(errs))
for e in errs: print("  cell", *e)
tot = sum(r[0] for r in rows)
print(f"total cell time {tot:.0f} s over {len(rows)} code cells")
print("slowest cells:")
for d, i, f in sorted(rows, reverse=True)[:12]: print(f"  {d:7.1f} s  cell {i:3d}  {f}")
print("grader verdicts:")
for v in verdicts: print("  ", v)
n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
n_img = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
print("outputs:", n_out, "png figures:", n_img)
