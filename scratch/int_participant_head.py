"""Execute the participant notebook's code cells up to (not including) the first cell with '# YOUR CODE HERE'."""
import nbformat, time, sys, os, builtins
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
nb = nbformat.read("schwinger_hadron_participant.ipynb", as_version=4)
builtins.display = lambda *a, **k: print("[display]", *[type(x).__name__ for x in a])
ns = {"__name__": "__main__"}
n = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    if "# YOUR CODE HERE" in c.source:
        print(f"first prompt at cell {i}; executed {n} code cells before it"); break
    t0 = time.time()
    exec(compile(c.source, f"cell{i}", "exec"), ns)
    n += 1
    print(f"cell {i} ok ({time.time()-t0:.1f}s)")
print("PARTICIPANT HEAD OK")
