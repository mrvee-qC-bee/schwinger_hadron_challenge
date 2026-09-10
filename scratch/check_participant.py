"""Execute the participant notebook's cells in order until the first '# YOUR CODE HERE'."""
import json, sys, os, io, contextlib
sys.path.insert(0, os.getcwd())
os.environ["MPLBACKEND"]="Agg"
nb=json.load(open("schwinger_hadron_participant.ipynb"))
ns={"__name__":"__main__"}
ran=0
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    src="".join(c["source"])
    if "# YOUR CODE HERE" in src:
        print(f"stopped at the first prompt: cell {i}")
        break
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src,f"<cell {i}>","exec"), ns)
        ran+=1
    except SystemExit:
        print(f"cell {i}: SystemExit (ok)"); ran+=1
    except Exception as e:
        print(f"CELL {i} FAILED: {type(e).__name__}: {e}\n{src[:300]}"); sys.exit(1)
print(f"{ran} pre-prompt cells ran cleanly")
