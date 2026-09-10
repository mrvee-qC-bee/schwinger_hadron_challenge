"""Shared setup for the adversarial grader review scripts (scratch only)."""
import contextlib, io, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup(tag):
    sub = os.path.join(ROOT, "scratch", f"review_sub_{tag}")
    os.makedirs(sub, exist_ok=True)
    for f in os.listdir(sub):
        p = os.path.join(sub, f)
        if os.path.isfile(p):
            os.remove(p)
    os.environ["FF_SUBMISSION_DIR"] = sub
    sys.path[:0] = [ROOT, os.path.join(ROOT, "organizer")]
    import fallfest_grader as ff
    ff.SUBMISSION_DIR = sub
    return ff

def run(label, fn, *a, **kw):
    buf = io.StringIO(); t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            pts = fn(*a, **kw)
        line = buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ""
        print(f"{label:60s} -> {pts:5.1f}   {line[:170]}  [{time.time()-t0:.1f}s]")
        return pts
    except Exception as e:
        print(f"{label:60s} -> RAISED {type(e).__name__}: {str(e)[:150]}  [{time.time()-t0:.1f}s]")
        return None
