"""Stub grader for the nbconvert smoke run (prints and returns 0)."""
def check_env():
    print("grader stub: check_env"); return 0
def summary():
    print("grader stub: summary"); return 0
def __getattr__(name):
    if name.startswith("grade_"):
        def _g(*a, **k):
            print(f"grader stub: {name}({len(a)} args)"); return 0
        return _g
    raise AttributeError(name)
