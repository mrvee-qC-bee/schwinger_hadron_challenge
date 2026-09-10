"""Corrected native rxx/ryy Trotter step (angle dt/4) -- a valid alternative gate set."""
from review_grader_common import setup, run, ROOT
import os, sys, numpy as np
ff = setup("rxx")
import schwinger_reference as R, challenge_utils as cu
sys.path.insert(0, os.path.join(ROOT, "tools")); import test_grader as T
ff.SUBMISSION_DIR = os.path.join(ROOT, "scratch", "review_sub_rxx")
def rz_layer(qc, L, dt, g):
    for k in range(L // 2 - 1):
        qc.rz(g**2 * dt, 2 * k); qc.rz(0.5 * g**2 * dt, 2 * k + 1)
    qc.rz(0.5 * g**2 * dt, L - 2); qc.rz(-0.5 * g**2 * dt, L + 1)
    for k in range(1, L // 2):
        qc.rz(-0.5 * g**2 * dt, L + 2 * k); qc.rz(-g**2 * dt, L + 2 * k + 1)
def step_native(qc, L, dt, m, g):
    n = 2 * L
    def kin(js):
        for j in js: qc.rxx(dt / 4, j, j + 1); qc.ryy(dt / 4, j, j + 1)
    kin(range(1, n - 1, 2)); kin(range(0, n - 1, 2))
    for j in range(n): qc.rz((-1) ** j * m * dt, j)
    rz_layer(qc, L, dt, g); qc = cu.trotter_step_electric_2q(qc, L, dt, g)
    kin(range(0, n - 1, 2)); kin(range(1, n - 1, 2))
    return qc
run("ex0.5 native rxx/ryy(dt/4) kinetic gates, mass layer first", ff.grade_ex0_5, step_native, T.make_evolve(step_native), R.prep_wave)
