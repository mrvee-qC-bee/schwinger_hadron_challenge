import sys, time, math; sys.path[:0]=['.', 'organizer']
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKingston
b=FakeKingston()

def target_error_tables(backend, dead=0.5):
    tg=backend.target
    cz={}
    for q,p in tg['cz'].items():
        e=p.error if p is not None and p.error is not None and not math.isnan(p.error) else 1.0
        cz[tuple(sorted(q))]=min(max(e, cz.get(tuple(sorted(q)),0.0)),1.0) if tuple(sorted(q)) in cz else e
    ro={q[0]:(p.error if p is not None and p.error is not None else 1.0) for q,p in tg['measure'].items()}
    sx={q[0]:(p.error if p is not None and p.error is not None else 1.0) for q,p in tg['sx'].items()}
    return cz, ro, sx

def chain_cost(chain, cz, ro, sx, n_cz_per_bond=79.0, n_sx_per_qubit=155.0):
    def nl(e): return -math.log(max(1e-12, 1.0-min(e, 0.999)))
    c=0.0
    for a,bq in zip(chain[:-1], chain[1:]): c+=n_cz_per_bond*nl(cz[tuple(sorted((a,bq)))])
    for q in chain: c+=nl(ro[q])+n_sx_per_qubit*nl(sx[q])
    return c

def select_chain(backend, n_qubits=68, time_budget=15.0, seed=0, dead=0.5):
    cz,ro,sx=target_error_tables(backend)
    adj={q:set() for q in range(backend.target.num_qubits)}
    for (a,bq),e in cz.items():
        if e<dead and ro[a]<dead and ro[bq]<dead: adj[a].add(bq); adj[bq].add(a)
    rng=np.random.default_rng(seed)
    def nl(e): return -math.log(max(1e-12,1.0-min(e,0.999)))
    edge_w={e:79.0*nl(v) for e,v in cz.items()}
    node_w={q:nl(ro[q])+155.0*nl(sx[q]) for q in ro}
    best=None; best_cost=float('inf'); n_restart=0
    t0=time.time()
    sys.setrecursionlimit(10000)
    while time.time()-t0<time_budget:
        start=int(rng.choice([q for q in adj if adj[q]]))
        path=[start]; used={start}; found=None
        temp=rng.uniform(0.0, 1.0)
        def dfs():
            nonlocal found
            if found is not None or time.time()-t0>time_budget: return
            if len(path)==n_qubits: found=list(path); return
            last=path[-1]
            cand=[q for q in adj[last] if q not in used]
            cand.sort(key=lambda q: edge_w[tuple(sorted((last,q)))]+node_w[q]+temp*rng.exponential(0.05))
            for q in cand:
                path.append(q); used.add(q); dfs()
                if found is not None: return
                path.pop(); used.discard(q)
        dfs(); n_restart+=1
        if found is not None:
            c=chain_cost(found,cz,ro,sx)
            if c<best_cost: best_cost, best = c, found
    return best, best_cost, n_restart

t0=time.time(); chain,c,nr=select_chain(b, time_budget=10); print(time.time()-t0, nr, c, chain)
cz,ro,sx=target_error_tables(b)
lay=[141, 142, 143, 136, 123, 124, 125, 126, 127, 128, 129, 118, 109, 110, 111, 98, 91, 90, 89, 78, 69, 68, 67, 57, 47, 48, 49, 50, 51, 58, 71, 72, 73, 74, 75, 59, 55, 54, 53, 39, 33, 34, 35, 19, 15, 14, 13, 12, 11, 18, 31, 30, 29, 28, 27, 17, 7, 6, 5, 4, 3, 16, 23, 22, 21, 36, 41, 42]
print("transpiler layout cost", chain_cost(lay,cz,ro,sx))
print("edges used by transpiler with err:", sorted([cz[tuple(sorted((a,bb)))] for a,bb in zip(lay[:-1],lay[1:])])[-5:], " ro:", sorted([ro[q] for q in lay])[-5:])
print("chain worst:", sorted([cz[tuple(sorted((a,bb)))] for a,bb in zip(chain[:-1],chain[1:])])[-5:], sorted([ro[q] for q in chain])[-5:])
# check valid path
cm=set(tuple(sorted(e)) for e in b.coupling_map.get_edges())
print("valid", all(tuple(sorted((a,bb))) in cm for a,bb in zip(chain[:-1],chain[1:])), len(set(chain)))
