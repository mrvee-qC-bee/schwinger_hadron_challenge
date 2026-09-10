import sys, os, time, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
refs=np.load(os.path.join(ROOT,'reference_data','grader_refs.npz'))
def summary(name):
    return {k: refs[f'tgt_{name}_{k}'] for k in ['num_qubits','edges','cz_error','sx_error','x_error','readout_error','t1','t2']}
def cost_tables(s):
    nq=int(s['num_qubits']); node=np.full(nq,np.inf); ro=s['readout_error']
    ok=np.isfinite(ro)&(ro<0.5); node[ok]=-np.log1p(-ro[ok])
    adj={q:{} for q in range(nq)}
    for (a,b),e in zip(s['edges'],s['cz_error']):
        if np.isfinite(e) and e<0.5:
            c=-np.log1p(-e); adj[int(a)][int(b)]=c; adj[int(b)][int(a)]=c
    return node,adj
def chain_cost(s, chain):
    node,adj=cost_tables(s); c=0.0
    for q in chain: c+=node[q]
    for a,b in zip(chain[:-1],chain[1:]):
        if b not in adj[a]: return np.inf
        c+=adj[a][b]
    return c
def search(s, n=68, seed=2026, restarts=300, budget=10.0, temp=0.3, expand_cap=20000):
    node,adj=cost_tables(s); nq=len(node); rng=np.random.default_rng(seed)
    good=[q for q in range(nq) if np.isfinite(node[q]) and adj[q]]
    best=(np.inf,None); t0=time.time(); done=0
    for r in range(restarts):
        if time.time()-t0>budget: break
        start=int(rng.choice(good)); path=[start]; visited={start}
        # iterative DFS with candidate stacks
        def cands(q):
            cs=[(adj[q][v]+node[v]+temp*rng.gumbel()*0.01, v) for v in adj[q] if v not in visited]
            cs.sort(); return [v for _,v in cs]
        stack=[cands(start)]; expansions=0
        while stack and len(path)<n and expansions<expand_cap:
            if stack[-1]:
                v=stack[-1].pop(0); path.append(v); visited.add(v); stack.append(cands(v)); expansions+=1
            else:
                stack.pop(); visited.discard(path.pop())
        done+=1
        if len(path)==n:
            c=chain_cost(s,path)
            if c<best[0]: best=(c,list(path))
    return best, done, time.time()-t0
for name in ['kingston','fez','marrakesh']:
    s=summary(name)
    for temp in [0.0, 0.3, 1.0, 3.0]:
        (c,p),done,dt=search(s, temp=temp)
        print(f"{name} temp={temp}: best cost={c:.4f} restarts={done} {dt:.1f}s path[:5]={p[:5] if p else None}", flush=True)
# transpiler O3 layout cost on kingston
lay=list(refs['layout_t8_kingston_O3_seed42']); print("kingston O3 layout cost", chain_cost(summary('kingston'), lay))
lay_boston=[113, 119, 133, 134, 135, 139, 155, 154, 153, 152, 151, 138, 131, 130, 129, 128, 127, 126, 125, 117, 105, 106, 107, 97, 87, 88, 89, 78, 69, 68, 67, 66, 65, 64, 63, 56, 43, 44, 45, 37, 25, 24, 23, 16, 3, 4, 5, 6, 7, 17, 27, 28, 29, 30, 31, 32, 33, 39, 53, 54, 55, 59, 75, 74, 73, 79, 93, 94]
hw=np.load(os.path.join(ROOT,'reference_data','hardware_ibm_kingston_2026-07-25.npz')); print("kingston hardware layout cost", chain_cost(summary('kingston'), list(hw['initial_layout'])), "on fez:", chain_cost(summary('fez'), list(hw['initial_layout'])))
