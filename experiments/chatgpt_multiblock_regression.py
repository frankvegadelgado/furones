from __future__ import annotations
import json, random
from pathlib import Path
import networkx as nx
from furones.algorithm import find_dominating_set


def multiblock_adversary(k:int, q:int, b:int, p:float, beta:int, seed:int=0)->nx.Graph:
    rng=random.Random(seed)
    G=nx.Graph()
    P=[f"p{i}" for i in range(k)]
    D=[f"d{j}" for j in range(q)]
    E=[[f"e{i}_{a}" for a in range(b)] for i in range(k)]
    B=[[f"b{i}_{a}" for a in range(beta)] for i in range(k)]
    # node order: decoys, planted, elements, boosters
    G.add_nodes_from(D); G.add_nodes_from(P)
    for block in E: G.add_nodes_from(block)
    for block in B: G.add_nodes_from(block)
    selector=P+D
    for i,u in enumerate(selector):
        for v in selector[i+1:]:
            G.add_edge(u,v)
    for i,pi in enumerate(P):
        for e in E[i]: G.add_edge(pi,e)
        for bb in B[i]: G.add_edge(pi,bb)
    allE=[e for block in E for e in block]
    allB=[bb for block in B for bb in block]
    for d in D:
        for e in allE:
            if rng.random()<p:
                G.add_edge(d,e)
        for bb in allB:
            G.add_edge(d,bb)
    return G

def old_two_block_boosted(q=160,b=4800,p=0.35,beta=2000,seed=0):
    rng=random.Random(seed)
    G=nx.Graph(); P=['p0','p1']; D=[f'd{j}' for j in range(q)]
    E0=[f'e0_{i}' for i in range(b)]; E1=[f'e1_{i}' for i in range(b)]
    B=[f'b{i}' for i in range(beta)]
    G.add_nodes_from(D+P+E0+E1+B)
    selector=P+D
    for i,u in enumerate(selector):
        for v in selector[i+1:]: G.add_edge(u,v)
    for e in E0: G.add_edge('p0',e)
    for e in E1: G.add_edge('p1',e)
    for bb in B: G.add_edge('p0',bb)
    for d in D:
        for e in E0+E1:
            if rng.random()<p: G.add_edge(d,e)
        for bb in B: G.add_edge(d,bb)
    return G

def row(name,G,k):
    S=set(find_dominating_set(G))
    valid=nx.is_dominating_set(G,S)
    return {"name":name,"n":G.number_of_nodes(),"m":G.number_of_edges(),"known_ds_size":k,"output_size":len(S),"certified_ratio":len(S)/k,"valid":valid,"output":sorted(S,key=str)}

def main():
    cases=[]
    cases.append(row('old_two_block_boosted_q160_b4800_p0.35_beta2000', old_two_block_boosted(), 2))
    for k,q,b,p,beta in [(8,200,200,0.10,30),(8,200,200,0.05,30),(8,200,200,0.04,30),(8,240,200,0.04,30),(10,200,200,0.08,30)]:
        cases.append(row(f'multiblock_k{k}_q{q}_b{b}_p{p}_beta{beta}', multiblock_adversary(k,q,b,p,beta), k))
    res={"metadata":{"solver_version":"0.3.4","experiment":"targeted multiblock planted-dominator regression","long_exhaustive_battery_rerun":False},"summary":{"instances":len(cases),"all_valid":all(c['valid'] for c in cases),"max_certified_ratio":max(c['certified_ratio'] for c in cases)},"cases":cases}
    out=Path(__file__).with_name('chatgpt_multiblock_regression_v0_3_4.json')
    out.write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
