"""Targeted v0.3.4 regression for random/boosted set-cover adversaries."""
from __future__ import annotations

import json, random
from pathlib import Path
from typing import Any
import networkx as nx
from furones.algorithm import find_dominating_set


def random_set_cover_graph(q:int,b:int,p:float,seed:int=0,beta:int=0)->nx.Graph:
    rng=random.Random(seed)
    G=nx.Graph()
    D=[f"d{i}" for i in range(q)]
    P=["p0","p1"]
    E0=[f"e0_{j}" for j in range(b)]
    E1=[f"e1_{j}" for j in range(b)]
    B=[f"b{j}" for j in range(2 * beta)]
    # requested order: decoys first, then planted, then elements, then boosters
    G.add_nodes_from(D+P+E0+E1+B)
    selector=P+D
    for i,u in enumerate(selector):
        for v in selector[i+1:]:
            G.add_edge(u,v)
    # planted pair dominates element blocks
    for e in E0: G.add_edge("p0", e)
    for e in E1: G.add_edge("p1", e)
    # random decoy-element coverage
    elements=E0+E1
    for d in D:
        for e in elements:
            if rng.random() < p:
                G.add_edge(d,e)
    # two beta-sized booster blocks increase decoy degrees while preserving {p0,p1} as a dominating set.
    # They are adjacent to p0 and all decoys, not to p1.
    for booster in B:
        G.add_edge("p0", booster)
        for d in D:
            G.add_edge(d, booster)
    return G


def row(name:str,G:nx.Graph,opt:int=2)->dict[str,Any]:
    D=set(find_dominating_set(G))
    valid=nx.is_dominating_set(G,D)
    return {"name":name,"n":G.number_of_nodes(),"m":G.number_of_edges(),"opt":opt,"output_size":len(D),"ratio":len(D)/opt,"valid":valid,"output":sorted(map(str,D))}


def main():
    cases=[
        ("random_q80_b2400_p0.5_s0", dict(q=80,b=2400,p=0.5,seed=0,beta=0)),
        ("random_q120_b2400_p0.5_s0", dict(q=120,b=2400,p=0.5,seed=0,beta=0)),
        ("random_q160_b2400_p0.5_s0", dict(q=160,b=2400,p=0.5,seed=0,beta=0)),
        ("random_q200_b2400_p0.5_s0", dict(q=200,b=2400,p=0.5,seed=0,beta=0)),
        ("boosted_q160_b2400_p0.35_beta1000_s0", dict(q=160,b=2400,p=0.35,seed=0,beta=1000)),
        ("boosted_q160_b4800_p0.35_beta2000_s0", dict(q=160,b=4800,p=0.35,seed=0,beta=2000)),
    ]
    out=[]
    for name,kw in cases:
        G=random_set_cover_graph(**kw)
        out.append(row(name,G,2))
        print(out[-1])
    result={"metadata":{"solver_version":"0.3.4","experiment":"targeted random and boosted set-cover regression","long_exhaustive_battery_rerun":False},"summary":{"instances":len(out),"all_valid":all(r['valid'] for r in out),"max_ratio":max(r['ratio'] for r in out)},"cases":out}
    Path(__file__).with_name("chatgpt_ratio75_regression.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
