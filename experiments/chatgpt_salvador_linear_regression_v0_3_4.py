"""Small sanity checks for the Salvador-style auxiliary candidate in Furones v0.3.4."""
import json
import networkx as nx
from furones.algorithm import find_dominating_set, salvador_planar_bipartite_baker_candidate


def row(name, G):
    aux = salvador_planar_bipartite_baker_candidate(G)
    sol = find_dominating_set(G)
    return {
        "name": name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "auxiliary_candidate_size": len(aux),
        "auxiliary_candidate_valid": nx.is_dominating_set(G, aux) if aux else False,
        "final_size": len(sol),
        "final_valid": nx.is_dominating_set(G, sol),
        "final_solution": sorted(map(str, sol)),
    }

cases = [
    row("path_12", nx.path_graph(12)),
    row("cycle_12", nx.cycle_graph(12)),
    row("complete_bipartite_4_5", nx.complete_bipartite_graph(4, 5)),
]
result = {"version": "0.3.4", "long_exhaustive_battery_rerun": False, "cases": cases}
print(json.dumps(result, indent=2))
with open("experiments/chatgpt_salvador_auxiliary_regression_v0_3_4.json", "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
    fh.write("\n")
