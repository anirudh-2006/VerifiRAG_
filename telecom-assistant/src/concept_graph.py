# src/concept_graph.py
import networkx as nx
import json
import os

GRAPH_PATH = "data/concept_graph.json"

def build_telecom_graph():
    """
    Build a directed graph where an edge A → B means
    'A is a prerequisite for B' / 'you need A to understand B'
    """
    G = nx.DiGraph()

    # --- Define all concepts with metadata ---
    concepts = [
        # (concept_name, level, keywords_for_retrieval)
        ("complex_numbers",       "foundation",    ["complex numbers", "imaginary unit", "phasor", "magnitude phase"]),
        ("differential_equations","foundation",    ["differential equation", "ODE", "initial conditions"]),
        ("fourier_series",        "foundation",    ["fourier series", "harmonic", "periodic signal", "coefficients ak"]),
        ("sampling_theorem",      "foundation",    ["sampling theorem", "nyquist", "sampling rate"]),
        ("fourier_transform",     "intermediate",  ["fourier transform", "X(jw)", "frequency domain", "rectangular pulse", "sinc"]),
        ("laplace_transform",     "intermediate",  ["laplace transform", "X(s)", "region of convergence", "ROC", "poles zeros"]),
        ("z_transform", "intermediate", [
            "z transform", "Z-transform", "DTFT", "discrete fourier",
            "X(z)", "unit circle", "discrete time", "causal exponential"
        ]),
        ("convolution",           "intermediate",  ["convolution", "impulse response", "h(t)", "h[n]"]),
        ("transfer_function",     "advanced",      ["transfer function", "H(s)", "H(z)", "system function"]),
        ("frequency_response",    "advanced",      ["frequency response", "bode plot", "magnitude response", "H(jw)"]),
        ("modulation",            "advanced",      ["modulation", "AM", "FM", "carrier signal", "bandwidth"]),
        ("filters",               "advanced",      ["filter", "low pass", "high pass", "band pass", "cutoff frequency"]),
        ("feedback_systems",      "advanced",      ["feedback", "closed loop", "stability", "pole placement"]),
    ]

    for name, level, keywords in concepts:
        G.add_node(name, level=level, keywords=keywords)

    # --- Define prerequisite edges (A → B means A is needed for B) ---
    prerequisites = [
        # foundations feed into intermediates
        ("complex_numbers",        "fourier_transform"),
        ("complex_numbers",        "laplace_transform"),
        ("complex_numbers",        "z_transform"),
        ("differential_equations", "laplace_transform"),
        ("differential_equations", "fourier_transform"),
        ("fourier_series",         "fourier_transform"),
        ("sampling_theorem",       "z_transform"),

        # intermediates feed into advanced
        ("fourier_transform",      "frequency_response"),
        ("fourier_transform",      "modulation"),
        ("fourier_transform",      "filters"),
        ("laplace_transform",      "transfer_function"),
        ("laplace_transform",      "frequency_response"),
        ("z_transform",            "transfer_function"),
        ("z_transform",            "frequency_response"),
        ("convolution",            "transfer_function"),
        ("convolution",            "filters"),

        # advanced interdependencies
        ("transfer_function",      "feedback_systems"),
        ("transfer_function",      "filters"),
        ("frequency_response",     "filters"),
        ("frequency_response",     "modulation"),
    ]

    for src, dst in prerequisites:
        G.add_edge(src, dst)

    return G

def get_prerequisites(G, concept: str, depth: int = 2) -> list:
    """
    Given a concept, return all prerequisites up to `depth` levels back.
    depth=1 → immediate prerequisites only
    depth=2 → prerequisites of prerequisites too (recommended)
    """
    if concept not in G:
        return []

    prerequisites = set()
    # walk backwards through the graph
    current_level = {concept}

    for _ in range(depth):
        prev_level = set()
        for node in current_level:
            parents = set(G.predecessors(node))
            prev_level.update(parents)
        prerequisites.update(prev_level)
        current_level = prev_level

    # return sorted by level so foundations come first
    level_order = {"foundation": 0, "intermediate": 1, "advanced": 2}
    result = sorted(
        prerequisites,
        key=lambda n: level_order.get(G.nodes[n].get("level", "foundation"), 0)
    )
    return result

def identify_concepts(query: str, G) -> list:
    """
    Look at the query and return which concept nodes it matches.
    Simple keyword matching — good enough for now.
    """
    query_lower = query.lower()
    matched = []

    for node, data in G.nodes(data=True):
        keywords = data.get("keywords", [])
        for kw in keywords:
            if kw.lower() in query_lower:
                matched.append(node)
                break  # one match per node is enough

    return matched

def save_graph(G, path=GRAPH_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved to {path}")

def load_graph(path=GRAPH_PATH):
    with open(path, "r") as f:
        data = json.load(f)
    return nx.node_link_graph(data)

if __name__ == "__main__":
    G = build_telecom_graph()
    save_graph(G)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # test it
    test_concept = "frequency_response"
    prereqs = get_prerequisites(G, test_concept, depth=2)
    print(f"\nPrerequisites for '{test_concept}':")
    for p in prereqs:
        level = G.nodes[p].get("level")
        print(f"  {p} ({level})")

    test_query = "explain the frequency response of a discrete time filter"
    matched = identify_concepts(test_query, G)
    print(f"\nConcepts matched in query: {matched}")