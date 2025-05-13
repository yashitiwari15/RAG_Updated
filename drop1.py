import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
DFA = nx.DiGraph()

# Add nodes with labels
DFA.add_node("q0", label="q0 (Start)")
DFA.add_node("q1", label="q1")
DFA.add_node("q2", label="q2 (Accept)")
DFA.add_node("qd", label="q_d (Dead)")

# Add edges with labels for transitions
DFA.add_edge("q0", "q0", label="a")
DFA.add_edge("q0", "q1", label="b")
DFA.add_edge("q1", "q1", label="b")
DFA.add_edge("q1", "qd", label="a")
DFA.add_edge("qd", "qd", label="a,b")

# Layout for graph visualization
pos = nx.spring_layout(DFA)

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(DFA, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")

# Add edge labels
edge_labels = nx.get_edge_attributes(DFA, "label")
nx.draw_networkx_edge_labels(DFA, pos, edge_labels=edge_labels, font_size=10)

# Highlight start and accept states
nx.draw_networkx_nodes(DFA, pos, nodelist=["q0"], node_color="lightgreen", node_size=3000)  # Start state
nx.draw_networkx_nodes(DFA, pos, nodelist=["q2"], node_color="lightcoral", node_size=3000)  # Accept state

plt.title("DFA for L = {a^m b^n | m, n ≥ 1}")
plt.show()
