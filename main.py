import os
from math import ceil
from typing import Dict, Tuple, List, Hashable, Any, Set, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from numpy import ndarray
from pandas import DataFrame
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
import matplotlib.pyplot as plt

try:
    from ipywidgets import IntSlider, fixed, interact
except ImportError:
    IntSlider = None
    fixed = None
    interact = None

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None

try:
    from IPython.display import display
except ImportError:
    display = None


PRINT_MODE: bool = True
PLOT_MODE: bool = True


########## CLASSES ##########

class Representation:
    entity: str
    matrix: np.ndarray
    graph: nx.Graph
    labels: List[Hashable]

    def __init__(self, entity: str, labels: List[Hashable], groups: List[List[Hashable]]) -> None:
        self.entity = entity
        self.labels = labels
        self.graph = nx.Graph()
        self.graph, self.matrix = build_graph_and_adjacency_matrix(self.labels, groups)


    def compute_graph_summary(self) -> Dict[str, object]:
        component_sizes = [len(component) for component in nx.connected_components(self.graph)]

        return {
            "graph": self.entity,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }

    def plot_adjacency_matrix_heatmap(self, color: str) -> None:
        size = ceil(self.matrix.shape[0] * 0.22)
        plot_heatmap(self.matrix, self.labels, self.labels, (size, size), x_on_top=True,
                     title=f"{self.entity} Projection Adjacency Matrix",
                     file_name=f"Adjacency_Matrix_{self.entity}.png", color_map=color)


    def plot_graph(self, min_weight: float | None = None, max_edges: int | None = None, label_top_n: int = 0,
                   show_isolated_nodes: bool = False, node_color: str = "deeppink") -> None:
        networkx_graph = get_display_graph(self.graph, min_weight, max_edges, show_isolated_nodes)
        node_count = networkx_graph.number_of_nodes()

        figure_size = (14, 14) if node_count <= 50 else (22, 22)
        figure, axis = plt.subplots(figsize=figure_size)

        positions = get_graph_layout(networkx_graph)
        draw_projection_graph(axis, networkx_graph, positions, node_color, label_top_n,
                              f"{self.entity} Projection Graph")
        save_plot(figure, f"Graph_{self.entity}_projection.png")


    def explore_graph_interactively(self, show_isolated_nodes: bool = False, label_top_n: int = 0,
                                    node_color: str = "deeppink") -> None:
        if IntSlider is None or fixed is None or interact is None:
            raise ImportError("ipywidgets is required for interactive graph exploration.")

        weights = nx.get_edge_attributes(self.graph, "weight").values()
        max_weight = int(max(weights)) if weights else 1

        interact(self.plot_filtered_graph_inline,
                 min_weight=IntSlider(value=1, min=1, max=max_weight, step=1, description="Min weight"),
                 show_isolated_nodes=fixed(show_isolated_nodes),
                 label_top_n=fixed(label_top_n),
                 node_color=fixed(node_color))


    def plot_filtered_graph_inline(self, min_weight: int, show_isolated_nodes: bool = False,
                                   label_top_n: int = 0, node_color: str = "deeppink") -> None:
        networkx_graph = get_display_graph(self.graph, min_weight, None, show_isolated_nodes)
        node_count = networkx_graph.number_of_nodes()

        figure_size = (10, 10) if node_count <= 50 else (14, 14)
        figure, axis = plt.subplots(figsize=figure_size)
        positions = get_graph_layout(self.graph)
        title = f"{self.entity} Projection Graph - Minimum Edge Weight: {min_weight}"
        draw_projection_graph(axis, networkx_graph, positions, node_color, label_top_n, title)
        plt.show()


    def compute_centrality_measures(self) -> DataFrame:
        centrality_table = pd.DataFrame({
            "degree_centrality": nx.degree_centrality(self.graph),
            "closeness_centrality": nx.closeness_centrality(self.graph),
            "betweenness_centrality": nx.betweenness_centrality(self.graph)
        })
        centrality_table.index.name = "node"
        return centrality_table

    def identify_communities(self) -> Dict[str, int]:
        louvain_communities = self.detect_communities_with_louvain()
        girvan_newman_communities = self.detect_communities_with_girvan_newman()

        self.show_communities(louvain_communities, "Louvain", color="Set1")
        self.show_communities(girvan_newman_communities, "Girvan-Newman", "Set1")
        return {
            "Louvain": len(louvain_communities),
            "Girvan-Newman": len(girvan_newman_communities)
        }


    def detect_communities_with_girvan_newman(self) -> List[Set]:
        best_communities = [set(community) for community in nx.connected_components(self.graph)]
        best_modularity = nx.community.modularity(self.graph, best_communities, weight="weight")
        max_splits = min(10, self.graph.number_of_nodes() - 1)

        for index, communities in enumerate(nx.community.girvan_newman(self.graph)):
            communities = [set(community) for community in communities]
            modularity = nx.community.modularity(self.graph, communities, weight="weight")
            if modularity > best_modularity:
                best_communities = communities
                best_modularity = modularity
            if index + 1 >= max_splits:
                break
        return best_communities

    def detect_communities_with_louvain(self) -> List[Set]:
        try:
            return [set(community)
                    for community in nx.community.louvain_communities(self.graph, weight="weight", seed=42)]
        except AttributeError:
            return [set(community) for community in nx.community.greedy_modularity_communities(
                self.graph, weight="weight")]


    def show_communities(self, communities: List[Set], algorithm: str, color: str,
                         edge_mode: str = "intra_community") -> None:
        print_section(f"COMMUNITIES: {self.entity} - {algorithm}")
        print_community_statistics(self.graph, communities)
        for community_index, community in enumerate(communities):
            print(format_community_members(community_index, community))
        figure_size = (14, 14) if self.graph.number_of_nodes() <= 50 else (22, 22)
        figure, axis = plt.subplots(figsize=figure_size)

        positions = get_graph_layout(self.graph)
        node_communities = get_node_communities(communities)
        edges = get_community_plot_edges(self.graph, node_communities, edge_mode)
        plot_weighted_edges(axis, positions, edges)

        color_map = plt.get_cmap(color)
        node_colors = {}
        for community_index, community in enumerate(communities):
            color = color_map(community_index % color_map.N)
            for node in community:
                node_colors[node] = color

        for node in self.graph.nodes:
            x, y = positions[node]
            axis.scatter(x, y, s=70, color=node_colors[node],
                         edgecolor="white", linewidth=0.6, zorder=2)

        axis.set_title(f"{self.entity} Communities: {algorithm.replace('_', ' ').title()}")
        axis.set_aspect("equal")
        axis.margins(0.08)
        axis.axis("off")
        save_plot(figure, f"Communities_{self.entity}_{algorithm}.png")


    def identify_largest_clique(self) -> Set:
        largest_clique = get_largest_clique(self.graph)
        print_section(f"LARGEST CLIQUE: {self.entity}")
        print(f"Clique size: {len(largest_clique)}")
        print_limited_nodes(largest_clique)
        self.plot_largest_clique(largest_clique)
        return largest_clique


    def plot_largest_clique(self, largest_clique: Set, clique_color: str = "deeppink",
                            node_color: str = "lightgray") -> None:
        figure_size = (14, 14) if self.graph.number_of_nodes() <= 50 else (22, 22)
        figure, axis = plt.subplots(figsize=figure_size)

        positions = get_graph_layout(self.graph)
        clique_edges = set()
        for node0, node1 in self.graph.edges:
            if node0 in largest_clique and node1 in largest_clique:
                clique_edges.add((node0, node1))
                clique_edges.add((node1, node0))

        for node0, node1, _ in self.graph.edges(data=True):
            is_clique_edge = (node0, node1) in clique_edges
            axis.plot([positions[node0][0], positions[node1][0]],
                      [positions[node0][1], positions[node1][1]],
                      color="mistyrose" if is_clique_edge else "ghostwhite",
                      zorder=1, linewidth=1.0 if is_clique_edge else 0.4,
                      alpha=0.65 if is_clique_edge else 0.2)

        for node in self.graph.nodes:
            x, y = positions[node]
            is_clique_node = node in largest_clique
            axis.scatter(x, y, s=90 if is_clique_node else 55,
                         color=clique_color if is_clique_node else node_color,
                         edgecolor="white", linewidth=0.6, zorder=2)

        axis.set_title(f"{self.entity} Largest Clique")
        axis.set_aspect("equal")
        axis.margins(0.08)
        axis.axis("off")
        save_plot(figure, f"Largest_Clique_{self.entity}.png")


    def identify_heavy_hitters(self, top_n: int = 10) -> None:
        print_section(f"HEAVY HITTERS: {self.entity}")
        print_node_heavy_hitters(self.graph, top_n)
        print_edge_heavy_hitters(self.graph, top_n)


class Data:
    dataframe: pd.DataFrame
    computers: Representation
    servers: Representation

    def __init__(self):
        self.dataframe: pd.DataFrame = read_data(get_data_path(), "ComputerNumber")
        self.computers = Representation("Computers", self.dataframe.index.tolist(),
                                        [column[column == 1].index.tolist() for _, column in self.dataframe.items()])
        self.servers = (Representation("Servers", self.dataframe.columns.tolist(),
                                       [row[row == 1].index.tolist() for _, row in self.dataframe.iterrows()]))
        self.community_counts: Dict[str, Dict[str, int]] = {}
        self.largest_clique_sizes: Dict[str, int] = {}

    def explore(self) -> None:
        if PRINT_MODE:
            self.print_basic_statistics()
            self.print_graph_summaries()
        if PLOT_MODE:
            self.plot_connection_heatmap()
            self.plot_connection_counts()

        for representation in [self.computers, self.servers]:
            if PLOT_MODE:
                representation.plot_adjacency_matrix_heatmap(color = "RdPu")
                if is_notebook_environment() and has_notebook_widgets():
                    representation.explore_graph_interactively()
                else:
                    representation.plot_graph()
            if PRINT_MODE:
                print_top_centrality_tables(representation.compute_centrality_measures(), representation.entity)


    def print_basic_statistics(self) -> None:
        print_section("BASIC STATISTICS")
        rows, columns = self.dataframe.shape
        memory_usage = self.dataframe.memory_usage(deep=True).sum()
        print(f"{'Rows':<20}{rows:>10}")
        print(f"{'Columns':<20}{columns:>10}")
        print(f"{'Memory usage':<20}{format_memory_usage(memory_usage):>10}")


    def plot_connection_heatmap(self) -> None:
        rows, columns = self.dataframe.shape
        plot_heatmap(self.dataframe.to_numpy(), self.dataframe.columns.tolist(), self.dataframe.index.tolist(),
                     (ceil(columns * 0.7), ceil(rows * 0.22)),
                     "Heatmap_connections.png", x_font_size=12, y_font_size=8, grid_color="deeppink", x_on_top=True)


    def plot_connection_counts(self) -> None:
        plot_server_connection_counts(self.dataframe.sum(axis=0))
        plot_computer_connection_counts(self.dataframe.sum(axis=1))


    def print_graph_summaries(self) -> None:
        print_section("GRAPH SUMMARY")
        print_graph_summary(self.computers.compute_graph_summary())
        print()
        print_graph_summary(self.servers.compute_graph_summary())


    def indentify_communities(self) -> None:
        self.community_counts["Servers"] = self.servers.identify_communities()
        self.community_counts["Computers"] = self.computers.identify_communities()


    def identify_largest_cliques(self) -> None:
        self.largest_clique_sizes["Servers"] = len(self.servers.identify_largest_clique())
        self.largest_clique_sizes["Computers"] = len(self.computers.identify_largest_clique())


    def identify_heavy_hitters(self) -> None:
        self.servers.identify_heavy_hitters()
        self.computers.identify_heavy_hitters()

    def print_final_summary(self) -> None:
        print_section("FINAL SUMMARY")
        print(f"Computers: {self.computers.graph.number_of_nodes()}")
        print(f"Servers: {self.servers.graph.number_of_nodes()}")
        print(f"Largest computer clique: {self.largest_clique_sizes.get('Computers', 'Not computed')}")
        print(f"Largest server clique: {self.largest_clique_sizes.get('Servers', 'Not computed')}")
        print("Analysis completed successfully")


########## HELPER FUNCTIONS ##########
##### CONSOLE OUTPUT #####
def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_memory_usage(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count} B"
    if bytes_count < 1024 ** 2:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count / (1024 ** 2):.1f} MB"


def is_notebook_environment() -> bool:
    if get_ipython is None:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return "IPKernelApp" in shell.config


def has_notebook_widgets() -> bool:
    return IntSlider is not None and fixed is not None and interact is not None


def print_limited_nodes(nodes: Set, limit: int = 10) -> None:
    sorted_nodes = sort_nodes(nodes)
    visible_nodes = sorted_nodes[:limit]
    omitted_count = len(sorted_nodes) - len(visible_nodes)
    suffix = ", ..." if omitted_count > 0 else ""

    print(f"First members: {', '.join(str(node) for node in visible_nodes)}{suffix}")
    if omitted_count > 0:
        print(f"({omitted_count} additional nodes omitted)")


def format_community_members(community_index: int, community: Set, limit: int = 10) -> str:
    sorted_nodes = sort_nodes(community)
    suffix = ", ..." if len(sorted_nodes) > limit else ""
    members = ", ".join(str(node) for node in sorted_nodes[:limit])
    return f"Community {community_index + 1} (size={len(community)}): {members}{suffix}"


def sort_nodes(nodes: Set) -> List:
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=str)


def print_graph_summary(summary: Dict[str, object]) -> None:
    print(summary["graph"])
    print(f"  {'Nodes':<24}{summary['nodes']:>10}")
    print(f"  {'Edges':<24}{summary['edges']:>10}")
    print(f"  {'Density':<24}{summary['density']:>10.3f}")
    print(f"  {'Connected components':<24}{summary['connected_components']:>10}")
    print(f"  {'Largest component':<24}{summary['largest_component_size']:>10}")
    print(f"  {'Connected':<24}{str(summary['is_connected']):>10}")


##### READ-WRITE OPERATIONS #####
def get_data_path() -> Path:
    return Path(os.path.join("data" if os.path.exists("data") else ".", "Autonomous_Systems.csv"))


def read_data(path: Path, index: str) -> DataFrame:
    dataset: DataFrame = pd.read_csv(path)
    dataset.set_index(index, inplace=True)
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.loc[dataset.sum(axis=1) > 0, dataset.sum(axis=0) > 0]
    dataset = dataset.sort_index(axis=1)
    dataset = dataset.loc[dataset.sum(axis=1) > 0, dataset.sum(axis=0) > 0]
    return dataset


def save_data(data: DataFrame, path: Path) -> None:
    data.to_csv(path, index=True)


def save_plot(figure: plt.Figure, file_name: str) -> None:
    os.makedirs("res", exist_ok=True)
    figure.tight_layout()
    figure.savefig(f"res/{file_name}", bbox_inches="tight")
    if is_notebook_environment():
        plt.figure(figure.number)
        plt.show()
        return
    plt.close(figure)


##### PLOTS #####
def get_gradient_colors(values: pd.Series, color: str = "magma_r") -> List:
    color_map = plt.get_cmap(color)
    normalizer = Normalize(vmin=values.min(), vmax=values.max())
    return [color_map(normalizer(value)) for value in values]


def format_heatmap_axis(axis: plt.Axes, x_labels: List, y_labels: List, x_font_size: int = 6, y_font_size: int = 6,
                        grid_color: str = "lightgray", x_on_top: bool = False) -> None:
    rows, columns = len(y_labels), len(x_labels)

    axis.set_xticks(range(columns))
    axis.set_yticks(range(rows))
    axis.set_xticklabels(x_labels, fontsize=x_font_size)
    axis.set_yticklabels(y_labels, fontsize=y_font_size)
    axis.tick_params(axis="x", which="major", rotation=90, length=0)
    axis.tick_params(axis="y", which="major", length=0)
    axis.set_xticks([index - 0.5 for index in range(columns + 1)], minor=True)
    axis.set_yticks([index - 0.5 for index in range(rows + 1)], minor=True)
    axis.grid(which="minor", color=grid_color, linestyle="-", linewidth=0.4)
    axis.tick_params(which="minor", bottom=False, left=False, top=False)

    if x_on_top:
        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")


def plot_heatmap(matrix: np.ndarray, x_labels: List, y_labels: List, figure_size: Tuple[int, int],
                 file_name: str, title: str = "", x_font_size: int = 6, y_font_size: int = 6,
                 grid_color: str = "lightgray", x_on_top: bool = False,
                 color_map: str | ListedColormap = ListedColormap(["mistyrose", "crimson"])) -> None:
    figure, axis = plt.subplots(figsize=figure_size, dpi=555)
    axis.imshow(matrix, cmap=color_map, aspect="equal")
    axis.set_title(title)
    format_heatmap_axis(axis, x_labels, y_labels, x_font_size, y_font_size, grid_color, x_on_top)
    save_plot(figure, file_name)


def plot_server_connection_counts(counts: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(18, 7), dpi=222)
    bars = axis.bar(counts.index, counts.values, color=get_gradient_colors(counts, "Wistia"))
    axis.bar_label(bars, padding=2, fontsize=8)
    axis.set_title("Servers: Number of incoming connections from computers")
    # axis.set_xlabel("Server")
    # axis.set_ylabel("Number of Computers")
    axis.tick_params(axis="x", rotation=0)
    axis.margins(x=0.01, y=0.1)
    save_plot(figure, "Barchart_Servers.png")


def plot_computer_connection_counts(counts: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(9, 24), dpi=555)
    axis.barh(counts.index.astype(str), counts.values, color=get_gradient_colors(counts, "cool_r"))
    axis.set_title("Computers: Number of outgoing connections to servers")
    axis.set_xlabel("Number of Servers")
    axis.set_ylabel("Computer ID")
    axis.tick_params(axis="y", labelsize=6)
    axis.invert_yaxis()
    axis.margins(x=0.0, y=0.002)
    save_plot(figure, "Barchart_Computers.png")


def draw_projection_graph(axis: plt.Axes, graph: nx.Graph, positions: Dict[Hashable, np.ndarray],
                          node_color: str, label_top_n: int, title: str) -> None:
    plot_weighted_edges(axis, positions, list(graph.edges(data=True)))
    plot_nodes(axis, positions, graph.nodes, node_color)
    plot_node_labels(axis, positions, graph, label_top_n)

    axis.set_title(title)
    axis.set_aspect("equal")
    axis.margins(0.08)
    axis.axis("off")


def plot_nodes(axis: plt.Axes, positions: Dict[Hashable, np.ndarray], nodes: Iterable[Hashable],
               node_color: str) -> None:
    for node in nodes:
        x, y = positions[node]
        axis.scatter(x, y, s=70, color=node_color,
                     edgecolor="white", linewidth=0.6, zorder=2)


def plot_node_labels(axis: plt.Axes, positions: Dict[Hashable, np.ndarray], graph: nx.Graph,
                     label_top_n: int) -> None:
    degrees = get_weighted_degrees(graph)
    label_nodes = get_label_nodes(degrees, graph.number_of_nodes(), label_top_n)
    for node in label_nodes:
        x, y = positions[node]
        axis.text(x * 1.08, y * 1.08, str(node), ha="center", va="center", fontsize=8)


def plot_weighted_edges(axis: plt.Axes, positions: Dict[Hashable, np.ndarray],
                        edges: List[Tuple[Hashable, Hashable, Dict[str, float]]]) -> None:
    weights = get_weights(edges)
    min_edge_weight, max_edge_weight = min(weights) if weights else 0, max(weights) if weights else 0
    edge_color_map = LinearSegmentedColormap.from_list("edge_weight", ["lightgray", "black"])
    edge_color_normalizer = Normalize(vmin=min_edge_weight, vmax=max_edge_weight)

    for node0, node1, attributes in edges:
        weight = attributes["weight"]
        axis.plot([positions[node0][0], positions[node1][0]],
                  [positions[node0][1], positions[node1][1]], color=edge_color_map(edge_color_normalizer(weight)),
                  zorder=1, linewidth=0.6, alpha=0.7)



##### PRE-PROCESSING DATA #####
def build_graph_and_adjacency_matrix(labels: List[Hashable], groups: List[List[Hashable]]) -> Tuple[nx.Graph, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    label_index = {label: index for index, label in enumerate(labels)}

    size = len(labels)
    matrix: np.ndarray = np.zeros((size, size), dtype=int)
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                node0 = group[i]
                node1 = group[j]
                add_weighted_edge(graph, node0, node1)
                matrix[label_index[node0], label_index[node1]] += 1
                matrix[label_index[node1], label_index[node0]] += 1
    return graph, matrix


def add_weighted_edge(graph: nx.Graph, node0: Hashable, node1: Hashable, weight: float = 1.0) -> None:
    if graph.has_edge(node0, node1):
        graph[node0][node1]["weight"] += weight
        return

    graph.add_edge(node0, node1, weight=weight)


def get_weighted_degrees(graph: nx.Graph) -> Dict[Hashable, float]:
    return dict(graph.degree(weight="weight"))


def scale_value(value: float, source_min: float, source_max: float, target_min: float,
                target_max: float) -> float:
    if source_min == source_max:
        return (target_min + target_max) / 2
    return target_min + ((value - source_min) / (source_max - source_min)) * (target_max - target_min)


def get_filtered_graph(graph: nx.Graph, min_weight: float | None = None) -> nx.Graph:
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(graph.nodes)

    for node0, node1, attributes in graph.edges(data=True):
        if min_weight is None or attributes["weight"] >= min_weight:
            filtered_graph.add_edge(node0, node1, weight=attributes["weight"])
    return filtered_graph


def get_display_graph(graph: nx.Graph, min_weight: float | None = None, max_edges: int | None = None,
                      show_isolated_nodes: bool = False) -> nx.Graph:
    display_graph = get_filtered_graph(graph, min_weight)
    display_graph = get_strongest_edges_graph(display_graph, max_edges)
    if not show_isolated_nodes:
        display_graph.remove_nodes_from(list(nx.isolates(display_graph)))
    return display_graph


def get_strongest_edges_graph(graph: nx.Graph, max_edges: int | None) -> nx.Graph:
    if max_edges is None or graph.number_of_edges() <= max_edges:
        return graph

    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(graph.nodes)
    strongest_edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]["weight"], reverse=True)[:max_edges]

    for node0, node1, attributes in strongest_edges:
        filtered_graph.add_edge(node0, node1, weight=attributes["weight"])
    return filtered_graph


def get_largest_clique(graph: nx.Graph) -> Set:
    if graph.number_of_nodes() == 0:
        return set()
    return set(max(nx.find_cliques(graph), key=len))


def get_node_communities(communities: List[Set]) -> Dict[Hashable, int]:
    node_communities = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_communities[node] = community_index
    return node_communities


def get_community_plot_edges(graph: nx.Graph, node_communities: Dict[Hashable, int],
                             edge_mode: str) -> List[Tuple[Hashable, Hashable, Dict[str, float]]]:
    if edge_mode == "all":
        return list(graph.edges(data=True))

    if edge_mode == "intra_community":
        return [(node0, node1, attributes) for node0, node1, attributes in graph.edges(data=True)
                if node_communities[node0] == node_communities[node1]]

    if edge_mode == "inter_community":
        return [(node0, node1, attributes) for node0, node1, attributes in graph.edges(data=True)
                if node_communities[node0] != node_communities[node1]]

    raise ValueError(f"Unknown community edge mode: {edge_mode}")


def print_community_statistics(graph: nx.Graph, communities: List[Set]) -> None:
    modularity = nx.community.modularity(graph, communities, weight="weight") if communities else 0

    print(f"{'Number of communities':<24}{len(communities):>10}")
    print(f"{'Modularity':<24}{modularity:>10.3f}")


def get_graph_layout(graph: nx.Graph) -> Dict[Hashable, np.ndarray]:
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return {}
    if graph.number_of_edges() == 0:
        return nx.circular_layout(graph, scale=max(3.0, np.sqrt(node_count)))

    distance = 2.5 / np.sqrt(node_count)
    scale = max(4.0, np.sqrt(node_count))
    return nx.spring_layout(graph, weight="weight", seed=42, iterations=1000, k=distance, scale=scale)


def get_edge_width_range(edge_count: int, density: float) -> Tuple[float, float]:
    if edge_count <= 200 and density <= 0.65:
        return 0.4, 2.4
    if edge_count <= 800:
        return 0.15, 1.2
    return 0.05, 0.55


def get_edge_alpha_range(edge_count: int, density: float) -> Tuple[float, float]:
    if edge_count <= 200 and density <= 0.65:
        return 0.18, 0.55
    if edge_count <= 800:
        return 0.06, 0.28
    return 0.02, 0.12


def get_node_size_range(node_count: int) -> Tuple[float, float]:
    if node_count <= 50:
        return 160, 950
    if node_count <= 150:
        return 35, 260
    return 12, 130


def get_label_nodes(degrees: Dict[Hashable, float], node_count: int, label_top_n: int | None) -> List[Hashable]:
    if label_top_n is None:
        label_top_n = node_count if node_count <= 50 else 0
    if label_top_n <= 0:
        return []
    return sorted(degrees, key=degrees.get, reverse=True)[:label_top_n]


def get_community_colors(graph: nx.Graph) -> (
        Dict[Any, ndarray | Tuple[float, float, float, float]] | Dict[Hashable, Tuple]):
    if graph.number_of_edges() == 0:
        return {node_id: plt.get_cmap("tab20")(0) for node_id in graph.nodes}

    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    color_map = plt.get_cmap("tab20")
    colors: Dict[Hashable, tuple] = {}

    for community_index, community in enumerate(communities):
        color = color_map(community_index % color_map.N)
        for node_id in community:
            colors[node_id] = color
    return colors


def get_weights(edges: List[Tuple[Hashable, Hashable, Dict[str, float]]]) -> List[float]:
    return [attributes["weight"] for _, _, attributes in edges]


def print_node_heavy_hitters(graph: nx.Graph, top_n: int) -> None:
    top_n = min(top_n, 10)
    weighted_degrees = sorted(graph.degree(weight="weight"), key=lambda item: item[1], reverse=True)[:top_n]

    print(f"\nTop {top_n} nodes by weighted degree")
    print(f"{'Rank':>4} | {'Node':<18} | {'Weighted Degree':>15}")
    for rank, (node, weighted_degree) in enumerate(weighted_degrees, start=1):
        print(f"{rank:>4} | {str(node):<18} | {weighted_degree:>15.0f}")


def print_edge_heavy_hitters(graph: nx.Graph, top_n: int) -> None:
    top_n = min(top_n, 10)
    edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]["weight"], reverse=True)[:top_n]

    print(f"\nTop {top_n} edges by weight")
    print(f"{'Rank':>4} | {'Edge':<36} | {'Weight':>6}")
    for rank, (node0, node1, attributes) in enumerate(edges, start=1):
        edge = f"{node0} ↔ {node1}"
        print(f"{rank:>4} | {edge:<36} | {attributes['weight']:>6.0f}")


def print_top_centrality_tables(centrality_table: DataFrame, entity: str, top_n: int = 10) -> None:
    print_section(f"TOP CENTRALITY: {entity}")
    top_n = min(top_n, 10)
    top_nodes = centrality_table.sort_values("degree_centrality", ascending=False).head(top_n).round(3)

    print(f"Top {top_n} {entity} nodes by degree centrality")
    print(f"{'Rank':>4} | {'Node':<18} | {'Degree':>8} | {'Closeness':>9} | {'Betweenness':>11}")
    for rank, (node, row) in enumerate(top_nodes.iterrows(), start=1):
        print(f"{rank:>4} | {str(node):<18} | {row['degree_centrality']:>8.3f} | "
              f"{row['closeness_centrality']:>9.3f} | {row['betweenness_centrality']:>11.3f}")


if __name__ == '__main__':
    data_class: Data = Data()
    data_class.explore()
    data_class.indentify_communities()
    data_class.identify_largest_cliques()
    data_class.identify_heavy_hitters()
    data_class.print_final_summary()
