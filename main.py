import os
from dataclasses import dataclass
from math import ceil
from typing import Dict, Tuple, List, Hashable, Any
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from numpy import ndarray
from pandas import DataFrame
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt

########## CLASSES ##########

@dataclass(frozen=True)
class Node:
    node_id: Hashable

@dataclass(frozen=True)
class Edge:
    node0: Hashable
    node1: Hashable
    weight: float # TODO: now int, change to float


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[Hashable, Node] = {}
        self.edges: Dict[Tuple[Hashable, Hashable], Edge] = {}

    def add_node(self, node_id: Hashable) -> Node:
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists.")
        node = Node(node_id)
        self.nodes[node_id] = node
        return node

    def get_node(self, node_id: Hashable) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found.")
        return self.nodes[node_id]

    @staticmethod
    def get_edge_key(node0: Hashable, node1: Hashable) -> Tuple[Hashable, Hashable]:
        return (node0, node1) if node0 <= node1 else (node1, node0)

    def get_edge(self, node0: Hashable, node1: Hashable) -> Edge:
        key = self.get_edge_key(node0, node1)
        if key not in self.edges:
            raise KeyError(f"Edge between {node0} and {node1} not found.")
        return self.edges[key]

    def get_or_add_node(self, node_id: Hashable) -> Node:
        if node_id not in self.nodes:
            return self.add_node(node_id)
        return self.nodes[node_id]

    def add_edge(self, node0: Hashable, node1: Hashable, weight: float = 1.0) -> Edge:
        self.get_or_add_node(node0)
        self.get_or_add_node(node1)

        key = self.get_edge_key(node0, node1)
        if key in self.edges:
            edge = self.edges[key]
            updated_edge = Edge(edge.node0, edge.node1, edge.weight + weight)
            self.edges[key] = updated_edge
            return updated_edge

        edge = Edge(key[0], key[1], weight)
        self.edges[key] = edge
        return edge

    def print_nodes(self) -> None:
        for node_id, node in self.nodes.items():
            print(node)

    def print_edges(self) -> None:
        print("Edges:")
        for edge_key, edge in self.edges.items():
            print(f"({edge_key[0]} - {edge_key[1]})")

    def print_graph(self) -> None:
        self.print_nodes()
        self.print_edges()


class Representation:
    entity: str
    matrix: np.ndarray
    graph: Graph
    labels: List[Hashable]

    def __init__(self, entity: str, labels: List[Hashable], groups: List[List[Hashable]]) -> None:
        self.entity = entity
        self.labels = labels
        self.graph = Graph()
        label_index = {label: index for index, label in enumerate(self.labels)}

        for label in self.labels:
            self.graph.add_node(label)

        size = len(self.labels)
        self.matrix = np.zeros((size, size))
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    node0 = group[i]
                    node1 = group[j]
                    self.graph.add_edge(node0, node1, weight=1.0)  # TODO: modify weight
                    self.matrix[label_index[node0], label_index[node1]] = 1
                    self.matrix[label_index[node1], label_index[node0]] = 1


    def compute_graph_summary(self) -> Dict[str, str | int]:
        graph = to_networkx_graph(self.graph)
        component_sizes = [len(component) for component in nx.connected_components(graph)]

        return {
            "graph": self.entity,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "connected_components": nx.number_connected_components(graph),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
        }

    def plot_adjacency_matrix_heatmap(self) -> None:
        size = ceil(self.matrix.shape[0] * 0.22)
        plot_heatmap(self.matrix, self.labels, self.labels, (size, size), title=f"{self.entity} Projection Adjacency Matrix",
                     file_name=f"Heatmap_{self.entity}_adjacency_matrix.png")


    def plot_graph(self, min_weight: float = 1.0) -> None:
        figure, axis = plt.subplots(figsize=(14, 14) if len(self.labels) <= 50 else (22, 22))
        edge_width_range = (0.2, 2.0) if len(self.labels) <= 50 else (0.05, 0.5)
        edge_alpha_range = (0.05, 0.35) if len(self.labels) <= 50 else (0.015, 0.08)
        degrees = get_weighted_degrees(self.graph, min_weight)
        networkx_graph = to_networkx_graph(self.graph, min_weight)
        positions = get_graph_layout(networkx_graph)
        edges = [edge for edge in self.graph.edges.values() if edge.weight >= min_weight]

        weights = get_weights(edges)
        min_weight, max_weight = min(weights) if weights else 0, max(weights) if weights else 0
        for edge in edges:
            axis.plot([positions[edge.node0][0], positions[edge.node1][0]],
                      [positions[edge.node0][1], positions[edge.node1][1]], color="gray", zorder=1,
                      linewidth=scale_value(edge.weight, min_weight, max_weight, edge_width_range[0],
                                            edge_width_range[1]),
                      alpha=scale_value(edge.weight, min_weight, max_weight, edge_alpha_range[0], edge_alpha_range[1]))

        sizes = (120, 900) if len(self.labels) <= 50 else (12, 140)
        for node in self.labels:
            x, y = positions[node]
            axis.scatter(x, y,
                         s=scale_value(degrees[node], min(degrees.values()), max(degrees.values()), sizes[0], sizes[1]),
                         color=get_community_colors(networkx_graph)[node],
                         edgecolor="cadetblue", linewidth=0.6, zorder=2)

        if len(self.labels) <= 50:
            for node in self.labels:
                x, y = positions[node]
                axis.text(x * 1.08, y * 1.08, str(node), ha="center", va="center", fontsize=8)

        axis.set_title(f"{self.entity} Projection Graph")
        axis.set_aspect("equal")
        axis.axis("off")
        save_plot(figure, f"Graph_{self.entity}_projection.png")


    def compute_centrality_measures(self) -> DataFrame:
        graph = to_networkx_graph(self.graph)
        centrality_table = pd.DataFrame({
            "degree_centrality": nx.degree_centrality(graph),
            "closeness_centrality": nx.closeness_centrality(graph),
            "betweenness_centrality": nx.betweenness_centrality(graph)
        })
        centrality_table.index.name = "node"
        return centrality_table


@dataclass
class Data:
    dataframe: pd.DataFrame
    computers: Representation
    servers: Representation

    def __init__(self):
        self.dataframe: pd.DataFrame = read_data(get_data_path(), "ComputerNumber")
        self.computers = Representation("Computers", self.dataframe.index.tolist(),
                                        [column[column==1].index.tolist() for _, column in self.dataframe.items()])
        self.servers = (Representation("Servers", self.dataframe.columns.tolist(),
                                       [row[row==1].index.tolist() for _, row in self.dataframe.iterrows()]))

    def explore(self) -> None:
        self.print_basic_statistics()
        self.plot_connection_heatmap()
        self.plot_connection_counts()
        self.print_graph_summaries()

        for representation in [self.computers, self.servers]:
            representation.plot_adjacency_matrix_heatmap()
            representation.plot_graph()
            print_top_centrality_tables(representation.compute_centrality_measures(), representation.entity)


    def print_basic_statistics(self) -> None:
        print(self.dataframe.head())
        print(self.dataframe.describe())
        self.dataframe.info()


    def plot_connection_heatmap(self) -> None:
        rows, columns = self.dataframe.shape
        plot_heatmap(self.dataframe.to_numpy(), self.dataframe.columns.tolist(), self.dataframe.index.tolist(),
                     (ceil(columns * 0.7), ceil(rows * 0.22)),
                     "Heatmap_connections.png", x_font_size=12, y_font_size=8, grid_color="deeppink", x_on_top=True)


    def plot_connection_counts(self) -> None:
        plot_server_connection_counts(self.dataframe.sum(axis=0))
        plot_computer_connection_counts(self.dataframe.sum(axis=1))


    def print_graph_summaries(self) -> None:
        computers: Dict[str, object] = self.computers.compute_graph_summary()
        servers: Dict[str, object] = self.servers.compute_graph_summary()
        summary_table = pd.DataFrame([computers, servers]).set_index("graph")

        print("\nProjected graph summary")
        print(summary_table.to_string(float_format=lambda value: f"{value:.4f}"))


########## HELPER FUNCTIONS ##########
##### READ-WRITE OPERATIONS #####
def get_data_path() -> Path:
    return Path(os.path.join("data" if os.path.exists("data") else ".", "Autonomous_Systems.csv"))


def read_data(path: Path, index: str) -> DataFrame:
    dataset: DataFrame = pd.read_csv(path)
    dataset.set_index(index, inplace=True)
    dataset.columns = dataset.columns.str.strip()
    return dataset


def save_data(data: DataFrame, path: Path) -> None:
    data.to_csv(path, index=True)


def save_plot(figure: plt.Figure, file_name: str) -> None:
    os.makedirs("res", exist_ok=True)
    figure.tight_layout()
    figure.savefig(f"res/{file_name}", bbox_inches="tight")
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
                 grid_color: str = "lightgray", x_on_top: bool = False) -> None:
    figure, axis = plt.subplots(figsize=figure_size)
    axis.imshow(matrix, cmap=ListedColormap(["mistyrose", "crimson"]), aspect="equal")
    axis.set_title(title)
    format_heatmap_axis(axis, x_labels, y_labels, x_font_size, y_font_size, grid_color, x_on_top)
    save_plot(figure, file_name)


def plot_server_connection_counts(counts: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.bar(counts.index, counts.values, color=get_gradient_colors(counts))
    axis.set_title("Computers Connected to Each Server")
    axis.set_xlabel("Server")
    axis.set_ylabel("Number of Computers")
    axis.tick_params(axis="x", rotation=90)
    save_plot(figure, "server_barchart.png")


def plot_computer_connection_counts(counts: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(18, 7))
    axis.bar(counts.index.astype(str), counts.values, color=get_gradient_colors(counts))
    axis.set_title("Servers Connected to Each Computer")
    axis.set_xlabel("Computer ID")
    axis.set_ylabel("Number of Servers")
    axis.tick_params(axis="x", rotation=90, labelsize=6)
    save_plot(figure, "Computer_barchart.png")



##### PRE-PROCESSING DATA #####
def get_weighted_degrees(graph: Graph, min_weight: float = 1.0) -> Dict[Hashable, float]:
    degrees = {node_id: 0.0 for node_id in graph.nodes}
    for edge in graph.edges.values():
        if edge.weight >= min_weight:
            degrees[edge.node0] += edge.weight
            degrees[edge.node1] += edge.weight
    return degrees


def scale_value(value: float, source_min: float, source_max: float, target_min: float,
                target_max: float) -> float:
    if source_min == source_max:
        return (target_min + target_max) / 2
    return target_min + ((value - source_min) / (source_max - source_min)) * (target_max - target_min)


def to_networkx_graph(graph: Graph, min_weight: float = 1.0) -> nx.Graph:
    networkx_graph = nx.Graph()
    networkx_graph.add_nodes_from(graph.nodes.keys())
    for edge in graph.edges.values():
        if edge.weight < min_weight:
            continue
        networkx_graph.add_edge(edge.node0, edge.node1, weight=edge.weight)
    return networkx_graph


def get_graph_layout(graph: nx.Graph) -> Dict[Hashable, np.ndarray]:
    if len(graph.nodes) <= 50:
        return nx.spring_layout(graph, weight="weight", seed=42, iterations=700, k=0.8, scale=3.0)
    return nx.spring_layout(graph, weight="weight", seed=42, iterations=1000, k=1.6, scale=10.0)


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


def get_weights(edges: List[Edge]) -> List[float]:
    return [edge.weight for edge in edges]

def print_top_centrality_tables(centrality_table: DataFrame, entity: str, top_n: int = 10) -> None:
    for column_name in centrality_table.columns:
        top_nodes = centrality_table.sort_values(column_name, ascending=False).head(top_n)
        print(f"\nTop {top_n} {entity} nodes by {column_name.replace('_', ' ')}")
        print(top_nodes.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == '__main__':
    data_class: Data = Data()
    data_class.explore()
