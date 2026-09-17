import networkx as nx
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_map_graph(mapped_modules: Dict[str, Tuple[int, int]], edges: List[Tuple[str, str]]) -> nx.Graph:
    """Constructs a NetworkX graph from the extracted map data."""
    logger.info("Building NetworkX graph from map data...")
    G = nx.Graph()
    
    for module_name, coords in mapped_modules.items():
        G.add_node(module_name, pos=coords)
        
    for edge in edges:
        if edge[0] in mapped_modules and edge[1] in mapped_modules:
            G.add_edge(edge[0], edge[1])
            
    logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def find_nearest_node_to_player(G: nx.Graph, player_coords: Tuple[int, int]) -> str | None:
    """Finds the map node closest to the player's physical coordinates."""
    if not G.nodes:
        return None
        
    closest_node = None
    min_dist = float('inf')
    px, py = player_coords
    
    for node, data in G.nodes(data=True):
        if 'pos' in data:
            nx_coord, ny_coord = data['pos']
            dist = (px - nx_coord)**2 + (py - ny_coord)**2
            if dist < min_dist:
                min_dist = dist
                closest_node = node
                
    return closest_node

def calculate_optimal_route(G: nx.Graph, start_node: str, target_nodes: List[str]) -> List[str]:
    """Calculates a greedy nearest-neighbor route visiting target nodes."""
    logger.info(f"Calculating route from '{start_node}' to targets: {target_nodes}")
    
    valid_targets = [t for t in target_nodes if t in G]
    if not valid_targets:
        return []
        
    if start_node not in G:
        return []

    current_node = start_node
    remaining_targets = set(valid_targets)
    full_path = [current_node]
    
    while remaining_targets:
        closest_target = None
        shortest_path_len = float('inf')
        best_path_segment = []
        
        for target in remaining_targets:
            try:
                path = nx.shortest_path(G, source=current_node, target=target)
                path_len = len(path)
                if path_len < shortest_path_len:
                    shortest_path_len = path_len
                    closest_target = target
                    best_path_segment = path
            except nx.NetworkXNoPath:
                pass
                
        if closest_target is None:
            break
            
        if len(best_path_segment) > 1:
            full_path.extend(best_path_segment[1:])
            
        current_node = closest_target
        remaining_targets.remove(closest_target)
        
    return full_path