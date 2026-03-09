import json
import os
import sys
import torch
from torch_geometric.utils import coalesce, is_undirected, contains_self_loops
from tqdm import tqdm

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add current directory for train modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from utils.utils import setup_seed


def construct_sub_graph(cur_graph, full_edge_index, pred_edge, pad_val: int = -500):
    """
    Extract a subgraph from the full edge index, only keeping edges between non-padding nodes in cur_graph
    
    Args:
        cur_graph: array of nodes containing padding
        full_edge_index: edge index of the full graph [2, E]
        pad_val: value for padding
        
    Returns:
        edge_index: edge index of the subgraph, node IDs remapped to 0 to len(node_list)-1
        node_list: list of unique nodes after removing padding
    """
    # get unique nodes that are not padding
    node_list = cur_graph[cur_graph != pad_val].unique().tolist()
    node_list = sorted(node_list)
    
    if len(node_list) == 0:
        return torch.empty(2, 0, dtype=torch.long), node_list
    
    # create a set of nodes for quick lookup
    node_set = set(node_list)
    
    # filter edges from the full edge index to get the subgraph
    mask = torch.isin(full_edge_index[0], torch.tensor(node_list)) & \
           torch.isin(full_edge_index[1], torch.tensor(node_list))
    sub_edge_index = full_edge_index[:, mask]
    
    # if the subgraph contains the predicted edge, print and remove (also consider the reverse edge)
    if sub_edge_index.shape[1] > 0 and pred_edge is not None:
        u = int(pred_edge[0])
        v = int(pred_edge[1])
        e_before = sub_edge_index.shape[1]
        remove_mask = (sub_edge_index[0] == u) & (sub_edge_index[1] == v)
        remove_mask_rev = (sub_edge_index[0] == v) & (sub_edge_index[1] == u)
        to_keep = ~(remove_mask | remove_mask_rev)
        e_after = int(to_keep.sum().item())
        if e_after < e_before:
            print(f"[construct_sub_graph] remove pred_edge {(u, v)} (and reverse if exists) from subgraph: {e_before} -> {e_after}")
            sub_edge_index = sub_edge_index[:, to_keep]
    
    if sub_edge_index.shape[1] == 0:
        return torch.empty(2, 0, dtype=torch.long), node_list
    
    # create a mapping from original node IDs to new indices
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # remap the edge indices
    remapped_edge_index = torch.zeros_like(sub_edge_index)
    remapped_edge_index[0] = torch.tensor([node_to_idx[node.item()] for node in sub_edge_index[0]])
    remapped_edge_index[1] = torch.tensor([node_to_idx[node.item()] for node in sub_edge_index[1]])
    
    return remapped_edge_index, node_list 



setup_seed(0)
dataset_list = ['ogbn-arxiv']
mode = ['train', 'test']
use_hop = 2
sample_neighbor_size = 10
DEFAULT_GRAPH_PAD_ID = -500
for dataset in dataset_list:
    for m in mode:
        data_path = f'./{dataset}/edge_sampled_{use_hop}_{sample_neighbor_size}_only_{m}.jsonl'
        data = torch.load(f'./{dataset}/processed_data.pt')
        is_undirected_graph = is_undirected(data.edge_index)
        has_self_loops = contains_self_loops(data.edge_index)
        
        # use coalesce function to remove duplicate edges, then compare the number of edges
        edge_index_no_duplicates, _ = coalesce(data.edge_index, None, data.x.shape[0])
        has_duplicate_edges = data.edge_index.shape[1] != edge_index_no_duplicates.shape[1]

        print(f"is_undirected_graph: {is_undirected_graph}")
        print(f"has_self_loops: {has_self_loops}")
        print(f"has_duplicate_edges: {has_duplicate_edges}")
        print(f"Load from {data_path}\n")
        
        lines = open(data_path, "r").readlines()
        questions = [json.loads(q) for q in lines]
        for line in tqdm(questions):
            # keep the level-order array containing padding for correct parent-child relationship calculation
            graph_full = torch.tensor(line["graph"])  # level-order array containing -500
            pred_edge = torch.tensor(line["id"])
            edge_index_list, node_list_list = [], []
            for graph in graph_full:
                edge_index, node_list = construct_sub_graph(graph, data.edge_index, pred_edge, DEFAULT_GRAPH_PAD_ID)
                edge_index_list.append(edge_index.tolist())
                node_list_list.append(node_list)
            line['edge_index'] = edge_index_list
            line["node_list"] = node_list_list
            
        save_path = f'./{dataset}/edge_sampled_{use_hop}_{sample_neighbor_size}_only_{m}.jsonl'
        with open(save_path, "w") as f:
            for line in questions:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        print(f"Saved to {save_path}")
        print(f"Processed {len(questions)} data\n")