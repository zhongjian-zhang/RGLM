import json
import os
import sys
import torch
import json
import os.path as osp
import numpy as np
from scipy.sparse import csr_array
# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add current directory for train modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from utils.utils import setup_seed
from utils.data_process import generate_edge_list, get_fix_shape_subgraph_sequence_fast


def dump_jsonl(data, json_filepath):
    path_name = osp.dirname(json_filepath)
    os.makedirs(path_name, exist_ok=True)
    with open(json_filepath, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')


def generate_negative_samples(edge_index, num_nodes, pos_index):
    adj = csr_array((torch.ones(len(edge_index[0])), (edge_index[0], edge_index[1]),), shape=(num_nodes, num_nodes), )
    dense_adj = adj.todense() == 0
    neg_row, neg_col = np.nonzero(dense_adj)
    neg_edge_idx = np.random.permutation(len(neg_row))[:pos_index]
    neg_row, neg_col = neg_row[neg_edge_idx], neg_col[neg_edge_idx]
    neg_edges = np.stack([neg_row, neg_col], axis=1)
    return torch.tensor(neg_edges).t()



def generate_edge_level_prompt_files(dataset_name, data):
    """
        Here, the data should be the test data object (assuming using randomlinksplit)
    """
    train_idx = data.train_id
    val_idx = data.val_id
    test_idx = data.test_id
    edge_index = data.edge_index
    orig_edge_index = data.edge_index
    ## remove all test_idx to prevent leakage
    edge_index = edge_index[:, train_idx]
    data.edge_index = edge_index
    edge_list = generate_edge_list(data)
    edge_index = orig_edge_index

    train_len, val_len, test_len = len(train_idx), len(val_idx), len(test_idx)
    pos_index = train_len + val_len + test_len
    num_nodes = data.x.shape[0]
    neg_edge_idx = generate_negative_samples(edge_index, num_nodes, pos_index)

    for split_name, idx in zip(['train', 'test'], [train_idx, test_idx]):
        name = osp.join(f'./dataset/{dataset_name}', f"edge_sampled_2_10_only_{split_name}.jsonl")
        ## positive edges
        pos_edges = edge_index[:, idx]
        ## negative edges
        neg_edges = neg_edge_idx[:, idx]

        jsonl_list = []
        for i in range(idx.shape[0]):
            tree = {}
            left, right = pos_edges[:, i]
            tree['id'] = [left.item(), right.item()]
            left_g = get_fix_shape_subgraph_sequence_fast(edge_list, left.item(), 2, 10, avoid_idx=right.item())
            right_g = get_fix_shape_subgraph_sequence_fast(edge_list, right.item(), 2, 10, avoid_idx=left.item())
            tree['graph'] = [left_g, right_g]
            tree['conversations'] = [
                {"from": "human", "value": f"{dataset_name}_pos_edge"},
                {"from": "gpt", "value": "yes"}
            ]
            jsonl_list.append(tree)

            tree = {}
            left, right = neg_edges[:, i]
            tree['id'] = [left.item(), right.item()]
            left_g = get_fix_shape_subgraph_sequence_fast(edge_list, left.item(), 2, 10, avoid_idx=right.item())
            right_g = get_fix_shape_subgraph_sequence_fast(edge_list, right.item(), 2, 10, avoid_idx=left.item())
            tree['graph'] = [left_g, right_g]
            tree['conversations'] = [
                {"from": "human", "value": f"{dataset_name}_neg_edge"},
                {"from": "gpt", "value": "no"}
            ]
            jsonl_list.append(tree)
        dump_jsonl(jsonl_list, name)

if __name__ == "__main__":
    setup_seed(0)
    dataset_name = "reddit"
    data = torch.load(f"./dataset/{dataset_name}/processed_data.pt")
    generate_edge_level_prompt_files(dataset_name, data)