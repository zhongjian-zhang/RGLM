import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import negative_sampling


class GraphDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, depth=1):
        super().__init__()

        modules = [nn.Linear(input_dim, output_dim)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_dim, output_dim))
        self.feat_recon_decoder = nn.Sequential(*modules)
        
        if depth == 1:
            modules = [nn.Linear(2 * input_dim, 1)]
        else:
            modules = [nn.Linear(2 * input_dim, output_dim)]
            for _ in range(1, depth-1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_dim, 1))
        self.topo_recon_decoder = nn.Sequential(*modules)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming for hidden Linear (with GELU), small-normal for final layers."""
        # initialize the feature reconstruction decoder: hidden layers Kaiming, final layer small normal
        fr_linear_layers = [m for m in self.feat_recon_decoder if isinstance(m, nn.Linear)]
        for lin in fr_linear_layers:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if len(fr_linear_layers) > 0:
            fr_last = fr_linear_layers[-1]
            nn.init.normal_(fr_last.weight, mean=0.0, std=0.02)
            if fr_last.bias is not None:
                nn.init.zeros_(fr_last.bias)

        # initialize the topology reconstruction decoder: hidden layers Kaiming, final layer small normal (logit layer)
        topo_linear_layers = [m for m in self.topo_recon_decoder if isinstance(m, nn.Linear)]
        for lin in topo_linear_layers:
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        if len(topo_linear_layers) > 0:
            topo_last = topo_linear_layers[-1]
            nn.init.normal_(topo_last.weight, mean=0.0, std=0.02)
            if topo_last.bias is not None:
                nn.init.zeros_(topo_last.bias)

    
    def feat_recon(self, z):
        return self.feat_recon_decoder(z)

    def feat_recon_loss(self, z, x):
        return F.mse_loss(self.feat_recon(z), x)
    
    @staticmethod
    def prepare_edge_labels(pos_edge_index, num_nodes, negative_sampling_ratio):
        """
        prepare edge labels, including positive and negative samples
        """
        num_pos_edges = pos_edge_index.size(1)
        num_neg_samples = max(1, int(num_pos_edges * negative_sampling_ratio))
        
        # negative sample edges
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            method="dense"
        )

        # merge positive and negative samples
        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).long()
        edge_labels = torch.cat([
            torch.ones(pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ]).to(pos_edge_index.device)
    
        return all_edge_index, edge_labels
    

    def decode_edges(self, z, edge_index):
        """
        edge decoder: predict the probability of the existence of edges
        """
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=1)
        return self.topo_recon_decoder(edge_features).squeeze()
        
    
    def forward(self, query, orig_x, orig_edge_index, topo_recon_ratio):
        feat_recon_loss = self.feat_recon_loss(query, orig_x)
        
        all_edge_index, edge_labels = self.prepare_edge_labels(orig_edge_index, orig_x.size(0), topo_recon_ratio)

        if edge_labels.size(0) == 0 or query.shape[0] == 1:
            return feat_recon_loss, torch.tensor(0.0).to(query.device)
        
        edge_logits = self.decode_edges(query, all_edge_index)
        topo_recon_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, reduction='mean')
        
        return feat_recon_loss, topo_recon_loss
