import math
import torch
import torch.nn as nn
from model.denoiser.diffusion_utils import create_diffusion

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GraphTransformerBlock(nn.Module):
    """Transformer block for graph sequence processing."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Adaptive Layer Norm modulation for conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c, node_mask):
        """
        x: graph sequence features [B, max_n_nodes, hidden_size]
        c: conditioning features (LLM hidden states + timestep) [B, max_n_nodes, hidden_size]
        node_mask: attention mask (can be None)
        """
        # # Apply adaptive layer norm modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention with modulation
        norm_x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Use key_padding_mask for padding positions, ~node_mask because True means "ignore this position"
        padding_mask = ~node_mask if node_mask is not None else None
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=padding_mask)
        x = x + gate_msa * attn_out
        
        # MLP with modulation
        norm_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp * mlp_out
        
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_dim,
        hidden_size,
        z_dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.out_channels = in_dim
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.z_embedder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, hidden_size, bias=True),
        )
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, self.out_channels)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if hasattr(m, 'out_proj') and isinstance(m.out_proj, nn.Linear):
                    nn.init.normal_(m.out_proj.weight, mean=0.0, std=0.02)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

        nn.init.normal_(self.z_embedder[1].weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        nn.init.constant_(self.final_layer[-1].weight, 0)
        nn.init.constant_(self.final_layer[-1].bias, 0)

    
    def forward(self, x, t, context, node_mask):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, C, H, W) tensor of conditions
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)                      # (N, D)

        z = self.z_embedder(context)
        c = t.unsqueeze(1) + z      # (N, T, D)
        for block in self.blocks:
            x = block(x, c, node_mask)         # (N, T, D)
        x = self.final_layer(x)
        return x
    
class GraphDenoiser(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        embed_dim,
        depth,
        timesteps='500',
    ):
        super().__init__()
        self.in_dim = x_dim
        self.ln_pre = nn.LayerNorm(z_dim, elementwise_affine=False)

        self.net = DiT(
            in_dim=x_dim,
            hidden_size=embed_dim,
            z_dim=z_dim,
            depth=depth,
            num_heads=embed_dim//64,
        )
        self.train_diffusion = create_diffusion(timestep_respacing=timesteps, noise_schedule="cosine")

    def forward(self, z, target, node_mask):
        t = torch.randint(self.train_diffusion.num_timesteps, size=(target.shape[0],), device=target.device).long()
        model_kwargs = dict(context=z, node_mask=node_mask)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]

        return loss