import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * (-embeddings))
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, scale1=None, shift1=None, scale2=None, shift2=None):
        if scale1 is not None and shift1 is not None:
            x_norm = self.norm1(x) * (1 + scale1) + shift1
        else:
            x_norm = self.norm1(x)

        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        if scale2 is not None and shift2 is not None:
            x_norm = self.norm2(x) * (1 + scale2) + shift2
        else:
            x_norm = self.norm2(x)

        x = x + self.mlp(x_norm)
        return x


class ConditionalProteinDiT(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, depth=8, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 添加BatchNorm层
        self.bn = nn.BatchNorm1d(hidden_dim)

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.condition_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, mlp_ratio=4) for _ in range(depth)
        ])

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        self.positional_embedding = nn.Embedding(100, hidden_dim)

    def forward(self, x, t, condition):
        h = self.input_proj(x)
        h = h.permute(0, 2, 1)  # 为BatchNorm准备
        h = self.bn(h)
        h = h.permute(0, 2, 1)  # 恢复形状

        positions = torch.arange(
            0, x.size(1), dtype=torch.long, device=x.device
        ).unsqueeze(0).repeat(x.size(0), 1)
        pos_emb = self.positional_embedding(positions)
        h = h + pos_emb

        t_emb = self.time_embed(t)
        cond_emb = self.condition_proj(condition)
        cond_summary = cond_emb.mean(dim=1)
        fused_emb = torch.cat([t_emb, cond_summary], dim=-1)
        fused_emb = self.fusion_layer(fused_emb)

        mod_params = self.adaLN_modulation(fused_emb)
        scale1, shift1, scale2, shift2, gate_scale, gate_shift = mod_params.chunk(6, dim=1)

        for block in self.blocks:
            h_with_cond = h + cond_emb
            h = block(
                h_with_cond,
                scale1=scale1.unsqueeze(1),
                shift1=shift1.unsqueeze(1),
                scale2=scale2.unsqueeze(1),
                shift2=shift2.unsqueeze(1)
            )

        h = self.output_norm(h) * (1 + gate_scale.unsqueeze(1)) + gate_shift.unsqueeze(1)
        return self.output_proj(h)


class KinaseClassifier(nn.Module):
    def __init__(self, esm_model, tokenizer, hidden_dim=256, freeze_esm=True):
        super().__init__()
        self.esm = esm_model
        self.tokenizer = tokenizer

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

        # 增加Dropout率并添加BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(esm_model.config.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward_with_embeddings(self, input_embeddings, attention_mask):
        position_ids = torch.arange(
            0, input_embeddings.size(1),
            dtype=torch.long,
            device=input_embeddings.device
        )
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)
        position_embeddings = self.esm.embeddings.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings

        if hasattr(self.esm.embeddings, 'LayerNorm') and callable(self.esm.embeddings.LayerNorm):
            embeddings = self.esm.embeddings.LayerNorm(embeddings)
        elif hasattr(self.esm.embeddings, 'layer_norm') and callable(self.esm.embeddings.layer_norm):
            embeddings = self.esm.embeddings.layer_norm(embeddings)
        elif hasattr(self.esm.embeddings, 'LayerNorm'):
            embeddings = self.esm.embeddings.LayerNorm(embeddings)
        else:
            layer_norm = nn.LayerNorm(self.esm.config.hidden_size).to(input_embeddings.device)
            embeddings = layer_norm(embeddings)

        embeddings = self.esm.embeddings.dropout(embeddings)
        encoder_outputs = self.esm.encoder(
            embeddings,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze()

    def forward(self, input_ids=None, attention_mask=None, input_embeddings=None):
        if input_embeddings is not None:
            return self.forward_with_embeddings(input_embeddings, attention_mask)

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze()