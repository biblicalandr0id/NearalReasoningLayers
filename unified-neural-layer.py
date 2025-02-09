import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedReasoningLayer(nn.Module):
    def __init__(self, data_dim, reasoning_dim, hidden_dim=256):
        super().__init__()
        self.data_dim = data_dim
        self.reasoning_dim = reasoning_dim
        self.hidden_dim = hidden_dim

        # Data perception pathway
        self.data_encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Reasoning pathway
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(reasoning_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Bayesian reasoning module
        self.bayes_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Knowledge integration
        self.knowledge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        # Unified output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Meta-learning parameters
        self.meta_weights = nn.Parameter(torch.ones(3))
        self.consistency_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, data_input, reasoning_input):
        # Normalize meta-weights
        weights = F.softmax(self.meta_weights, dim=0)

        # Process data and reasoning paths
        data_encoded = self.data_encoder(data_input)
        reasoning_encoded = self.reasoning_encoder(reasoning_input)

        # Cross-attention between data and reasoning
        attended_features, _ = self.cross_attention(
            data_encoded.unsqueeze(0),
            reasoning_encoded.unsqueeze(0),
            reasoning_encoded.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)

        # Bayesian integration
        bayes_features = self.bayes_transform(
            torch.cat([data_encoded, reasoning_encoded], dim=-1)
        )

        # Knowledge integration with gating
        knowledge_gate = self.knowledge_gate(
            torch.cat([data_encoded, reasoning_encoded, attended_features], dim=-1)
        )
        
        # Combine all pathways with learned weights
        unified_features = (
            weights[0] * attended_features +
            weights[1] * bayes_features +
            weights[2] * (knowledge_gate * data_encoded)
        )

        # Final transformation with consistency check
        output = self.output_transform(unified_features)
        consistency_score = F.cosine_similarity(output, reasoning_encoded, dim=-1)
        
        # Apply consistency threshold
        mask = (consistency_score > self.consistency_threshold).float()
        output = output * mask.unsqueeze(-1)

        return output, consistency_score

    def get_complexity_loss(self):
        """Compute complexity regularization loss"""
        return torch.norm(self.meta_weights, p=1) + torch.norm(self.consistency_threshold)

    @property
    def reasoning_capacity(self):
        """Estimate current reasoning capacity"""
        return F.sigmoid(self.meta_weights[1] / self.meta_weights[0])
