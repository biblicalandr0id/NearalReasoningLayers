import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasonedCompressionLayer(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Dimensions
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        
        # Vector Quantization Components
        self.codebook_size = 512
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, bottleneck_dim))
        
        # Attention-based Compression
        self.attention_compressor = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Logical Structure Preservation
        self.structure_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, bottleneck_dim)
        )
        
        # Residual Gating
        self.gate = nn.Sequential(
            nn.Linear(input_dim + bottleneck_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Compression Networks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        # Logical Consistency Checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Entropy Estimation
        self.entropy_estimator = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.LayerNorm(bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, 1)
        )
        
    def vector_quantize(self, x):
        # Calculate distances to codebook vectors
        distances = torch.cdist(x, self.codebook)
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=-1)
        quantized = F.embedding(indices, self.codebook)
        
        # Straight-through estimator
        x_q = x + (quantized - x).detach()
        
        # Calculate commitment loss
        commitment_loss = F.mse_loss(x.detach(), quantized)
        
        return x_q, commitment_loss
    
    def compress_attention(self, x):
        # Self-attention based compression
        attended, _ = self.attention_compressor(x, x, x)
        
        # Extract key features through attention
        compressed = self.structure_encoder(attended)
        
        return compressed
    
    def estimate_logical_consistency(self, compressed):
        # Check if compressed representation maintains logical structure
        consistency_score = self.consistency_checker(compressed)
        return consistency_score
    
    def calculate_entropy(self, compressed):
        # Estimate information content of compressed representation
        entropy = self.entropy_estimator(compressed)
        return entropy
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initial compression through attention
        attention_compressed = self.compress_attention(x)
        
        # Main compression pathway
        encoded = self.encoder(x)
        
        # Vector quantization
        quantized, commitment_loss = self.vector_quantize(encoded)
        
        # Logical consistency check
        consistency_score = self.estimate_logical_consistency(quantized)
        
        # Entropy estimation
        entropy = self.calculate_entropy(quantized)
        
        # Adaptive gating based on input and compressed representation
        gate_input = torch.cat([x, quantized], dim=-1)
        gate_value = self.gate(gate_input)
        
        # Reconstruction with residual connection
        decoded = self.decoder(quantized)
        output = gate_value * decoded + (1 - gate_value) * x
        
        # Calculate compression ratio
        compression_ratio = self.input_dim / self.bottleneck_dim
        
        return {
            'compressed': quantized,
            'reconstructed': output,
            'commitment_loss': commitment_loss,
            'consistency_score': consistency_score,
            'entropy': entropy,
            'compression_ratio': compression_ratio,
            'attention_compressed': attention_compressed
        }
    
    def get_compression_stats(self, x):
        with torch.no_grad():
            results = self.forward(x)
            stats = {
                'compression_ratio': results['compression_ratio'],
                'consistency': results['consistency_score'].mean().item(),
                'entropy': results['entropy'].mean().item(),
                'reconstruction_error': F.mse_loss(x, results['reconstructed']).item()
            }
            return stats

class CompressionLoss(nn.Module):
    def __init__(self, 
                 reconstruction_weight=1.0,
                 commitment_weight=0.25,
                 consistency_weight=0.5,
                 entropy_weight=0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.commitment_weight = commitment_weight
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
    
    def forward(self, x, results):
        # Reconstruction loss
        recon_loss = F.mse_loss(x, results['reconstructed'])
        
        # Combined loss with weights
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.commitment_weight * results['commitment_loss'] +
            self.consistency_weight * (1 - results['consistency_score'].mean()) +
            self.entropy_weight * torch.abs(results['entropy']).mean()
        )
        
        return total_loss