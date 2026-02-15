
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Any, Optional


def gelu_exact(x):
    """Exact GELU (matches PyTorch default)."""
    return nn.gelu(x, approximate=False)


class MLP(nn.Module):
    hidden_size: int
    activation: Callable = gelu_exact
    
    @nn.compact
    def __call__(self, x):
        input_size = x.shape[-1]
        x = nn.Dense(self.hidden_size, use_bias=False, name='linear1')(x)
        x = self.activation(x)
        x = nn.Dense(input_size, use_bias=False, name='linear2')(x)
        return x


class MultiHeadAttention(nn.Module):
    nhead: int
    d_model: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, x_kv=None, training: bool = False, 
                 reuse_first_head_kv: bool = False):
        """Multi-head attention.
        
        Args:
            x: [..., seq_len_q, d_model] - queries come from here
            x_kv: [..., seq_len_kv, d_model] - keys/values come from here
                  (if None, self-attention: x_kv = x)
            reuse_first_head_kv: if True, only compute KV for first head 
                  then broadcast to all heads (multiquery attention)
        """
        head_dim = self.d_model // self.nhead
        
        if x_kv is None:
            x_kv = x
        
        # Fused QKV kernel: [d_model, 3*d_model]
        qkv_kernel = self.param(
            'w_qkv_kernel',
            nn.initializers.lecun_normal(),
            (x.shape[-1], 3 * self.d_model)
        )
        
        q = x @ qkv_kernel[:, :self.d_model]
        
        if reuse_first_head_kv:
            # Only use the first head's KV weights
            k_full = x_kv @ qkv_kernel[:, self.d_model:2*self.d_model]
            v_full = x_kv @ qkv_kernel[:, 2*self.d_model:]
            # Take only first head
            k_first = k_full[..., :head_dim]  # [..., seq_kv, head_dim]
            v_first = v_full[..., :head_dim]  # [..., seq_kv, head_dim]
            # Broadcast to all heads: [..., seq_kv, nhead, head_dim]
            k = jnp.broadcast_to(k_first[..., None, :], 
                                k_first.shape[:-1] + (self.nhead, head_dim))
            v = jnp.broadcast_to(v_first[..., None, :],
                                v_first.shape[:-1] + (self.nhead, head_dim))
            q = q.reshape(q.shape[:-1] + (self.nhead, head_dim))
        else:
            k = x_kv @ qkv_kernel[:, self.d_model:2*self.d_model]
            v = x_kv @ qkv_kernel[:, 2*self.d_model:]
            q = q.reshape(q.shape[:-1] + (self.nhead, head_dim))
            k = k.reshape(k.shape[:-1] + (self.nhead, head_dim))
            v = v.reshape(v.shape[:-1] + (self.nhead, head_dim))
        
        # [..., S, H, D] -> [..., H, S, D]
        q = jnp.moveaxis(q, -2, -3)
        k = jnp.moveaxis(k, -2, -3)
        v = jnp.moveaxis(v, -2, -3)
        
        scale = 1.0 / jnp.sqrt(jnp.float32(head_dim))
        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn_weights = nn.softmax(attn_logits, axis=-1)
        
        if training:
            attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=False)
            
        output = jnp.matmul(attn_weights, v)  # [..., H, S_q, D]
        output = jnp.moveaxis(output, -3, -2)  # [..., S_q, H, D]
        output = output.reshape(output.shape[:-2] + (self.d_model,))
        
        output = nn.Dense(self.d_model, use_bias=False, name='w_out')(output)
        return output


class PerFeatureEncoderLayer(nn.Module):
    config: Any
    dim_feedforward: int
    
    @nn.compact
    def __call__(self, x, single_eval_pos: int = 0, training: bool = False):
        """
        x: [batch, items, features, d_model]
        single_eval_pos: position separating train/test items
        
        Uses multiquery item attention: 
          - train items self-attend
          - test items attend to train items with reuse_first_head_kv=True
        """
        # 1. Attention between features (self-attention)
        y = MultiHeadAttention(
            nhead=self.config.nhead, d_model=self.config.emsize, 
            name='self_attn_between_features'
        )(x, training=training)
        x = nn.LayerNorm(epsilon=1e-5, use_scale=False, use_bias=False, 
                        name='layer_norms_0')(x + y)
        
        # 2. Attention between items (multiquery for test set)
        # Transpose: [batch, items, features, d_model] -> [batch, features, items, d_model]
        x_T = jnp.swapaxes(x, 1, 2)
        
        item_attn = MultiHeadAttention(
            nhead=self.config.nhead, d_model=self.config.emsize,
            name='self_attn_between_items'
        )
        
        if single_eval_pos > 0 and single_eval_pos < x.shape[1]:
            x_train_T = x_T[:, :, :single_eval_pos, :]  # [batch, features, train, d_model]
            x_test_T = x_T[:, :, single_eval_pos:, :]    # [batch, features, test, d_model]
            
            # Train items: self-attention (Q=train, KV=train)
            y_train_T = item_attn(x_train_T, x_kv=x_train_T, training=training)
            
            # Test items: cross-attention to train items with multiquery
            y_test_T = item_attn(x_test_T, x_kv=x_train_T, training=training,
                                reuse_first_head_kv=True)
            
            # Concatenate train and test outputs
            y_T = jnp.concatenate([y_train_T, y_test_T], axis=2)
        else:
            y_T = item_attn(x_T, training=training)
        
        y = jnp.swapaxes(y_T, 1, 2)
        x = nn.LayerNorm(epsilon=1e-5, use_scale=False, use_bias=False, 
                        name='layer_norms_1')(x + y)
        
        # 3. MLP
        y = MLP(hidden_size=self.dim_feedforward, name='mlp')(x)
        x = nn.LayerNorm(epsilon=1e-5, use_scale=False, use_bias=False, 
                        name='layer_norms_2')(x + y)
        
        return x


class FlaxPerFeatureTransformer(nn.Module):
    config: Any
    n_out: int
    col_embedding: jnp.ndarray
    
    @nn.compact
    def __call__(self, embedded_input, single_eval_pos: int, training: bool = False,
                 skip_positional_embedding: bool = False):
        # embedded_input: [batch, items, features, d_model]
        x = embedded_input
        batch, items, features, d_model = x.shape
        
        # Feature Positional Embedding (only for x features, not y)
        if not skip_positional_embedding:
            n_x_features = features - 1  # last feature is y
            if n_x_features > 0 and n_x_features <= self.col_embedding.shape[0]:
                embs_raw = self.col_embedding[:n_x_features]
            elif n_x_features > self.col_embedding.shape[0]:
                embs_raw = jnp.pad(self.col_embedding, 
                                  ((0, n_x_features - self.col_embedding.shape[0]), (0, 0)))
            else:
                embs_raw = None
            
            if embs_raw is not None:
                embs = nn.Dense(d_model, name='feature_positional_embedding_embeddings')(embs_raw)
                x = x.at[:, :, :n_x_features, :].add(embs[None, None, :, :])
        
        # Transformer Layers
        for i in range(self.config.nlayers):
            x = PerFeatureEncoderLayer(
                config=self.config,
                dim_feedforward=self.config.emsize * self.config.nhid_factor,
                name=f'layers_{i}'
            )(x, single_eval_pos=single_eval_pos, training=training)
            
        # Decoder
        x_out = x[:, single_eval_pos:, -1, :]  # [batch, test_len, d_model]
        
        y = nn.Dense(self.config.emsize * self.config.nhid_factor, 
                    name='decoder_linear1')(x_out)
        y = gelu_exact(y)
        y = nn.Dense(self.n_out, name='decoder_linear2')(y)
        
        return y
