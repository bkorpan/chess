# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Didactic example of an autoregressive Transformer-based language model.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
- V: Vocabulary size.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


@dataclasses.dataclass
class EncoderStack(hk.Module):

    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
    name: Optional[str] = None  # Optional identifier for the module.

    def __call__(
            self,
            embeddings: jax.Array,  # [B, T, D]
            is_training: bool
    ) -> jax.Array:  # [B, T, D]

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        _, _, model_size = embeddings.shape

        h = embeddings
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.attn_size,
                model_size=model_size,
                w_init=initializer,
            )
            h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm)
            if is_training:
                h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
            h = h + h_attn

            # Then the dense block.
            dense_block = hk.Sequential([
                hk.Linear(int(self.widening_factor * model_size), w_init=initializer),
                jax.nn.gelu,
                hk.Linear(model_size, w_init=initializer),
            ])
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            if is_training:
                h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
            h = h + h_dense

        return h


@dataclasses.dataclass
class Chessformer(hk.Module):

    encoder_stack: EncoderStack
    model_size: int  # Embedding size.
    #position_vocab_size: int
    num_actions: int
    num_tokens: int
    name: Optional[str] = None  # Optional identifier for the module.

    def __call__(
            self,
            tokens: jax.Array,  # Batch of sequences of input tokens, shape [B, T].
            is_training: bool
    ) -> jax.Array:  # Batch of sequences of output token logits, shape [B, T, V].

        # TODO use token embeddings
        # Embed the input tokens and positions.
        #embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        #token_embedding_map = hk.Embed(
        #    self.position_vocab_size, embed_dim=self.model_size, w_init=embed_init)
        #token_embeddings = token_embedding_map(tokens)
        #positional_embeddings = hk.get_parameter(
        #    'positional_embeddings', [seq_len, self.model_size], init=embed_init)
        #input_embeddings = token_embeddings + positional_embeddings  # [B, T, D]

        token_size = tokens.shape[-1]
        tokens = tokens.astype(jnp.float32)
        tokens = tokens.reshape(-1, self.num_tokens, token_size)

        # Use linear layer to embed inputs for now
        input_embeddor = hk.Linear(self.model_size)
        input_embeddings = input_embeddor(tokens)

        # Run the transformer over the inputs.
        board_embeddings = self.encoder_stack(input_embeddings, is_training)  # [B, T, D]
        board_flattened = board_embeddings.reshape(-1, self.model_size * self.num_tokens)
        move_logits = hk.Linear(1)(board_embeddings).reshape(-1, self.num_actions-1)
        pass_logits = hk.Linear(1)(board_flattened)
        policy_logits = jnp.concatenate((move_logits, pass_logits), axis=-1)
        value = hk.Linear(1)(board_flattened)
        value = value.reshape(-1)

        return policy_logits, value
