from functools import partial

from .block import Block
from .mha import MHA
from .mlp import MLP
from .memory_query import MemoryQueryBlock


def _create_mha_cls(num_attention_heads: int, attn_dropout: float, causal: bool):
    return partial(
        MHA, num_heads=num_attention_heads, attn_dropout=attn_dropout, causal=causal
    )


def _create_mlp_cls(intermediate_size: int):
    return partial(MLP, hidden_dim=intermediate_size)


def create_block(
    emb_dim: int,
    intermediate_size: int,
    num_attention_heads: int,
    attn_dropout: float | None = None,
    block_resid_dropout1: float | None = None,
    block_resid_dropout2: float | None = None,
    causal: bool = False,
) -> Block:
    """Create a BF16 block; use ``causal=True`` for autoregressive attention."""
    mixer_cls = _create_mha_cls(
        num_attention_heads=num_attention_heads,
        attn_dropout=0.0 if attn_dropout is None else attn_dropout,
        causal=causal,
    )
    mlp_cls = _create_mlp_cls(intermediate_size)

    return Block(
        emb_dim=emb_dim,
        mixer_cls=mixer_cls,
        mlp_cls=mlp_cls,
        resid_dropout1=block_resid_dropout1
        if block_resid_dropout1 is not None
        else 0.0,
        resid_dropout2=block_resid_dropout2
        if block_resid_dropout2 is not None
        else 0.0,
    )


def create_memory_query_block(
    emb_dim: int,
    intermediate_size: int,
    num_attention_heads: int,
    attn_dropout: float | None = None,
    entry_resid_dropout: float | None = None,
    sa_resid_dropout: float | None = None,
    ca_resid_dropout: float | None = None,
    norm_eps: float = 1e-5,
) -> MemoryQueryBlock:
    """Causal memory SA -> shared V/Q CA -> shared FFN, with BF16 branch/residual."""
    return MemoryQueryBlock(
        emb_dim,
        intermediate_size,
        num_attention_heads,
        attn_dropout=0.0 if attn_dropout is None else attn_dropout,
        entry_resid_dropout=0.0 if entry_resid_dropout is None else entry_resid_dropout,
        sa_resid_dropout=0.0 if sa_resid_dropout is None else sa_resid_dropout,
        ca_resid_dropout=0.0 if ca_resid_dropout is None else ca_resid_dropout,
        norm_eps=norm_eps,
    )
