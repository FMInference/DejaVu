# Copyright (c) 2023, Tri Dao.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=16),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=16),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=16),
    ],
    key=["CACHE_KEY_N"],
)
@triton.jit
def qkv_proj_sparse_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    HEAD_IDX,
    BATCH_IDX,
    BIAS,
    # Matrix dimensions
    N,
    batch_size,
    CACHE_KEY_N,
    stride_a_three,
    stride_a_nheads,
    stride_a_headdim,
    stride_x_batch,
    stride_bias_three,
    stride_y_batch,
    stride_y_three,
    # Meta-parameters
    BLOCK_B: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (batch_size, N)
    - Weight has shape (3, NHEADS, HEADDIM, N)
    - HEAD_IDX has shape (NNZ)
    - BATCH_IDX has shape (NNZ, batch_size)
    - BIAS has shape (3, NHEADS, HEADDIM)
    - Output has shape (batch_size, 3, NHEADS, HEADDIM)
    """
    head_id = tl.load(HEAD_IDX + tl.program_id(0))
    qkv_id = tl.program_id(2)
    rh = tl.program_id(1) * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    rn = tl.arange(0, BLOCK_N)
    rb = tl.arange(0, BLOCK_B)

    BATCH_IDX = BATCH_IDX + tl.program_id(0) * batch_size + rb
    batch_idx = tl.load(BATCH_IDX, mask=rb < batch_size, other=-1)

    A = (
        A
        + qkv_id * stride_a_three
        + head_id * stride_a_nheads
        + (rh[None, :] * stride_a_headdim + rn[:, None])
    )
    X = X + (batch_idx[:, None] * stride_x_batch + rn[None, :])

    acc = tl.zeros((BLOCK_B, BLOCK_HEAD), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_N):
        x = tl.load(X, mask=batch_idx[:, None] >= 0, other=0.0)
        a = tl.load(A)
        acc += tl.dot(x, a)
        A += BLOCK_N
        X += BLOCK_N
    if HAS_BIAS:
        bias = tl.load(BIAS + qkv_id * stride_bias_three + head_id * HEADDIM + rh).to(
            tl.float32
        )
        acc += bias[None, :]

    # rematerialize rb to save registers
    rb = tl.arange(0, BLOCK_B)
    # write back result
    Y = (
        Y
        + qkv_id * stride_y_three
        + head_id * HEADDIM
        + (batch_idx[:, None] * stride_y_batch + rh[None, :])
    )
    tl.store(Y, acc, mask=batch_idx[:, None] >= 0)


def qkv_proj_sparse(
    x: torch.Tensor,
    weight: torch.Tensor,
    head_idx: torch.Tensor,
    batch_idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute y = torch.einsum("bm,thdm->bthd", x, weight), but only for the active heads in @head_idx
    and only for the batch indices in @batch_idx. Negative indices in batch_idx will be ignored.
    :param x: input tensor, (batch_size, hidden_dim)
    :param weight: weight matrix, (3, nheads, head_dim, hidden_dim)
    :param head_idx: int32, (nnz,)
    :param batch_idx: int32, (nnz, batch_size). Negative indices are ignored.
    :param bias: indices, (3, nheads, head_dim)
    :return: result tensor, (batch_size, 3, nheads, head_dim)
    """
    three, nheads, head_dim, hidden_dim = weight.shape
    assert three == 3
    assert head_dim in [32, 64, 128]
    assert hidden_dim % 128 == 0
    batch_size, _ = x.shape
    assert batch_size <= 16
    assert x.shape == (batch_size, hidden_dim)
    n_active = head_idx.shape[0]
    assert n_active <= nheads
    assert head_idx.shape == (n_active,)
    assert head_idx.dtype == torch.int32
    assert batch_idx.shape == (n_active, batch_size)
    assert batch_idx.dtype == torch.int32
    x = x.contiguous()
    if weight.stride(-1) > 1:
        weight = weight.contiguous()
    head_idx = head_idx.contiguous()
    batch_idx = batch_idx.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:
        bias = bias.contiguous()
        assert bias.shape == (
            3,
            nheads,
            head_dim,
        ), "Incompatible dimensions in between weight and bias"
        assert (
            x.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    output = torch.empty(
        batch_size, 3, nheads, head_dim, device=x.device, dtype=x.dtype
    )

    BLOCK_HEAD = 32
    # 1D launch kernel where each head gets its own program.
    grid = lambda META: (n_active, head_dim // BLOCK_HEAD, 3)  # noqa

    qkv_proj_sparse_kernel[grid](
        output,  # data ptrs
        weight,
        x,
        head_idx,
        batch_idx,
        bias,
        hidden_dim,  # shapes
        batch_size,
        hidden_dim // 256,  # key for triton cache (limit number of compilations)
        weight.stride(0),  # strides
        weight.stride(1),
        weight.stride(2),
        x.stride(0),
        bias.stride(0),
        output.stride(0),
        output.stride(1),
        16,  # Can't use kwargs because auto-tuner requires args
        head_dim,
        # 64,
        BLOCK_HEAD=BLOCK_HEAD,
        HAS_BIAS=bias is not None,
        # num_warps=2,
        # num_stages=16
        # num_stages=16
    )

    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=16),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=16),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=8),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=16),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=8),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=16),
    ],
    key=["CACHE_KEY_N"],
)
@triton.jit
def out_proj_sparse_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    HEAD_IDX,
    BIAS,
    # Matrix dimensions
    N,
    n_active_heads,
    batch_size,  # actual batch size
    CACHE_KEY_N,
    stride_a_n,
    stride_x_batch,
    stride_y_batch,
    # Meta-parameters
    BLOCK_B: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BLOCK_B, NHEADS, HEADDIM)
    - Weight has shape (N, NHEADS, HEADDIM)
    - HEAD_IDX has shape (NNZ)
    - BIAS has shape (N)
    - Output has shape (BLOCK_B, N)
    """
    start_n = tl.program_id(0) * BLOCK_N
    rh = tl.arange(0, HEADDIM)
    rn = tl.arange(0, BLOCK_N)
    rb = tl.arange(0, BLOCK_B)

    A = A + start_n * stride_a_n + (rn[None, :] * stride_a_n + rh[:, None])
    X = X + (rb[:, None] * stride_x_batch + rh[None, :])

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for h in range(n_active_heads):
        head_id = tl.load(HEAD_IDX + h)
        x = tl.load(X + head_id * HEADDIM, mask=rb[:, None] < batch_size, other=0.0)
        a = tl.load(A + head_id * HEADDIM)
        acc += tl.dot(x, a)
    if HAS_BIAS:
        bias = tl.load(BIAS + start_n + rn).to(tl.float32)
        acc += bias[None, :]

    # rematerialize rn and rn to save registers
    rb = tl.arange(0, BLOCK_B)
    # write back result
    Y = Y + start_n + (rb[:, None] * stride_y_batch + rn[None, :])
    tl.store(Y, acc, mask=rb[:, None] < batch_size)


def out_proj_sparse(
    x: torch.Tensor,
    weight: torch.Tensor,
    head_idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute y = torch.einsum("bhd,nhd->bn", x, weight), but only for the active heads in @head_idx.
    :param x: input tensor, (batch, nheads, head_dim)
    :param weight: weight matrix, (hidden_dim, nheads, head_dim)
    :param head_idx: int32, (nnz,)
    :param bias: indices, (hidden_dim)
    :return: result tensor, (batch, hidden_dim)
    """
    hidden_dim, nheads, head_dim = weight.shape
    assert head_dim in [32, 64, 128]
    assert hidden_dim % 128 == 0
    batch, _, _ = x.shape
    assert x.shape == (batch, nheads, head_dim)
    n_active = head_idx.shape[0]
    assert head_idx.shape == (n_active,)
    assert head_idx.dtype == torch.int32
    assert batch <= 16
    x = x.contiguous()
    weight = weight.contiguous()
    assert (
        x.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:
        bias = bias.contiguous()
        assert bias.shape == (
            hidden_dim,
        ), "Incompatible dimensions in between weight and bias"
        assert (
            x.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    output = torch.empty(batch, hidden_dim, device=x.device, dtype=x.dtype)

    # 1D launch kernel where each head gets its own program.
    grid = lambda META: (triton.cdiv(hidden_dim, META["BLOCK_N"]),)  # noqa

    out_proj_sparse_kernel[grid](
        output,  # data ptrs
        weight,
        x,
        head_idx,
        bias,
        hidden_dim,  # shapes
        n_active,
        batch,
        hidden_dim // 256,  # key for triton cache (limit number of compilations)
        weight.stride(0),  # strides
        x.stride(0),
        output.stride(0),
        16,  # Can't use kwargs because auto-tuner requires args
        head_dim,
        # 64,
        HAS_BIAS=bias is not None,
        # num_warps=2,
        # num_stages=10
        # num_stages=8
    )

    return output


@triton.jit
def v_cache_copy_sparse_kernel(
    V,  # Pointers to matrices
    V_CACHE,
    HEAD_IDX,
    SEQLEN_IDX,
    PADDING,
    # Matrix dimensions
    nnz_seqlen,
    nheads,
    seqlen,
    head_dim,
    stride_v_cache_nheads,
    stride_v_cache_seqlen,
    stride_v_seqlen,
    stride_v_nheads,
    # Meta-parameters
    BLOCK_B: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    """
    We will not check that the indices are valid, for performance reason.
    - Input V has shape (nnz_seqlen, nheads, headdim)
    - V_CACHE has shape (1, nheads, seqlen, headdim)
    - HEAD_IDX has shape (NNZ)
    - SEQLEN_IDX has shape (NNZ, nnz_seqlen)
    """
    head_id = tl.load(HEAD_IDX + tl.program_id(0))
    rh = tl.program_id(1) * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    rb = tl.arange(0, BLOCK_B)

    if HAS_PADDING:
        padding = tl.load(PADDING)
    else:
        padding = 0

    SEQLEN_IDX = SEQLEN_IDX + tl.program_id(0) * nnz_seqlen + rb
    seqlen_idx = tl.load(SEQLEN_IDX, mask=rb < nnz_seqlen, other=-1)

    V = (
        V
        + head_id * stride_v_nheads
        + (seqlen_idx[:, None] * stride_v_seqlen + rh[None, :])
    )
    V_CACHE = (
        V_CACHE
        + head_id * stride_v_cache_nheads
        + ((seqlen_idx + padding)[:, None] * stride_v_cache_seqlen + rh[None, :])
    )

    v = tl.load(V, mask=seqlen_idx[:, None] >= 0, other=0.0)
    tl.store(V_CACHE, v, mask=seqlen_idx[:, None] >= 0)


def v_cache_copy_sparse(
    v: torch.Tensor,
    v_cache: torch.Tensor,
    head_idx: torch.Tensor,
    seqlen_idx: torch.Tensor,
    padding: torch.Tensor = None,
) -> None:
    """
    :param v: input tensor, (nnz_seqlen, nheads, head_dim)
    :param v_cache: input tensor, (1, nheads, seqlen, head_dim)
    :param head_idx: int32, (nnz,)
    :param seqlen_idx: int32, (nnz, nnz_seqlen). Negative indices are ignored.
    :param padding: int32, (1). Padding is added to indices in seqlen_idx before writing to v_cache
    """
    nnz_seqlen, nheads, head_dim = v.shape
    _, _, seqlen, _ = v_cache.shape
    assert head_dim in [32, 64, 128]
    assert nnz_seqlen <= 16
    assert v_cache.shape == (1, nheads, seqlen, head_dim)
    n_active = head_idx.shape[0]
    assert n_active <= nheads
    assert head_idx.shape == (n_active,)
    assert head_idx.dtype == torch.int32
    assert seqlen_idx.shape == (n_active, nnz_seqlen)
    assert seqlen_idx.dtype == torch.int32
    v = v.contiguous()
    assert v_cache.stride(-1) == 1
    head_idx = head_idx.contiguous()
    seqlen_idx = seqlen_idx.contiguous()
    assert (
        v.dtype == v_cache.dtype
    ), f"v and v_cache must have the same dtype, got {v.dtype} and {v_cache.dtype}"
    if padding is not None:
        assert padding.shape == (1,)
        assert padding.dtype == torch.int32

    BLOCK_HEAD = 32
    # 1D launch kernel where each head gets its own program.
    grid = lambda META: (n_active, head_dim // BLOCK_HEAD)  # noqa

    v_cache_copy_sparse_kernel[grid](
        v,  # data ptrs
        v_cache,
        head_idx,
        seqlen_idx,
        padding,
        nnz_seqlen,  # shapes
        nheads,
        seqlen,
        head_dim,
        v_cache.stride(1),  # strides
        v_cache.stride(2),
        v.stride(0),
        v.stride(1),
        16,  # Can't use kwargs because auto-tuner requires args
        BLOCK_HEAD=BLOCK_HEAD,
        HAS_PADDING=padding is not None
        # num_warps=2,
        # num_stages=16
        # num_stages=16
    )


@triton.jit
def k_cache_copy_sparse_kernel(
    K,  # Pointers to matrices
    K_CACHE,
    HEAD_IDX,
    SEQLEN_IDX,
    PADDING,
    # Matrix dimensions
    nnz_seqlen,
    nheads,
    seqlen,
    head_dim,
    stride_k_cache_nheads,
    stride_k_cache_headdimpack,
    stride_k_cache_seqlen,
    stride_k_seqlen,
    stride_k_nheads,
    # Meta-parameters
    PACKSIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_HEAD_PACK: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    """
    We will not check that the indices are valid, for performance reason.
    - Input K has shape (nnz_seqlen, nheads, headdim)
    - K_CACHE has shape (1, nheads, headdim / PACKSIZE, seqlen, PACKSIZE)
    - HEAD_IDX has shape (NNZ)
    - SEQLEN_IDX has shape (NNZ, nnz_seqlen)
    """
    head_id = tl.load(HEAD_IDX + tl.program_id(0))
    rh = tl.program_id(1) * BLOCK_HEAD_PACK + tl.arange(0, BLOCK_HEAD_PACK)
    rp = tl.arange(0, PACKSIZE)
    rb = tl.arange(0, BLOCK_B)

    if HAS_PADDING:
        padding = tl.load(PADDING)
    else:
        padding = 0

    SEQLEN_IDX = SEQLEN_IDX + tl.program_id(0) * nnz_seqlen + rb
    seqlen_idx = tl.load(SEQLEN_IDX, mask=rb < nnz_seqlen, other=-1)

    K = (
        K
        + head_id * stride_k_nheads
        + (
            seqlen_idx[:, None, None] * stride_k_seqlen
            + rh[None, :, None] * PACKSIZE
            + rp[None, None, :]
        )
    )
    K_CACHE = (
        K_CACHE
        + head_id * stride_k_cache_nheads
        + (
            (seqlen_idx + padding)[:, None, None] * stride_k_cache_seqlen
            + rh[None, :, None] * stride_k_cache_headdimpack
            + rp[None, None, :]
        )
    )

    k = tl.load(K, mask=seqlen_idx[:, None, None] >= 0, other=0.0)
    tl.store(K_CACHE, k, mask=seqlen_idx[:, None, None] >= 0)


def k_cache_copy_sparse(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    head_idx: torch.Tensor,
    seqlen_idx: torch.Tensor,
    padding: torch.Tensor = None,
) -> None:
    """
    :param k: input tensor, (nnz_seqlen, nheads, head_dim)
    :param k_cache: input tensor, (1, nheads, headdim / PACKSIZE, seqlen, PACKSIZE), where
        PACKSIZE = 8 if fp16/bf16 and 4 if fp32.
    :param head_idx: int32, (nnz,)
    :param seqlen_idx: int32, (nnz, nnz_seqlen). Negative indices are ignored.
    :param padding: int32, (1). Padding is added to indices in seqlen_idx before writing to v_cache
    """
    assert k.dtype in [torch.float32, torch.float16, torch.bfloat16]
    packsize = 4 if k.dtype == torch.float32 else 8
    nnz_seqlen, nheads, head_dim = k.shape
    _, _, _, seqlen, _ = k_cache.shape
    assert head_dim in [32, 64, 128]
    assert nnz_seqlen <= 16
    assert k_cache.shape == (1, nheads, head_dim // packsize, seqlen, packsize)
    n_active = head_idx.shape[0]
    assert n_active <= nheads
    assert head_idx.shape == (n_active,)
    assert head_idx.dtype == torch.int32
    assert seqlen_idx.shape == (n_active, nnz_seqlen)
    assert seqlen_idx.dtype == torch.int32
    k = k.contiguous()
    assert k_cache.stride(-1) == 1
    head_idx = head_idx.contiguous()
    seqlen_idx = seqlen_idx.contiguous()
    assert (
        k.dtype == k_cache.dtype
    ), f"k and k_cache must have the same dtype, got {k.dtype} and {k_cache.dtype}"
    if padding is not None:
        assert padding.shape == (1,)
        assert padding.dtype == torch.int32

    BLOCK_HEAD = 32
    # 1D launch kernel where each head gets its own program.
    grid = lambda META: (n_active, head_dim // BLOCK_HEAD)  # noqa

    k_cache_copy_sparse_kernel[grid](
        k,  # data ptrs
        k_cache,
        head_idx,
        seqlen_idx,
        padding,
        nnz_seqlen,  # shapes
        nheads,
        seqlen,
        head_dim,
        k_cache.stride(1),  # strides
        k_cache.stride(2),
        k_cache.stride(3),
        k.stride(0),
        k.stride(1),
        packsize,  # Can't use kwargs because auto-tuner requires args
        16,
        BLOCK_HEAD_PACK=BLOCK_HEAD // packsize,
        HAS_PADDING=padding is not None
        # num_warps=2,
        # num_stages=16
        # num_stages=16
    )


if __name__ == "__main__":
    from src.utils.benchmark import pytorch_profiler
    from einops import rearrange, repeat

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    nheads = 12
    hidden_dim = 12 * 1024

    head_dim = 128
    batch_size = 16
    # n_active_heads = nheads
    n_active_heads = nheads // 3
    x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    weight = torch.randn(3, nheads, head_dim, hidden_dim, device=device, dtype=dtype)
    bias = torch.randn(3, nheads, head_dim, device=device, dtype=dtype)
    # head_idx = torch.arange(nheads, dtype=torch.int32, device=device)
    # head_idx = torch.arange(nheads // 2, dtype=torch.int32, device=device) + nheads // 2
    head_idx = torch.randperm(nheads, device=device, dtype=torch.int32)[:n_active_heads]
    # batch_idx = repeat(torch.arange(batch_size, dtype=torch.int32, device=device),
    #                    "b -> h b", h=n_active_heads).contiguous()
    # batch_idx = batch_idx.flip(-1).contiguous()
    batch_idx = torch.stack(
        [
            torch.randperm(batch_size, dtype=torch.int32, device=device)
            for _ in range(n_active_heads)
        ]
    )
    # batch_mask = torch.randint(0, 2, (n_active_heads, batch_size), dtype=torch.bool, device=device)
    # batch_idx[batch_mask] = -1
    out = qkv_proj_sparse(x, weight, head_idx, batch_idx, bias)
    out_ref = torch.einsum("thdn,bn->bthd", weight, x) + bias
    # out_tt = rearrange(triton.ops.matmul(x, rearrange(weight, "t h d n -> (t h d) n").t()), "b (t h d) -> b t h d", t=3, h=nheads) + bias
    print((out - out_ref)[:, :, head_idx].abs().max())
    # print((out - out_tt)[:, :, head_idx].abs().max())
    # breakpoint()

    # pytorch_profiler(qkv_proj_sparse, x, weight, head_idx, batch_idx, bias)
    # x_tmp = torch.randn(1, hidden_dim, device=device, dtype=dtype)
    # pytorch_profiler(torch.einsum, 'thdm,bm->bthd', weight, x_tmp)
    # # pytorch_profiler(torch.einsum, 'bn,hdn->bhd', x, weight)
    # # pytorch_profiler(torch.sum, weight, dim=-1)
    # # pytorch_profiler(triton.ops.matmul, x, rearrange(weight, "t h d n -> (t h d) n").t())

    nheads = 12
    head_dim = 128
    x = torch.randn(2, nheads, head_dim, device=device, dtype=dtype)
    weight = torch.randn(hidden_dim, nheads, head_dim, device=device, dtype=dtype)
    bias = torch.randn(hidden_dim, device=device, dtype=dtype)
    # head_idx = torch.arange(nheads, dtype=torch.int32, device=device)
    head_idx = torch.randperm(nheads, device=device, dtype=torch.int32)[:n_active_heads]
    out = out_proj_sparse(x, weight, head_idx, bias)
    out_ref = torch.einsum("nhd,bhd->bn", weight[:, head_idx], x[:, head_idx]) + bias
    print((out - out_ref).abs().max())

    # pytorch_profiler(out_proj_sparse, x, weight, head_idx, bias)
    # # pytorch_profiler(torch.einsum, 'bhd,nhd->bn', x, weight)
    # pytorch_profiler(torch.einsum, 'bd,nd->bn',
    #                  rearrange(x, 'b h d -> b (h d)'), rearrange(weight, 'n h d -> n (h d)'))

    batch_size = 16
    seqlen = 32
    v = torch.randn(batch_size, nheads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(1, nheads, seqlen, head_dim, device=device, dtype=dtype)
    padding = torch.tensor([16], dtype=torch.int32, device=device)
    v_cache_copy_sparse(v, v_cache, head_idx, batch_idx, padding)
    # pytorch_profiler(v_cache_copy_sparse, v, v_cache, head_idx, batch_idx)
    v[batch_idx[0], 0, 0]
    v_cache[0, 0, batch_idx[0] + padding, 0]

    packsize = 4 if dtype == torch.float32 else 8
    k = torch.randn(batch_size, nheads, head_dim, device=device, dtype=dtype)
    k_cache = torch.zeros(
        1, nheads, head_dim // packsize, seqlen, packsize, device=device, dtype=dtype
    )
    k_cache_copy_sparse(k, k_cache, head_idx, batch_idx, padding)
    k_cache_og = rearrange(k_cache, "1 h d s p -> 1 h s (d p)")
    pytorch_profiler(k_cache_copy_sparse, k, k_cache, head_idx, batch_idx)
    k[batch_idx[0], 0, 0]
    k_cache_og[0, 0, batch_idx[0] + padding, 0]
