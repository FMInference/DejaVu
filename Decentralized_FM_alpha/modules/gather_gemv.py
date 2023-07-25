from typing import Optional

import torch
import triton
import triton.language as tl

from .k_activations import relu, gelu, gelu_approx, squared_relu


@triton.jit
def apply_activation(x, ACTIVATION: tl.constexpr):
    if ACTIVATION == "relu":
        x = relu(x)
    elif ACTIVATION == "gelu":
        x = gelu(x)
    elif ACTIVATION == "gelu_approx":
        x = gelu_approx(x)
    elif ACTIVATION == "squared_relu":
        x = squared_relu(x)
    return x


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_gemv_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = ACTIVATION(A[IDX, :] @ X + BIAS[IDX]), where A is a dense matrix with
    Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, N)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - BIAS has shape (Z,)
    - Output has shape (BATCHSIZE, M)
    """
    start_m = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X = X + rn
    if HAS_BIAS:  # This load is slow because it's not coalesced, so we kick it off early
        bias = tl.load(BIAS + idx, mask=rm < M, other=0.0).to(tl.float32)

    # We're copying-pasting code because it's hard to get Triton to batch things
    # without massive slowdown.
    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            x2 = tl.load(X + 2 * N) if EVEN_N else tl.load(X + 2 * N, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
            acc2 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
        acc2 = apply_activation(acc2, ACTIVATION=ACTIVATION)
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n in range(N, 0, -BLOCK_N):
            a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < n, other=0.0)
            x0 = tl.load(X) if EVEN_N else tl.load(X, mask=rn < n, other=0.0)
            x1 = tl.load(X + N) if EVEN_N else tl.load(X + N, mask=rn < n, other=0.0)
            x2 = tl.load(X + 2 * N) if EVEN_N else tl.load(X + 2 * N, mask=rn < n, other=0.0)
            x3 = tl.load(X + 3 * N) if EVEN_N else tl.load(X + 3 * N, mask=rn < n, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[None, :], 1)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[None, :], 1)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[None, :], 1)
            acc3 += tl.sum(a.to(tl.float32) * x3.to(tl.float32)[None, :], 1)
            A += BLOCK_N
            X += BLOCK_N
        if HAS_BIAS:
            acc0 += bias
            acc1 += bias
            acc2 += bias
            acc3 += bias
        # optional: fused activation (while the data is in shared memory)
        acc0 = apply_activation(acc0, ACTIVATION=ACTIVATION)
        acc1 = apply_activation(acc1, ACTIVATION=ACTIVATION)
        acc2 = apply_activation(acc2, ACTIVATION=ACTIVATION)
        acc3 = apply_activation(acc3, ACTIVATION=ACTIVATION)

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    Y = Y + rm
    tl.store(Y, acc0, mask=rm < M)
    if BATCHSIZE >= 2:
        tl.store(Y + M, acc1, mask=rm < M)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * M, acc2, mask=rm < M)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * M, acc3, mask=rm < M)


def gather_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = 'id',
) -> torch.Tensor:
    """
    Compute y = activation(x @ weight[idx, :].T + bias[idx]).
    :param x: input tensor, (batch, N)
    :param weight: weight matrix, (Z, N)
    :param idx: indices, (M,)
    :param bias: indices, (Z,)
    :return: result tensor, (batch, M)
    """
    assert activation in ['id', 'relu', 'gelu', 'gelu_approx', 'squared_relu']
    Z, N = weight.shape
    batch, _ = x.shape
    assert x.shape == (batch, N)
    assert batch in [1, 2, 3, 4]
    M, = idx.shape
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:

        bias = bias.contiguous()
        assert bias.shape == (Z,), "Incompatible dimensions in between weight and bias"
        assert x.dtype == bias.dtype, f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    output = torch.empty(batch, M, device=x.device, dtype=x.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa

    gather_gemv_kernel[grid](
        output, # data ptrs
        weight,
        x,
        idx,
        bias,
        M,  # shapes
        N,
        M // 512,  # key for triton cache (limit number of compilations)
        N // 1024,  # key for triton cache (limit number of compilations)
        weight.stride(0),  # strides
        batch,  # Can't use kwargs because auto-tuner requires args
        HAS_BIAS=bias is not None,
        ACTIVATION=activation
    )

    return output


# This implementation iterates over the m dimension of the matrix, so the loading slower.
# The other implementation iterate over the n dimension, so the loading is faster.
# The tradeoff is that the other implementation has to do atomic add on the output.
# But overall it's still faster.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 64}, num_warps=8),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_n = tl.program_id(0)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + rm
    IDX = IDX + rm

    if BATCHSIZE == 1:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            a = tl.load(A_ptr) if EVEN_N else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
    elif BATCHSIZE == 2:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            a = tl.load(A_ptr) if EVEN_N else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
    elif BATCHSIZE == 3:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            x2 = tl.load(X + 2 * M, mask=rm < m, other=0.0)
            a = tl.load(A_ptr) if EVEN_N else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
            acc2 += bias
    elif BATCHSIZE == 4:
        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for m in range(M, 0, -BLOCK_M):
            idx = tl.load(IDX, mask=rm < m, other=0)
            A_ptr = A + (idx[:, None] * stride_am + rn[None, :])
            x0 = tl.load(X, mask=rm < m, other=0.0)
            x1 = tl.load(X + M, mask=rm < m, other=0.0)
            x2 = tl.load(X + 2 * M, mask=rm < m, other=0.0)
            x3 = tl.load(X + 3 * M, mask=rm < m, other=0.0)
            a = tl.load(A_ptr) if EVEN_N else tl.load(A_ptr, mask=rn[None, :] < N, other=0.0)
            acc0 += tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
            acc1 += tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
            acc2 += tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
            acc3 += tl.sum(a.to(tl.float32) * x3.to(tl.float32)[:, None], 0)
            IDX += BLOCK_M
            X += BLOCK_M
        if HAS_BIAS:
            bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
            acc0 += bias
            acc1 += bias
            acc2 += bias
            acc3 += bias

    # write back result
    Y = Y + rn
    tl.store(Y, acc0, mask=rn < N)
    if BATCHSIZE >= 2:
        tl.store(Y + N, acc1, mask=rn < N)
    if BATCHSIZE >= 3:
        tl.store(Y + 2 * N, acc2, mask=rn < N)
    if BATCHSIZE >= 4:
        tl.store(Y + 3 * N, acc3, mask=rn < N)


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("Y")),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gather_transposed_gemv_atomicadd_kernel(
    Y,  # Pointers to matrices
    A,
    X,
    IDX,
    BIAS,
    # Matrix dimensions
    M,
    N,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):

    """
    Kernel for computing Y = A[IDX, :]^T @ X + BIAS, where A is a dense matrix
    with Z rows and N columns. We also batch across the batch dimension of the input X.
    We will not check that the indices are valid, for performance reason.
    - Input X has shape (BATCHSIZE, M)
    - Weight has shape (Z, N)
    - IDX has shape (M), where M is the number of non-zero rows in A
    - Bias has shape (N)
    - Output has shape (BATCHSIZE, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices for rows (resp. col) of A
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    IDX = IDX + rm
    idx = tl.load(IDX, mask=rm < M, other=0)
    A = A + (idx[:, None] * stride_am + rn[None, :])
    X = X + rm
    Y = Y + rn
    if HAS_BIAS:
        BIAS = BIAS + rn
    a = tl.load(A) if EVEN_N else tl.load(A, mask=rn[None, :] < N, other=0.0)
    if BATCHSIZE == 1:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
    elif BATCHSIZE == 2:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
    elif BATCHSIZE == 3:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        x2 = tl.load(X + 2 * M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
                acc2 += bias
    elif BATCHSIZE == 4:
        x0 = tl.load(X, mask=rm < M, other=0.0)
        x1 = tl.load(X + M, mask=rm < M, other=0.0)
        x2 = tl.load(X + 2 * M, mask=rm < M, other=0.0)
        x3 = tl.load(X + 3 * M, mask=rm < M, other=0.0)
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
        acc1 = tl.sum(a.to(tl.float32) * x1.to(tl.float32)[:, None], 0)
        acc2 = tl.sum(a.to(tl.float32) * x2.to(tl.float32)[:, None], 0)
        acc3 = tl.sum(a.to(tl.float32) * x3.to(tl.float32)[:, None], 0)
        if HAS_BIAS:
            if start_m == 0:
                bias = tl.load(BIAS, mask=rn < N, other=0.0).to(tl.float32)
                acc0 += bias
                acc1 += bias
                acc2 += bias
                acc3 += bias
    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(Y, acc0, mask=rn < N)
    if BATCHSIZE >= 2:
        tl.atomic_add(Y + N, acc1, mask=rn < N)
    if BATCHSIZE >= 3:
        tl.atomic_add(Y + 2 * N, acc2, mask=rn < N)
    if BATCHSIZE >= 4:
        tl.atomic_add(Y + 3 * N, acc3, mask=rn < N)


def gather_transposed_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute y = weight[idx, :]^T @ x + bias.
    :param x: input tensor
    :param weight: weight matrix
    :param idx: indices
    :return: result tensor
    """
    Z, N = weight.shape
    M, = idx.shape
    batch, _ = x.shape
    assert x.shape == (batch, M,)
    x = x.contiguous()
    if weight.stride(1) > 1:
        weight = weight.contiguous()
    assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    if bias is not None:
        bias = bias.contiguous()
        assert bias.shape == (N,), "Incompatible dimensions in between weight and bias"
        assert x.dtype == bias.dtype, f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"

    # kernel_type = 'deterministic'
    kernel_type = 'atomicadd'  # This always seems to be faster for now

    output = torch.empty(batch, N, device=x.device,
                         dtype=x.dtype if kernel_type == 'deterministic' else torch.float32)

    # 1D launch kernel where each block gets its own program.
    if kernel_type == 'deterministic':
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)  # noqa
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))  # noqa

    kernel = (gather_transposed_gemv_kernel if kernel_type == 'deterministic'
              else gather_transposed_gemv_atomicadd_kernel)
    kernel[grid](
        output, # data ptrs
        weight,
        x,
        idx,
        bias,
        M,  # shapes
        N,
        M // 1024,  # key for triton cache (limit number of compilations)
        N // 32,
        weight.stride(0),  # strides
        batch,  # can't use kwargs because auto-tuner requires args
        HAS_BIAS=bias is not None
    )

    return output.to(dtype=x.dtype)
