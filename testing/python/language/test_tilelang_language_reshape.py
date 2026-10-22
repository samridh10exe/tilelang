import tilelang.testing
import tilelang as tl
from tilelang import language as T
import torch
import pytest


def reshape_test(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_reshaped = T.reshape(A, [N // M, M])
            T.copy(A_reshaped, B)

    return main


def run_reshape(N, M, dtype):
    program = reshape_test(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem():
    # Test reshape
    run_reshape(1024, 32, T.float32)
    run_reshape(2048, 64, T.float16)


def reshape_test_smem_1d_2_2d(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((N,), dtype)
            for i in T.Parallel(N):
                A_shared[i] = A[i]

            A_smem_reshaped = T.reshape(A_shared, [N // M, M])
            T.copy(A_smem_reshaped, B)

    return main


def run_reshape_smem_1d_2_2d(N, M, dtype):
    program = reshape_test_smem_1d_2_2d(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N // M, M)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem_1d_2_2d():
    run_reshape_smem_1d_2_2d(1024, 32, T.float32)
    run_reshape_smem_1d_2_2d(2048, 64, T.float16)


def reshape_test_smem_2d_2_1d(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N // M, M), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1) as _:
            A_shared = T.alloc_shared((N // M, M), dtype)
            for i, j in T.Parallel(N // M, M):
                A_shared[i, j] = A[i, j]

            A_smem_reshaped = T.reshape(A_shared, [N])
            T.copy(A_smem_reshaped, B)

    return main


def run_reshape_smem_2d_2_1d(N, M, dtype):
    program = reshape_test_smem_2d_2_1d(N, M, dtype)
    # TODO(lei): reshape cannot apply shared memory
    # layout transform propagation
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_smem_2d_2_1d():
    run_reshape_smem_2d_2_1d(1024, 32, T.float32)
    run_reshape_smem_2d_2_1d(2048, 64, T.float16)


def reshape_fragment_test(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N // M, M), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_shared = T.alloc_shared((N // M, M), dtype, scope="shared")
            A_local = T.alloc_fragment((N // M, M), dtype)
            B_shared = T.alloc_shared((N,), dtype, scope="shared")

            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            A_local_reshape = T.reshape(A_local, [N])
            T.copy(A_local_reshape, B_shared)
            T.copy(B_shared, B)

    return main


def run_reshape_fragment(N, M, dtype):
    program = reshape_fragment_test(N, M, dtype)
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_fragment():
    run_reshape_fragment(1024, 32, T.float32)
    run_reshape_fragment(2048, 64, T.float16)


def reshape_layout_transform_shared(N, M, dtype):
    from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout

    @T.prim_func
    def main(
        A: T.Tensor((N // M, M), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_shared = T.alloc_shared((N // M, M), dtype, scope="shared")

            T.annotate_layout(
                {
                    A_shared: make_mma_swizzle_layout(A_shared),
                }
            )
            T.copy(A, A_shared)
            A_shared_reshape = T.reshape(A_shared, [N])
            T.copy(A_shared_reshape, B)

    return main


def run_reshape_layout_transform_shared(N, M, dtype):
    program = reshape_layout_transform_shared(N, M, dtype)
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return A.reshape(N)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reshape_layout_transform_shared():
    run_reshape_layout_transform_shared(1024, 32, T.float32)
    run_reshape_layout_transform_shared(2048, 64, T.float16)


def reduce_after_reshape_test(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N // M,), dtype),
    ):
        with T.Kernel(1, threads=32) as _:
            A_shared = T.alloc_shared((N,), dtype, scope="shared")
            A_local = T.alloc_fragment((N,), dtype)
            B_local = T.alloc_fragment((N // M,), dtype)

            T.copy(A, A_shared)
            T.copy(A_shared, A_local)
            A_local_reshape = T.reshape(A_local, [N // M, M])
            T.reduce_max(A_local_reshape, B_local, dim=1)
            T.copy(B_local, B)

    return main


def run_reduce_after_reshape(N, M, dtype):
    program = reduce_after_reshape_test(N, M, dtype)
    jit_kernel = tl.compile(
        program,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    profiler = jit_kernel.get_profiler()

    def ref_program(A):
        return torch.max(A.reshape(N // M, M), dim=1).values

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_reduce_after_reshape():
    run_reduce_after_reshape(1024, 32, T.float32)
    run_reduce_after_reshape(2048, 64, T.float16)


def reshape_shape_mismatch_test(N, M, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N // M, M), dtype),
    ):
        with T.Kernel(1) as _:
            A_reshaped = T.reshape(A, [N // M, M + 1])
            T.copy(A_reshaped, B)

    return main


def test_reshape_shape_mismatch():
    with pytest.raises(AssertionError):
        reshape_shape_mismatch_test(1024, 32, T.float32)


def test_reduce_absmax_after_reshape_3d():
    M, N, num_groups, num_per_channels = 2, 384, 3, 128
    threads = 128
    dtype = "int64"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, num_groups), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            A_reshaped = T.reshape(A_local, [M, num_groups, num_per_channels])
            B_local = T.alloc_fragment((M, num_groups), dtype)
            T.copy(A, A_local)
            T.reduce_absmax(A_reshaped, B_local, dim=2)
            T.copy(B_local, B)

    jit_kernel = tl.compile(main, out_idx=-1)
    A_torch = torch.randint(-100, 100, (M, N), dtype=getattr(torch, dtype)).cuda()
    B_torch = jit_kernel(A_torch)

    ref = A_torch.abs().reshape(M, num_groups, num_per_channels).max(dim=2).values
    torch.testing.assert_close(B_torch, ref)


def reshape_smem_pipelined_gemm(use, M=64, N=64, K=256, BK=32, num_stages=2):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        A_tiled: T.Tensor((K // BK, M * BK), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, BK), T.float16)
            A_flat = T.reshape(A_shared, (M * BK,))
            A_view = T.reshape(A_shared, (M, BK))
            B_shared = T.alloc_shared((BK, N), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(K // BK, num_stages=num_stages):
                if use == "copy_flat":
                    # TMA cannot lower this 1-D copy on Hopper (independent of views).
                    T.copy(A_tiled[k, :], A_flat, disable_tma=True)
                else:
                    T.copy(A[0, k * BK], A_view if use == "copy_and_gemm" else A_shared)
                T.copy(B[k * BK, 0], B_shared)
                T.gemm(A_shared if use == "copy_flat" else A_view, B_shared, C_local)
            T.copy(C_local, C)

    return tl.compile(main, out_idx=[3])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("use", ["copy_flat", "gemm", "copy_and_gemm"])
def test_reshape_smem_pipelined(use):
    """A pipelined shared buffer accessed through a view must still be multi-versioned."""
    M, K, BK = 64, 256, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, 64, device="cuda", dtype=torch.float16)
    A_tiled = A.view(M, K // BK, BK).transpose(0, 1).reshape(K // BK, M * BK).contiguous()
    C = reshape_smem_pipelined_gemm(use)(A, A_tiled, B)
    torch.testing.assert_close(C, A.float() @ B.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
