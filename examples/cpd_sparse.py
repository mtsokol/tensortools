import argparse
import os
import time

os.environ["SPARSE_BACKEND"] = "Finch"

import tensortools as tt
import numpy as np
import sparse


parser = argparse.ArgumentParser()
parser.add_argument("I", type=int)
parser.add_argument("J", type=int)
parser.add_argument("K", type=int)
parser.add_argument("density", type=float)
args = parser.parse_args()

I, J, K, R, density = args.I, args.J, args.K, 4, args.density

print(f"Running benchmark for shape: ({I}, {J}, {K}), rank: {R}, and density: {density}")

# prepare data
rng_state = np.random.RandomState(12)
X = tt.rand_ktensor((I, J, K), rank=R, random_state=rng_state).full()

mask = sparse.random((I, J, K), density=density , random_state=24)
mask_dense = mask.todense().astype(bool)

mask_sp = sparse.asarray(mask > sparse.asarray(0.0), format="csf")
X_sp = sparse.asarray(X * mask.todense().astype(bool), format="csf")

# wrap in lazy for sparse
mask_sp = sparse.lazy(mask_sp)
X_sp = sparse.lazy(X_sp)

# run tensortools
start_ns = time.perf_counter_ns()
U = tt.mcp_als(X, mask=mask_dense, rank=R, verbose=True, random_state=rng_state)
final_ns = time.perf_counter_ns() - start_ns
print(f"Elapsed time for TensorTools: {final_ns / 1_000_000_000}")

# here is Array API compatible counterpart of the dense version to try
# U2 = tt.mcp_als_array_api(X, mask=mask_dense, rank=R, verbose=True, random_state=rng_state)

# precompile
V = tt.mcp_als_array_api_sparse(
    X, X_sp, mask=mask_dense, mask_sp=mask_sp, rank=R, verbose=False, random_state=rng_state
)
# run Finch
start_ns = time.perf_counter_ns()
V = tt.mcp_als_array_api_sparse(
    X, X_sp, mask=mask_dense, mask_sp=mask_sp, rank=R, verbose=True, random_state=rng_state
)
final_ns = time.perf_counter_ns() - start_ns
print(f"Elapsed time for Sparse Finch: {final_ns / 1_000_000_000}")

try:
    np.testing.assert_allclose(X, U.factors.full(), rtol=6e-2, atol=0)
except AssertionError as e:
    print("Mismatch for TensorTools")
    print(e)
try:
    np.testing.assert_allclose(X, V.factors.full(), rtol=6e-2, atol=0)
except AssertionError as e:
    print("Mismatch for Sparse Finch")
    print(e)

print("\n\n")
