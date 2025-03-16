import os
os.environ["SPARSE_BACKEND"] = "Finch"

import tensortools as tt
import numpy as np
import sparse

I, J, K, R = 100, 100, 50, 4  # dimensions and rank x2
#I, J, K, R = 210, 210, 50, 4  # dimensions and rank x3
#I, J, K, R = 300, 300, 50, 4  # dimensions and rank x4
#I, J, K, R = 400, 400, 50, 4  # dimensions and rank x5


# prepare data
rng_state = np.random.RandomState(12)
X = tt.rand_ktensor((I, J, K), rank=R, random_state=rng_state).full()

mask = sparse.random((I, J, K), density=0.03, random_state=24)
mask_dense = mask.todense().astype(bool)

mask_sp = sparse.asarray(mask > sparse.asarray(0.0), format="csf")
X_sp = sparse.asarray(X * mask.todense().astype(bool), format="csf")


# run tensortools
U = tt.mcp_als(X, mask=mask_dense, rank=R, verbose=True, random_state=rng_state)


# precompile
V = tt.mcp_als_sparse(X, X_sp, mask=mask_dense, mask_sp=mask_sp, rank=R, verbose=False, random_state=rng_state)
# run Finch
V = tt.mcp_als_sparse(X, X_sp, mask=mask_dense, mask_sp=mask_sp, rank=R, verbose=True, random_state=rng_state)


try:
    np.testing.assert_allclose(X, U.factors.full(), rtol=6e-2, atol=0)
except AssertionError as e:
    print(e)
try:
    np.testing.assert_allclose(X, V.factors.full(), rtol=6e-2, atol=0)
except AssertionError as e:
    print(e)
