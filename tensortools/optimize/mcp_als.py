"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
import sparse
from scipy import linalg

from tensortools.operations import unfold, khatri_rao
from tensortools.tensors import KTensor
from tensortools.optimize import FitResult, optim_utils


def mcp_als(X, rank, mask, random_state=None, init='randn', skip_modes=[], **options):
    """Fits CP Decomposition with missing data using Alternating Least Squares (ALS).

    Parameters
    ----------
    X : (I_1, ..., I_N) array_like
        A tensor with ``X.ndim >= 3``.

    rank : integer
        The `rank` sets the number of components to be computed.

    mask : (I_1, ..., I_N) array_like
        A binary tensor with the same shape as ``X``. All entries equal to zero
        correspond to held out or missing data in ``X``. All entries equal to
        one correspond to observed entries in ``X`` and the decomposition is
        fit to these datapoints.

    random_state : integer, ``RandomState``, or ``None``, optional (default ``None``)
        If integer, sets the seed of the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, use the RandomState instance used by ``numpy.random``.

    init : str, or KTensor, optional (default ``'randn'``).
        Specifies initial guess for KTensor factor matrices.
        If ``'randn'``, Gaussian random numbers are used to initialize.
        If ``'rand'``, uniform random numbers are used to initialize.
        If KTensor instance, a copy is made to initialize the optimization.

    skip_modes : iterable, optional (default ``[]``).
        Specifies modes of the tensor that are not fit. This can be
        used to fix certain factor matrices that have been previously
        fit.

    options : dict, specifying fitting options.

        tol : float, optional (default ``tol=1E-5``)
            Stopping tolerance for reconstruction error.

        max_iter : integer, optional (default ``max_iter = 500``)
            Maximum number of iterations to perform before exiting.

        min_iter : integer, optional (default ``min_iter = 1``)
            Minimum number of iterations to perform before exiting.

        max_time : integer, optional (default ``max_time = np.inf``)
            Maximum computational time before exiting.

        verbose : bool ``{'True', 'False'}``, optional (default ``verbose=True``)
            Display progress.


    Returns
    -------
    result : FitResult instance
        Object which holds the fitted results. It provides the factor matrices
        in form of a KTensor, ``result.factors``.


    Notes
    -----
    Fitting CP decompositions with missing data can be exploited to perform
    cross-validation.

    References
    ----------
    Williams, A. H.
    "Solving Least-Squares Regression with Missing Data."
    http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    """

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, _ = optim_utils._get_initial_ktensor(init, X, rank, random_state, scale_norm=False)
    result = FitResult(U, 'MCP_ALS', **options)
    normX = np.linalg.norm((X * mask))

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            # ii) Unfold data and mask along the nth mode.
            # inlined `unfold` for readability
            unf = np.moveaxis(X, n, 0).reshape((X.shape[n], -1))  # i_n x N
            m = np.moveaxis(mask, n, 0).reshape((mask.shape[n], -1))  # i_n x N

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            # inlined `khatri_rao` for readability
            n_columns = components[0].shape[1]
            n_factors = len(components)
            start = ord('a')
            common_dim = 'z'
            target = ''.join(chr(start + i) for i in range(n_factors))
            source = ','.join(i+common_dim for i in target)
            operation = source+'->'+target+common_dim
            kr = np.einsum(operation, *components).reshape((-1, n_columns))
            krt = kr.T

            # iv) Broadcasted solve of linear systems.
            # Left hand side of equations, R x R x X.shape[n]
            # Right hand side of equations, X.shape[n] x R x 1
            lhs_stack = np.matmul(m[:, None, :] * krt[None, :, :], krt.T[None, :, :])
            rhs_stack = np.dot(unf * m, krt.T)[:, :, None]

            # vi) Update factor.
            U[n] = np.linalg.solve(lhs_stack, rhs_stack).reshape(X.shape[n], rank)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        obj = linalg.norm(mask * (U.full() - X)) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()


def mcp_als_array_api(X, rank, mask, random_state=None, init='randn', skip_modes=[], **options):
    xp = np

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, _ = optim_utils._get_initial_ktensor(init, X, rank, random_state, scale_norm=False)
    result = FitResult(U, 'MCP_ALS', **options)
    normX = np.linalg.norm((X * mask))

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            unf = xp.moveaxis(X, n, 0)
            m = xp.moveaxis(mask, n, 0)

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            kr = components[0][:, None, :] * components[1][None, :, :]
            krt = xp.moveaxis(kr, -1, 0)

            # iv) Broadcasted solve of linear systems.
            lhs_stack = xp.sum(
                m[:, None, ..., None] * krt[None, ..., None] * kr[None, None, ...], axis=(-2, -3)
            )
            rhs_stack = xp.tensordot(unf * m, kr, axes=((-1,-2), (-2,-3)))[:, :, None]

            # vi) Update factor.
            U[n] = np.linalg.solve(lhs_stack, rhs_stack).reshape(X.shape[n], rank)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        obj = linalg.norm(mask * (U.full() - X)) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()


def mcp_als_array_api_sparse(
    X,
    X_sp,
    rank,
    mask,
    mask_sp,
    random_state=None,
    init='randn',
    skip_modes=[],
    **options,
):
    xp = sparse

    # Check inputs.
    optim_utils._check_cpd_inputs(X, rank)

    # Initialize problem.
    U, _ = optim_utils._get_initial_ktensor(init, X, rank, random_state, scale_norm=False)
    result = FitResult(U, 'MCP_ALS', **options)
    normX = np.linalg.norm((X * mask))

    # Main optimization loop.
    while result.still_optimizing:

        # Iterate over each tensor mode.
        for n in range(X.ndim):

            # Skip modes that are specified as fixed.
            if n in skip_modes:
                continue

            # i) Normalize factors to prevent singularities.
            U.rebalance()

            unf = xp.moveaxis(X_sp, n, 0)
            m = xp.moveaxis(mask_sp, n, 0)

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            components = [xp.lazy(xp.asarray(c, format="dense")) for c in components]
            kr = components[0][:, None, :] * components[1][None, :, :]
            krt = xp.moveaxis(kr, -1, 0)

            # iv) Broadcasted solve of linear systems.
            lhs_stack = xp.sum(
                m[:, None, ..., None] * krt[None, ..., None] * kr[None, None, ...], axis=(-2, -3)
            )
            rhs_stack = xp.tensordot(unf, kr, axes=((-1,-2), (-2,-3)))

            lhs_stack = xp.compute(lhs_stack).todense()
            rhs_stack = xp.compute(rhs_stack).todense()[:, :, None]

            # vi) Update factor.
            U[n] = np.linalg.solve(lhs_stack, rhs_stack).reshape(X.shape[n], rank)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        obj = linalg.norm(mask * (U.full() - X)) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()
