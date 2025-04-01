"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu> and Alex H. Williams
"""

import numpy as np
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
            unf = np.moveaxis(X, n, 0).reshape((X.shape[n], -1))  # i_n x N
            m = np.moveaxis(mask, n, 0).reshape((mask.shape[n], -1))  # i_n x N

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            #krt = khatri_rao(components).T  # N x r
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

            unf = np.moveaxis(X, n, 0)
            m = np.moveaxis(mask, n, 0)

            # iii) Form Khatri-Rao product of factors matrices.
            components = [U[j] for j in range(X.ndim) if j != n]
            kr = components[0][:, None, :] * components[1][None, :, :]
            krt = np.moveaxis(kr, -1, 0)

            # iv) Broadcasted solve of linear systems.
            lhs_stack = np.sum(
                m[:, None, ..., None] * krt[None, ..., None] * kr[None, None, ...], axis=(-2, -3)
            )
            rhs_stack = np.tensordot(unf * m, kr, axes=((-1,-2), (-2,-3)))[:, :, None]

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


import sparse
xp = sparse


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


##########################
#### FINCH - OUTDATED ####
##########################


def _get_available_idx(ndim: int, matricized_idx: int, reverse: bool) -> int:
    indices = reversed(range(ndim)) if reverse else range(ndim)
    for idx in indices:
        if idx == matricized_idx:
            continue
        return idx


def _get_component_slice(ndim: int, matricized_idx: int, reverse: bool) -> tuple:
    av_idx = _get_available_idx(ndim, matricized_idx, reverse)
    cs = [None for _ in range(ndim)]
    cs[av_idx] = slice(None)
    return tuple(cs)


@sparse.compiled()
def lhs_stack_kernel(mask, component1, component2, n, ndim):
    krt = xp.permute_dims(component1[None, :, :] * component2[:, None, :], (2, 1, 0))
    krt2 = xp.permute_dims(krt, (1, 2, 0))
    moveax = (n, *[i for i in range(ndim) if i != n])
    new_mask = xp.permute_dims(mask, moveax)
    return xp.sum(
        new_mask[:, None, :, :, None] * krt[None, :, :, :, None] * krt2[None, None, :, :, :],
        axis=(2, 3)
    )


@sparse.compiled()
def rhs_stack_kernel(X, component1, component2, n):
    ndim = X.ndim
    axis = tuple(i for i in range(ndim) if i != n)
    return xp.sum(
        (
            X[:, :, :, None] *
            component1[_get_component_slice(ndim, n, reverse=False) + (slice(None),)] *
            component2[_get_component_slice(ndim, n, reverse=True) + (slice(None),)]
        ),
        axis=axis,
    )


def mcp_als_sparse_outdated(
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

            # ii) Form Khatri-Rao product of factors matrices.
            # iii) Broadcasted solve of linear systems.
            components = [U[j] for j in range(X.ndim) if j != n]
            components_sp = [sparse.asarray(c, format="dense") for c in components]

            lhs_stack_shadow = lhs_stack_kernel(mask_sp, components_sp[0], components_sp[1], n, X.ndim).todense()
            rhs_stack_shadow = rhs_stack_kernel(X_sp, components_sp[0], components_sp[1], n)[:, :, None].todense()

            # iv) Update factor.
            U[n] = np.linalg.solve(lhs_stack_shadow, rhs_stack_shadow).reshape(X.shape[n], rank)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update the optimization result, checks for convergence.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        obj = linalg.norm(mask * (U.full() - X)) / normX

        # Update result
        result.update(obj)

    # Finalize and return the optimization result.
    return result.finalize()
