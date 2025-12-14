"""Input validation utilities (following sklearn patterns)."""

import numpy as np


def check_array(
    array,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
):
    """
    Input validation on an array, list, sparse matrix or similar.
    
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : bool, default=False
        Whether sparse matrices are accepted.
    accept_large_sparse : bool, default=True
        Whether large sparse matrices are accepted.
    dtype : str, type, list of type or None, default="numeric"
        Data type of result. If None, the dtype of the input is preserved.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : bool, default=False
        Whether a forced copy will be triggered.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan in array.
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples.
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features.
    estimator : str or estimator instance or None, default=None
        If passed, include the name of the estimator in warning messages.
        
    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if isinstance(array, np.ndarray):
        # Already a numpy array
        array_converted = array
    elif hasattr(array, '__array__'):
        # Array-like object (e.g., pandas DataFrame)
        array_converted = np.asarray(array)
    else:
        # List or other sequence
        array_converted = np.asarray(array)
    
    # Ensure 2D
    if ensure_2d and array_converted.ndim == 1:
        array_converted = array_converted.reshape(-1, 1)
    
    # Check dimensions
    if ensure_2d and array_converted.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {array_converted.ndim}D array instead."
        )
    
    if not allow_nd and array_converted.ndim > 2:
        raise ValueError(
            f"Found array with dim {array_converted.ndim}. Expected <= 2."
        )
    
    # Check minimum samples
    if array_converted.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {array_converted.shape[0]} sample(s) (shape={array_converted.shape}) "
            f"while a minimum of {ensure_min_samples} is required."
        )
    
    # Check minimum features
    if ensure_2d and array_converted.shape[1] < ensure_min_features:
        raise ValueError(
            f"Found array with {array_converted.shape[1]} feature(s) (shape={array_converted.shape}) "
            f"while a minimum of {ensure_min_features} is required."
        )
    
    # Check for finite values
    if force_all_finite:
        if force_all_finite == 'allow-nan':
            if np.any(np.isinf(array_converted)):
                raise ValueError("Input contains infinity.")
        else:
            if not np.all(np.isfinite(array_converted)):
                raise ValueError(
                    "Input contains NaN, infinity or a value too large for "
                    f"dtype('{array_converted.dtype}')."
                )
    
    # Convert dtype if needed
    if dtype is not None and dtype != "numeric":
        if dtype == "numeric":
            # Try to convert to numeric
            try:
                array_converted = array_converted.astype(np.float64)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unable to convert array of dtype {array_converted.dtype} to numeric."
                )
        else:
            array_converted = array_converted.astype(dtype)
    
    # Copy if requested
    if copy:
        array_converted = array_converted.copy()
    
    return array_converted


def check_X_y(
    X,
    y,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    multi_output=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    y_numeric=False,
    estimator=None,
):
    """
    Input validation for standard estimators.
    
    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Target data.
    accept_sparse : bool, default=False
        Whether sparse matrices are accepted.
    accept_large_sparse : bool, default=True
        Whether large sparse matrices are accepted.
    dtype : str, type, list of type or None, default="numeric"
        Data type of result. If None, the dtype of the input is preserved.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
    copy : bool, default=False
        Whether a forced copy will be triggered.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan in array.
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.
    multi_output : bool, default=False
        Whether to allow 2D y (multiple outputs).
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples.
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type.
    estimator : str or estimator instance or None, default=None
        If passed, include the name of the estimator in warning messages.
        
    Returns
    -------
    X_converted : object
        The validated and converted X.
    y_converted : object
        The validated and converted y.
    """
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        allow_nd=allow_nd,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        estimator=estimator,
    )
    
    y = check_array(
        y,
        accept_sparse=False,
        dtype=None if not y_numeric else np.float64,
        force_all_finite=force_all_finite,
        ensure_2d=False,
        allow_nd=multi_output,
        ensure_min_samples=ensure_min_samples,
        estimator=estimator,
    )
    
    # Check that X and y have compatible shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"[{X.shape[0]}, {y.shape[0]}]"
        )
    
    return X, y
