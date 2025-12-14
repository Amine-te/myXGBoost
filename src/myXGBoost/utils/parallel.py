"""Parallel computation utilities for gradient and histogram operations."""

import numpy as np
from typing import Tuple, Optional, Callable
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def _compute_grad_hess_chunk(
    loss_function: Callable,
    y_true_chunk: np.ndarray,
    y_pred_chunk: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients and hessians for a data chunk.
    
    Helper function for parallel gradient computation.
    
    Parameters
    ----------
    loss_function : LossFunction
        Loss function instance.
    y_true_chunk : ndarray
        True target values for this chunk.
    y_pred_chunk : ndarray
        Predicted values for this chunk.
        
    Returns
    -------
    grad_chunk : ndarray
        Gradients for this chunk.
    hess_chunk : ndarray
        Hessians for this chunk.
    """
    return loss_function.grad_hess(y_true_chunk, y_pred_chunk)


def compute_gradients_parallel(
    loss_function: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_jobs: int = -1,
    chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradients and hessians in parallel across data chunks.
    
    Splits the data into chunks and computes gradients/hessians for each
    chunk in parallel, then concatenates the results.
    
    Parameters
    ----------
    loss_function : LossFunction
        Loss function instance.
    y_true : ndarray
        True target values.
    y_pred : ndarray
        Predicted values.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all cores.
    chunk_size : int, optional
        Size of each chunk. If None, automatically determined.
        
    Returns
    -------
    grad : ndarray
        Gradients for all samples.
    hess : ndarray
        Hessians for all samples.
    """
    n_samples = len(y_true)
    
    # Determine chunk size
    if chunk_size is None:
        # Use a reasonable default: aim for ~4-8 chunks per core
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_chunks = max(1, min(n_jobs * 4, n_samples // 1000))
        chunk_size = max(100, n_samples // n_chunks) if n_chunks > 0 else n_samples
    
    # If dataset is small, don't parallelize
    if n_samples < chunk_size * 2:
        return loss_function.grad_hess(y_true, y_pred)
    
    # Split into chunks
    chunks = []
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunks.append((i, end))
    
    # Compute gradients in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_grad_hess_chunk)(
            loss_function,
            y_true[start:end],
            y_pred[start:end] if y_pred.ndim == 1 else y_pred[start:end, :]
        )
        for start, end in chunks
    )
    
    # Concatenate results
    grad_chunks, hess_chunks = zip(*results)
    grad = np.concatenate(grad_chunks, axis=0)
    hess = np.concatenate(hess_chunks, axis=0)
    
    return grad, hess


def _build_histogram_chunk(
    feature_values_chunk: np.ndarray,
    grad_chunk: np.ndarray,
    hess_chunk: np.ndarray,
    bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build histogram statistics for a data chunk.
    
    Helper function for parallel histogram building.
    
    Parameters
    ----------
    feature_values_chunk : ndarray
        Feature values for this chunk.
    grad_chunk : ndarray
        Gradient values for this chunk.
    hess_chunk : ndarray
        Hessian values for this chunk.
    bins : ndarray
        Histogram bin boundaries.
        
    Returns
    -------
    g_hist_chunk : ndarray
        Gradient histogram for this chunk.
    h_hist_chunk : ndarray
        Hessian histogram for this chunk.
    """
    # Filter missing values
    mask_non_missing = ~np.isnan(feature_values_chunk)
    if not np.any(mask_non_missing):
        # Return zero histograms
        minlength = len(bins) + 1
        return np.zeros(minlength, dtype=np.float64), np.zeros(minlength, dtype=np.float64)
    
    f = feature_values_chunk[mask_non_missing]
    g = grad_chunk[mask_non_missing]
    h = hess_chunk[mask_non_missing]
    
    # Map values to bin indices
    indices = np.digitize(f, bins)
    
    # Calculate histogram stats using bincount
    minlength = len(bins) + 1
    g_hist = np.bincount(indices, weights=g, minlength=minlength)
    h_hist = np.bincount(indices, weights=h, minlength=minlength)
    
    return g_hist, h_hist


def build_histogram_parallel(
    feature_values: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    bins: np.ndarray,
    n_jobs: int = -1,
    chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build histogram statistics in parallel across data chunks.
    
    Splits the data into chunks, builds histograms for each chunk in parallel,
    then merges (reduces) the histograms.
    
    Parameters
    ----------
    feature_values : ndarray
        Feature values for all samples.
    grad : ndarray
        Gradient values for all samples.
    hess : ndarray
        Hessian values for all samples.
    bins : ndarray
        Histogram bin boundaries.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all cores.
    chunk_size : int, optional
        Size of each chunk. If None, automatically determined.
        
    Returns
    -------
    g_hist : ndarray
        Merged gradient histogram.
    h_hist : ndarray
        Merged hessian histogram.
    """
    n_samples = len(feature_values)
    
    # Determine chunk size
    if chunk_size is None:
        # Use a reasonable default: aim for ~4-8 chunks per core
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_chunks = max(1, min(n_jobs * 4, n_samples // 1000))
        chunk_size = max(100, n_samples // n_chunks) if n_chunks > 0 else n_samples
    
    # If dataset is small, don't parallelize
    if n_samples < chunk_size * 2:
        return _build_histogram_chunk(feature_values, grad, hess, bins)
    
    # Split into chunks
    chunks = []
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunks.append((i, end))
    
    # Build histograms in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_build_histogram_chunk)(
            feature_values[start:end],
            grad[start:end],
            hess[start:end],
            bins
        )
        for start, end in chunks
    )
    
    # Merge (reduce) histograms
    g_hist_chunks, h_hist_chunks = zip(*results)
    g_hist = np.sum(g_hist_chunks, axis=0)
    h_hist = np.sum(h_hist_chunks, axis=0)
    
    return g_hist, h_hist

