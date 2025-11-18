"""Minimal Python version of MATLAB myMeasurementLikelihoodFcn.
Supports x_pred shape (4,) or (4, N). Returns scalar or ndarray of likelihoods.
Simplified checks; only uses numpy. Interpolation is linear.
"""

from __future__ import annotations

import numpy as np


def _rotate_unit_x_batch(qs):  # type: ignore
	"""Rotate unit X axis by each quaternion (qs shape (4,) or (4,N))."""
	qs = np.asarray(qs, dtype=float)
	if qs.ndim == 1:
		qs = qs.reshape(4, 1)
	# normalize
	norms = np.linalg.norm(qs, axis=0, keepdims=True)
	norms[norms == 0] = 1.0
	qs = qs / norms
	w = qs[0]; x = qs[1]; y = qs[2]; z = qs[3]
	# First column of rotation matrix (R * [1,0,0]^T)
	vx = 1 - 2*(y*y + z*z)
	vy = 2*(x*y + w*z)
	vz = 2*(x*z - w*y)
	out = np.vstack((vx, vy, vz))
	return out if out.shape[1] > 1 else out[:, 0]


def myMeasurementLikelihoodFcn(
	x_pred,
	y,
	fx,
	fy,
	T_ce,
	n_samples=10000,
	sigma=15.0,
	bin_width=0.05,
):  # type: ignore
	# Parse inputs
	y = np.asarray(y, dtype=float).reshape(-1)
	# Minimal shape assumption

	pt0 = y[0:2]
	pt1 = y[3:5]
	d = pt1 - pt0
	du, dv = d[0], d[1]

	# Sample noisy slopes (du+N)/(dv+N)
	rng = np.random.default_rng()
	Gx = sigma * rng.standard_normal(n_samples)
	Gy = sigma * rng.standard_normal(n_samples)
	denom = dv + Gy
	# Avoid division by exact zero by nudging extremely small denominators
	eps = 1e-12
	denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
	samples = (du + Gx) / denom

	# Compute predicted slope from orientation
	q = np.asarray(x_pred, dtype=float)
	# Conjugate for direction reversal
	if q.ndim == 1:
		q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)
	else:
		q_conj = np.vstack((q[0], -q[1], -q[2], -q[3]))
	l_w = _rotate_unit_x_batch(q_conj)
	# Remap axes
	if l_w.ndim == 1:
		l_w_correct = np.array([-l_w[2], -l_w[1], -l_w[0]], dtype=np.float64)
	else:
		l_w_correct = np.vstack((-l_w[2], -l_w[1], -l_w[0]))

	# camera frame
	T_ce = np.asarray(T_ce, dtype=float)
	if T_ce.shape != (3, 3):
		raise ValueError("T_ce must be a 3x3 matrix.")
	l_c = T_ce @ l_w_correct  # (3,) or (3,N)

	# project onto pixel frame and compute predicted slope
	du_pred = fx * l_c[0]
	dv_pred = fy * l_c[1]
	if isinstance(dv_pred, np.ndarray):
		dv_pred = np.where(np.abs(dv_pred) < eps, np.sign(dv_pred) * eps + (dv_pred == 0)*eps, dv_pred)
	else:
		if abs(dv_pred) < eps:
			dv_pred = eps if dv_pred == 0 else np.sign(dv_pred) * eps
	x_predicted = du_pred / dv_pred

	# Histogram-based likelihood P(samples at x_predicted)
	bw = float(bin_width)
	bw_half = bw / 2.0
	s_min, s_max = samples.min(), samples.max()
	if not np.isfinite(s_min) or not np.isfinite(s_max) or s_min == s_max:
		# Degenerate sampling; return near-zero likelihood
		return 0.0

	# Create edges similar to MATLAB's 'BinWidth'
	# Ensure at least two edges
	edges = np.arange(s_min, s_max + bw, bw)
	if edges.size < 2:
		edges = np.array([s_min - bw, s_min + bw])

	counts, edges = np.histogram(samples, bins=edges)
	bin_centers = edges[:-1] + bw_half
	density = counts.astype(float) / float(n_samples)

	# Linear interpolation (simple, low-complexity replacement for 'spline')
	likelihood = np.interp(x_predicted, bin_centers, density, left=0.0, right=0.0)
	return np.maximum(0.0, likelihood)


if __name__ == "__main__":
	# Minimal self-test to ensure the function executes
	# Dummy inputs
	q = np.array([1.0, 0.0, 0.0, 0.0])  # single quaternion
	qs = np.array([[1.0, 0.0, 0.0],  # w row after reshape later -> use shape (4,N)
				   [0.0, 0.0, 0.0],
				   [0.0, 0.1, 0.2],
				   [0.0, 0.0, 0.0]])  # batch (w,x,y,z) columns
	y = np.array([100.0, 200.0, 0.9, 120.0, 230.0])
	fx, fy = 1000.0, 1000.0
	T = np.eye(3)

	val_single = myMeasurementLikelihoodFcn(q, y, fx, fy, T)
	val_batch = myMeasurementLikelihoodFcn(qs, y, fx, fy, T)
	print("single:", val_single)
	print("batch:", val_batch)

