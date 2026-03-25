from __future__ import annotations

from typing import Union
from utilspy.transFromQuat import transFromQuat
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os


def _rotate_vector_batch(qs, v=np.array([1.0, 0.0, 0.0])):
	"""Rotate vector v by each quaternion (qs shape (4,) or (4,N))."""
	qs = np.asarray(qs, dtype=float)
	v = np.asarray(v, dtype=float).flatten()
	if qs.ndim == 1:
		qs = qs.reshape(4, 1)
	# normalize
	norms = np.linalg.norm(qs, axis=0, keepdims=True)
	norms[norms == 0] = 1.0
	qs = qs / norms
	w = qs[0]; x = qs[1]; y = qs[2]; z = qs[3]
	
	# v' = v + 2 * r x (r x v + w * v)
	vx, vy, vz = v[0], v[1], v[2]
	
	# r x v
	rxv_x = y*vz - z*vy
	rxv_y = z*vx - x*vz
	rxv_z = x*vy - y*vx
	
	# term = r x v + w * v
	term_x = rxv_x + w*vx
	term_y = rxv_y + w*vy
	term_z = rxv_z + w*vz
	
	# r x term
	rxterm_x = y*term_z - z*term_y
	rxterm_y = z*term_x - x*term_z
	rxterm_z = x*term_y - y*term_x
	
	out_x = vx + 2*rxterm_x
	out_y = vy + 2*rxterm_y
	out_z = vz + 2*rxterm_z
	
	out = np.vstack((out_x, out_y, out_z))
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
)-> Union[np.ndarray, float]:

	y = np.asarray(y, dtype=float).reshape(-1)

	pt0 = y[0:2]
	pt1 = y[3:5]
	d = pt1 - pt0
	du, dv = d[0], d[1]

	rng = np.random.default_rng(8)
	Gx = sigma * rng.standard_normal(n_samples)
	Gy = sigma * rng.standard_normal(n_samples)
	denom = dv + Gy

	eps = 1e-12
	denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)
	samples: np.ndarray = (du + Gx) / denom

	q = np.asarray(x_pred, dtype=float)

	l_w = _rotate_vector_batch(q, np.array([-1.0, 0.0, 0.0]))

	# camera frame
	T_ce = np.asarray(T_ce, dtype=float)
	if T_ce.shape != (3, 3):
		raise ValueError("T_ce must be a 3x3 matrix.")
	l_c = T_ce @ l_w  # (3,) or (3,N)

	# project onto pixel frame and compute predicted slope
	du_pred = fx * l_c[0]
	dv_pred = fy * l_c[1]
	if isinstance(dv_pred, np.ndarray):
		dv_pred = np.where(np.abs(dv_pred) < eps, np.sign(dv_pred) * eps + (dv_pred == 0)*eps, dv_pred)
	else:
		if abs(dv_pred) < eps:
			dv_pred = eps if dv_pred == 0 else np.sign(dv_pred) * eps
	x_predicted: Union[np.ndarray, float] = du_pred / dv_pred

	# Histogram-based likelihood P(samples at x_predicted)
	bw = float(bin_width)
	s_min, s_max = float(samples.min()), float(samples.max())
	if not np.isfinite(s_min) or not np.isfinite(s_max) or s_min == s_max:
		return 0.0

	max_bins = 65536
	span = float(s_max - s_min)
	bw_eff: float = float(bw)
	if span / bw_eff > max_bins:
		bw_eff = float(span / max_bins)
	
	bw_half = bw_eff / 2.0

	edges = np.arange(s_min, s_max + bw_eff, bw_eff, dtype=float)
	if edges.size < 2:
		edges = np.array([s_min - bw_eff, s_min + bw_eff])

	counts: np.ndarray
	edges: np.ndarray
	counts, edges = np.histogram(samples, bins=edges)
	bin_centers: np.ndarray = edges[:-1] + bw_half
	density: np.ndarray = counts.astype(float) / float(n_samples)

	# # Cubic spline (MATLAB 'spline' equivalent); allow extrapolation then clamp to >=0
	likelihood = CubicSpline(bin_centers, density, extrapolate=True)(x_predicted)

	
	# Linear interpolation over histogram bin centers; extrapolate with edge densities
	# if np.isscalar(x_predicted):
	# 	likelihood = np.interp(x_predicted, bin_centers, density, left=density[0], right=density[-1])
	# else:
	# 	likelihood = np.interp(x_predicted, bin_centers, density, left=density[0], right=density[-1])
	
	# # Visualization: Save comparison of samples and x_predicted distributions
	# # Use a function attribute to count calls
	# if not hasattr(myMeasurementLikelihoodFcn, "call_count"):
	# 	myMeasurementLikelihoodFcn.call_count = 0
	
	# # Save plot for the first few calls
	# if myMeasurementLikelihoodFcn.call_count < 2: 
	# 	try:
	# 		plt.figure(figsize=(10, 6))
			
	# 		# Plot samples (Measurement distribution)
	# 		plt.hist(samples, bins=edges, density=True, alpha=0.5, color='green', label='observations (samples)')
			
	# 		# Plot x_predicted (Particle predictions)
	# 		if np.isscalar(x_predicted):
	# 			plt.axvline(x=x_predicted, color='blue', linestyle='--', linewidth=2, label='Predicted (Scalar)')
	# 		else:
	# 			# Filter out extreme values for plotting if necessary
	# 			x_pred_valid = x_predicted[np.isfinite(x_predicted)]
	# 			# Clip to reasonable range for visualization if needed, or just plot
	# 			# To avoid outliers compressing the view, we can limit to samples range +/- margin
	# 			s_range = s_max - s_min
	# 			view_min = s_min - 0.5 * s_range
	# 			view_max = s_max + 0.5 * s_range
				
	# 			# Plot histogram of predictions
	# 			plt.hist(x_pred_valid, bins=50, density=True, alpha=0.5, color='blue', label='Predicted (Particles)')
	# 			plt.xlim(view_min, view_max)
			
	# 		plt.title(f'Likelihood vs Prediction (Call {myMeasurementLikelihoodFcn.call_count})')
	# 		plt.xlabel('Slope value')
	# 		plt.ylabel('Density')
	# 		plt.legend()
	# 		plt.grid(True, alpha=0.3)
			
	# 		# Save to 'debug_plots' directory
	# 		save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_plots')
	# 		os.makedirs(save_dir, exist_ok=True)
	# 		save_path = os.path.join(save_dir, f'likelihood_dist_{myMeasurementLikelihoodFcn.call_count}.png')
	# 		plt.savefig(save_path)
	# 		plt.close()
	# 		print(f"Saved debug plot to {save_path}")
	# 	except Exception as e:
	# 		print(f"Error plotting likelihood: {e}")

	# myMeasurementLikelihoodFcn.call_count += 1

	return np.maximum(0.0, likelihood)


if __name__ == "__main__":
	# Minimal self-test to ensure the function executes
	# Dummy inputs
	# q = np.array([1.0, 0.0, 0.0, 0.0])  # single quaternion
	# qs = np.array([[1.0, 0.0, 0.0],  # w row after reshape later -> use shape (4,N)
	# 			   [0.0, 0.0, 0.0],
	# 			   [0.0, 0.1, 0.2],
	# 			   [0.0, 0.0, 0.0]])  # batch (w,x,y,z) columns
	# y = np.array([100.0, 200.0, 0.9, 120.0, 230.0])
	# fx, fy = 1000.0, 1000.0
	# T = np.eye(3)

	# val_single = myMeasurementLikelihoodFcn(q, y, fx, fy, T)
	# val_batch = myMeasurementLikelihoodFcn(qs, y, fx, fy, T)
	# print("single:", val_single)
	# print("batch:", val_batch)

	# q_rand = np.random.rand(4)
	# q_rand /= np.linalg.norm(q_rand)
	q = _rotate_vector_batch(np.array([1.0, 0.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
	# print(q_rand)
	print("rotate single:", q)

