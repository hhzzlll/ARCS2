import numpy as np
import math
try:
	from numba import njit
except Exception:
	def njit(*args, **kwargs):
		def _wrap(func):
			return func
		return _wrap
from filterpy.monte_carlo import multinomial_resample
from tqdm.auto import tqdm

from myStateTransitionFcn import myStateTransitionFcn  # type: ignore
from myMeasurementLikelihoodFcn import myMeasurementLikelihoodFcn  # type: ignore


@njit(fastmath=True)
def _normalize_quaternions(q):  # type: ignore
	# q shape expected (4, N); numba-friendly column-wise normalization
	N = q.shape[1]
	for j in range(N):
		n = math.sqrt(q[0,j]*q[0,j] + q[1,j]*q[1,j] + q[2,j]*q[2,j] + q[3,j]*q[3,j])
		if n < 1e-12:
			n = 1e-12
		q[0,j] /= n; q[1,j] /= n; q[2,j] /= n; q[3,j] /= n


# resampling will use filterpy's multinomial_resample in-place at call sites


@njit(fastmath=True)
def _quat_mean(weights, particles):  # type: ignore
	# particles shape (4, N), weights shape (N,) — numba-friendly reduction
	est0 = 0.0; est1 = 0.0; est2 = 0.0; est3 = 0.0
	N = particles.shape[1]
	for j in range(N):
		w = weights[j]
		est0 += w * particles[0,j]
		est1 += w * particles[1,j]
		est2 += w * particles[2,j]
		est3 += w * particles[3,j]
	n = math.sqrt(est0*est0 + est1*est1 + est2*est2 + est3*est3)
	if n > 0.0:
		est0 /= n; est1 /= n; est2 /= n; est3 /= n
	return np.array([est0, est1, est2, est3], dtype=np.float64)


def estimatePose(q0,
				 N,
				 w_sw,
				 keypoints,
				 fx,
				 fy,
				 T_cw,
				 stateTransitionFcn,
				 measurementLikelihoodFcn,
				 noFilter=False):  # type: ignore
	# initialize particles around q0 with gaussian noise, then normalize (use shape 4xN)
	particles = np.tile(q0.reshape(4,1), (1,N)) + 0.1*np.random.randn(4,N)
	_normalize_quaternions(particles)
	weights = np.ones(N)/N

	sz_w = w_sw.shape[0]
	qEst = np.empty((4, sz_w), dtype=np.float64)
	qEst[:,0] = q0

	if noFilter:
		dt_w = w_sw[1,0] - w_sw[0,0]
		for k in range(1, sz_w):
			qEst[:,k] = stateTransitionFcn(qEst[:,k-1], w_sw[k,1:], dt_w, 1)
		return qEst

	dt_w = w_sw[1,0] - w_sw[0,0]
	idx_kpts = 0

	for idx_w in tqdm(range(sz_w), desc="Estimating", leave=True):
		t = w_sw[idx_w,0]
		has_meas = (idx_kpts < keypoints.shape[0]) and ((t - keypoints[idx_kpts,0]) > 0.0)

		if has_meas:
			krow = keypoints[idx_kpts,1:]
			valid = np.all(krow != 0.0)

			if valid:
				# correct then predict (mirror MATLAB order)
				L = measurementLikelihoodFcn(particles, krow, fx, fy, T_cw)
				weights *= L
				s = np.sum(weights)
				if s > 0.0:
					weights /= s
				else:
					weights[:] = 1.0 / N
				denom = np.sum(weights*weights) + 1e-12
				Neff = 1.0 / denom
				if Neff < 0.5 * N:
					idx = multinomial_resample(weights)
					particles = particles[:, idx]
					weights[:] = 1.0 / N
				qEst[:,idx_w] = _quat_mean(weights, particles)
				w_vec = w_sw[idx_w,1:]
				particles = stateTransitionFcn(particles, w_vec, dt_w, N)
				_normalize_quaternions(particles)
				idx_kpts += 1
			else:
				# invalid measurement: predict and consume this measurement
				w_vec = w_sw[idx_w,1:]
				particles = stateTransitionFcn(particles, w_vec, dt_w, N)
				_normalize_quaternions(particles)
				qEst[:,idx_w] = _quat_mean(weights, particles)
				idx_kpts += 1
		else:
			# no measurement: predict only
			w_vec = w_sw[idx_w,1:]
			particles = stateTransitionFcn(particles, w_vec, dt_w, N)
			_normalize_quaternions(particles)
			qEst[:,idx_w] = _quat_mean(weights, particles)

	return qEst


if __name__ == "__main__":
	# minimal smoke test with synthetic data
	N = 3000
	q0 = np.array([1.0,0.0,0.0,0.0])
	T = 15400
	t_w = np.linspace(0, 1, T)
	w_sw = np.column_stack((t_w, 0.1*np.ones(T), np.zeros(T), np.zeros(T)))
	key_t = t_w[::5]
	keypoints = np.column_stack((key_t, 100+5*np.ones(len(key_t)), 200+5*np.ones(len(key_t)), np.ones(len(key_t)), 120+5*np.ones(len(key_t)), 230+5*np.ones(len(key_t)), np.ones(len(key_t))))
	fx = 1000.0; fy = 1000.0; T_cw = np.eye(3)
	qEst = estimatePose(q0, N, w_sw, keypoints, fx, fy, T_cw, myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter=False)
	print("final quaternion:", qEst[:, -1])
