import numpy as np
from scipy.special import exp1
from utils import compute_stft, compute_istft, return_params

@return_params
def logmmse_enhance(y, sr, noise_frames=20):
    f, t, Y = compute_stft(y, fs=sr)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)

    noise_psd = np.mean(Y_mag[:, :noise_frames]**2, axis=1, keepdims=True)
    enhanced = np.zeros_like(Y_mag)
    prev_gain = np.ones_like(noise_psd)
    alpha = 0.98

    for m in range(Y_mag.shape[1]):
        gamma = (Y_mag[:, m:m+1]**2) / noise_psd
        xi = alpha * (prev_gain**2) * gamma + (1-alpha) * np.maximum(gamma-1, 0)
        v = gamma * xi / (1 + xi)
        gain = (xi/(1+xi)) * np.exp(0.5 * exp1(v))
        enhanced[:, m:m+1] = gain * Y_mag[:, m:m+1]
        prev_gain = gain

    S = enhanced * np.exp(1j * Y_phase)
    _, x = compute_istft(S, sr)
    return x