import numpy as np
from utils import compute_stft, compute_istft, return_params

@return_params
import numpy as np

def dereverb_wpe(y, sr, delay=3, taps=10, iters=3):
    f, t, Y = compute_stft(y, sr)
    F, T = Y.shape

    X = Y.copy()
    lam = np.abs(Y)**2 + 1e-8

    for _ in range(iters):

        for k in range(F):

            Yk = Y[k]
            Xk = X[k]
            lam_k = lam[k]

            if T < delay + taps:
                continue

            rows = T - delay - taps
            Ymat = np.zeros((taps, rows), dtype=complex)

            for i in range(taps):
                Ymat[i] = Yk[delay + taps - i - 1 : T - i - 1]

            yvec = Yk[delay + taps : T]

            w = 1.0 / lam_k[delay + taps : T]

            R = (Ymat * w) @ Ymat.conj().T
            p = (Ymat * w) @ yvec.conj()

            g = np.linalg.solve(R + 1e-6*np.eye(taps), p)

            for m in range(delay + taps, T):
                pred = np.dot(g.conj(), Yk[m-delay-taps:m-delay])
                X[k, m] = Yk[m] - pred

        lam = np.abs(X)**2 + 1e-8

    _, x = compute_istft(X, sr)
    return x[:len(y)]