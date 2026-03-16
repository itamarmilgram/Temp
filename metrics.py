import numpy as np
from pystoi import stoi
from pesq import pesq
from mir_eval.separation import bss_eval_sources
from utils import resample_audio, align_length


def si_snr(reference, estimation):
    reference, estimation = align_length(reference, estimation)
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    ref_energy = np.sum(reference ** 2)
    projection = np.sum(reference * estimation) * reference / ref_energy
    noise = estimation - projection
    ratio = np.sum(projection ** 2) / np.sum(noise ** 2)
    return 10 * np.log10(ratio)

def compute_stoi(reference, estimation, sr):
    reference, estimation = align_length(reference, estimation)
    return stoi(reference, estimation, sr, extended=False)

def compute_pesq(reference, estimation, sr):
    reference, estimation = align_length(reference, estimation)
    if sr not in [8000, 16000]:
        reference = resample_audio(reference, sr, 16000)
        estimation = resample_audio(estimation, sr, 16000)
        sr = 16000
    mode = "wb" if sr == 16000 else "nb"
    try:
        return pesq(sr, reference, estimation, mode)
    except:
        return np.nan

def compute_sdr(reference, estimation):
    reference, estimation = align_length(reference, estimation)
    sdr, _, _, _ = bss_eval_sources(
        reference[np.newaxis, :],
        estimation[np.newaxis, :]
    )
    return float(sdr[0])

def compute_all_metrics(ref, deg, sr):
    metrics = {}
    metrics["SI_SNR"] = si_snr(ref, deg)
    metrics["STOI"] = compute_stoi(ref, deg, sr)
    metrics["PESQ"] = compute_pesq(ref, deg, sr)
    metrics["SDR"] = compute_sdr(ref, deg)
    return metrics