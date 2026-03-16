from utils import append_test_to_excel, load_audio, cut_audio, align_to_ref, run_algos, save_audio
from dereverb import dereverb_wpe
from metrics import compute_all_metrics
from pathlib import Path
from functools import partial

if __name__ == "__main__":
    ref_path = Path(r"C:\Users\User\Desktop\sound\wav\nr\gezer speech ref.wav")
    deg_path = Path(r"C:\Users\User\Desktop\sound\wav\nr\gezer speech transfer.wav")

    ref, sr = load_audio(ref_path)
    deg, _ = cut_audio(*load_audio(deg_path), t_start=25)
    aligned_deg, delay = align_to_ref(ref, deg)

    # correlated file params
    N_start, N_end = 13, 33
    S_start, S_end = 41, 90

    N_ref, _ = cut_audio(ref, sr, t_start=N_start, t_end=N_end)
    N_deg, _ = cut_audio(aligned_deg, sr, t_start=N_start, t_end=N_end)

    S_ref, _ = cut_audio(ref, sr, t_start=S_start, t_end=S_end)
    S_deg, _ = cut_audio(aligned_deg, sr, t_start=S_start, t_end=S_end)

    # run
    tapss = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    for taps in tapss:
        name, result = run_algos(S_deg, sr,
                                 [
                                     partial(dereverb_wpe, sr=sr, delay=5, taps=taps)
                                 ]
                                 )

        metrics = compute_all_metrics(S_ref, result, sr)
        append_test_to_excel(name, metrics)