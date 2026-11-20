"""Replicate the warm start's psi fit (fit_partitions) and score it at every frame time."""
import sys, time, json
import numpy as np, torch
from motion_static.models.pou_net import POUNet
from motion_static.inverse.warm_start import fit_partitions, hat_functions

def score(t0, t1, frames, partitions, seed=0, device="cpu"):
    torch.manual_seed(seed)
    m = POUNet(248, [[0, 37.2], [0, 37.2], [t0, t1]], 3.6, partitions=partitions).to(device)
    start = time.perf_counter()
    nodes, loss = fit_partitions(m)
    t = torch.linspace(t0, t1, frames, device=device)
    with torch.no_grad():
        psi = m.partition(t); target = hat_functions(m.normalized_time(t), nodes)
    err = (psi - target).cpu().numpy(); psi = psi.cpu().numpy(); target = target.cpu().numpy()
    return dict(range_units=[float(m.normalized_time(torch.tensor(t0))), float(m.normalized_time(torch.tensor(t1)))],
                fit_loss=loss, frame_rms_error=float(np.sqrt((err ** 2).mean())), target_rms=float(np.sqrt((target ** 2).mean())),
                frame_change_psi=float(np.abs(np.diff(psi, axis=0)).sum(1).mean()),
                frame_change_target=float(np.abs(np.diff(target, axis=0)).sum(1).mean()),
                seconds=round(time.perf_counter() - start, 1)), psi, target

if __name__ == "__main__":
    out = sys.argv[1]
    full, psi, target = score(-36.0, 684.0, 14401, 20)
    print("full series (20 partitions):", json.dumps(full))
    window, psi_w, target_w = score(26.5, 34.0, 151, 10)
    print("window (10 partitions):", json.dumps(window))
    json.dump(dict(full_series_20_partitions=full, window_10_partitions=window), open(out + "/hat_fit.json", "w"), indent=1)
