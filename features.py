import numpy as np

def normalize_sequence(seq):
    """
    seq: (30, 126) raw landmarks [left 63 | right 63].
    Returns (30, 132): per-hand normalized shape (126) + raw wrist positions (6).
    """
    seq = np.asarray(seq)
    if seq.ndim != 2 or seq.shape[1] != 126:
        raise ValueError(
            f"Expected sequence shape (frames, 126), got {seq.shape}."
        )

    out = []
    for frame in seq:
        feats, wrists = [], []
        for h in range(2):
            pts = frame[h * 63:(h + 1) * 63].reshape(21, 3)
            if not pts.any():                      # hand missing
                feats.append(np.zeros(63)); wrists.append(np.zeros(3)); continue
            wrist = pts[0].copy()
            rel = pts - wrist                      # position-invariant
            scale = np.linalg.norm(rel, axis=1).max() + 1e-6
            feats.append((rel / scale).flatten())  # size-invariant
            wrists.append(wrist)
        out.append(np.concatenate(feats + wrists))
    return np.array(out, dtype=np.float32)