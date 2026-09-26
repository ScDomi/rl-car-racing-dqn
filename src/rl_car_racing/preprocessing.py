"""Frame preprocessing for Gymnasium CarRacing observations."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - fallback for minimal installs
    cv2 = None


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """Convert a RGB CarRacing frame into a normalized grayscale image.

    Parameters
    ----------
    frame:
        Raw RGB frame from Gymnasium CarRacing, usually shape ``(96, 96, 3)``.
    size:
        Output width/height. The DQN architecture expects ``84``.
    """

    if cv2 is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    else:
        # Lightweight fallback: luma conversion + nearest-neighbour resize.
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        y_idx = np.linspace(0, gray.shape[0] - 1, size).astype(int)
        x_idx = np.linspace(0, gray.shape[1] - 1, size).astype(int)
        resized = gray[np.ix_(y_idx, x_idx)]

    return resized.astype(np.float32) / 255.0
