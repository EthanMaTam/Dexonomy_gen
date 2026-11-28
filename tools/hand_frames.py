"""Display transforms from native hand coordinates to the Shadow palm frame.

Shadow: fingers +Z, index side +X, palm outward -Y.
O30/ZTE: fingers +Z, index side +Y, palm outward +X.
These transforms affect display only; model and annotation coordinates stay native.
"""

import numpy as np

NATIVE_TO_SHADOW = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
SHADOW_TO_NATIVE = NATIVE_TO_SHADOW.T


def display_rotation(hand: str) -> np.ndarray:
    return NATIVE_TO_SHADOW.copy() if hand in {
        "linker_o30_right", "zte_hand", "sharpa_left"
    } else np.eye(3)
