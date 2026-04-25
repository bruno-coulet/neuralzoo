"""
- Projet: Classification d'images (CIFAR-10) avec API d'inference.
- Role: fallback heuristique quand aucun modele TensorFlow n'est charge.
- Principe: calcule des logits deterministes a partir des moyennes RGB.
- Limite: composant de secours, non destine a remplacer un modele entraine.
"""

from __future__ import annotations

import numpy as np


def predict_cifar10_heuristic_logits(image_batch: np.ndarray) -> np.ndarray:
    """Retourne des logits heuristiques CIFAR-10 bases sur les canaux RGB.

    Cette fonction sert uniquement de fallback lorsque le modele entraine
    n'est pas disponible.
    """
    sample = image_batch[0]
    r_mean = float(sample[..., 0].mean())
    g_mean = float(sample[..., 1].mean())
    b_mean = float(sample[..., 2].mean())

    logits = np.array(
        [
            b_mean * 1.4,  # airplane
            r_mean * 1.2,  # automobile
            g_mean * 1.1,  # bird
            (r_mean + b_mean) * 0.9,  # cat
            g_mean * 1.0,  # deer
            (r_mean + g_mean) * 1.0,  # dog
            g_mean * 1.2,  # frog
            (r_mean + g_mean) * 1.1,  # horse
            b_mean * 1.6,  # ship
            r_mean * 1.3,  # truck
        ],
        dtype=np.float32,
    )
    logits = np.expand_dims(logits, axis=0)
    return logits
