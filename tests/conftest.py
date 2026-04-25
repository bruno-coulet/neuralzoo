"""
- Projet: tests automatises NeuralZOO.
- Role: configuration pytest (injection du root projet dans sys.path).
- Objectif: permettre des imports stables de src.* pendant les tests.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
