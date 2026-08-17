"""Framework adapters.

Each is import-guarded: pandas, torch and tensorflow are optional extras, and importing
``marinedata`` must never require any of them. A registry that pulls in TensorFlow to
answer a licence question is a registry people route around.
"""

from __future__ import annotations

__all__ = ["pandas", "tensorflow", "torch"]
