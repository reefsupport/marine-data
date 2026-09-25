"""P5 baselines: frozen linear probe + small fine-tune (design §4.4/§4.5).

Weights are pinned by HF revision sha, fetched anonymously (no login — D-AA). Two
permissive backbones only: DINOv2-S/14 (Apache-2.0) and OpenCLIP ViT-B/16, a LAION-2B
checkpoint (MIT).
"""

from __future__ import annotations
