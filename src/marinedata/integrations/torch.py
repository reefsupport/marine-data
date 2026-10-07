"""PyTorch adapter.

Yields exactly what a masked hierarchical head needs:

    {"image": Tensor, "labels": {axis: int}, "supervised": {axis: bool}, ...}

Unsupervised axes carry ``-100``, PyTorch's ``CrossEntropyLoss`` default
``ignore_index``. So the multi-source masked loss — the thing that lets Coralscapes,
BenthicNet and our own 2-class masks train one model together — works with a plain
``nn.CrossEntropyLoss()`` and no custom masking code:

    loss = sum(F.cross_entropy(logits[a], batch["labels"][a]) for a in heads)

Positions the source never annotated contribute nothing. That is the whole trick, and
it is easy to get wrong in a way no loss curve reveals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ..labelindex import IGNORE_INDEX
from ..sample import Sample
from ..schema import Axis

if TYPE_CHECKING:  # pragma: no cover
    from ..builder import Dataset


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is not installed. Install it with: pip install 'marinedata[torch]'"
        ) from exc
    return torch


def default_image_loader(path: str):
    """Decode an image to CHW float32 in [0, 1].

    Pillow only — no torchvision dependency, since torchvision pins torch versions and
    this package has no business constraining that for its users.
    """
    torch = _require_torch()
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow and numpy are needed to decode images: pip install 'marinedata[torch]'"
        ) from exc

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def default_mask_loader(path: str):
    """Decode a segmentation mask to an int64 HW tensor of raw class ids.

    Deliberately not remapped: mask palettes are source-specific, and silently
    reindexing them here would hide a mismatch that ought to be explicit in a crosswalk.
    Pass ``scheme=`` to :func:`to_torch_dataset` (or use :mod:`marinedata.labels`) to map the
    native ids onto a fixed scheme explicitly.
    """
    torch = _require_torch()
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        array = np.asarray(handle, dtype="int64")
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(array)


def to_torch_dataset(
    dataset: Dataset,
    *,
    split: str | None = None,
    decode_images: bool = True,
    decode_masks: bool = False,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    image_loader: Callable[[str], Any] | None = None,
    mask_loader: Callable[[str], Any] | None = None,
    scheme: str | None = None,
    scheme_options: Mapping[str, Iterable[str] | None] | None = None,
):
    """Wrap a :class:`~marinedata.builder.Dataset` as a ``torch.utils.data.Dataset``.

    Args:
        decode_images: set False to keep paths — useful when a custom pipeline handles
            IO, or when profiling the loader rather than the model.
        transform: applied to the assembled item dict, after decoding.
        scheme: a :mod:`marinedata.labels` scheme (e.g. ``"benthic-coarse"``). Decoded masks are
            then remapped from each sample's native ids to the scheme's fixed ids (255 = ignore),
            using the sample's ``source_id``. ``None`` (default) leaves masks untouched.
        scheme_options: ``exclude_conditions`` / ``ignore`` for the scheme (see ``labels``). Left
            out, the scheme defaults apply: dead coral is 255 in ``benthic-coarse`` and
            ``coral-binary``; ``{"exclude_conditions": ()}`` keeps it as coral.
    """
    torch = _require_torch()
    from torch.utils.data import Dataset as TorchDataset

    samples = dataset.split_samples(split) if split else dataset.samples
    index = dataset.label_index
    projector = dataset.projector
    load_image = image_loader or default_image_loader
    load_mask = mask_loader or default_mask_loader
    luts: dict[str, Any] = {}

    def remap(source_id: str, mask: Any) -> Any:
        """Native ids -> scheme ids via the source's LUT (built once per source)."""
        if source_id not in luts:
            from .. import labels

            table = labels.lut(source_id, scheme, **(scheme_options or {}))
            luts[source_id] = torch.from_numpy(table.astype("int64"))
        return luts[source_id][mask.long()]

    class MarineTorchDataset(TorchDataset):
        """Torch view over a licence-cleared, harmonised marine dataset."""

        def __init__(self) -> None:
            self.samples: list[Sample] = samples
            self.label_index = index
            self.num_classes = index.num_classes()

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, position: int) -> dict[str, Any]:
            sample = self.samples[position]
            encoded = index.encode(sample, projector=projector)

            item: dict[str, Any] = {
                "source_id": sample.source_id,
                "key": sample.key,
                "labels": {a.value: torch.tensor(v, dtype=torch.long) for a, v in encoded.items()},
                "supervised": {a.value: torch.tensor(a in sample.supervised) for a in encoded},
            }

            if sample.image is not None:
                item["image"] = (
                    load_image(str(sample.image)) if decode_images else str(sample.image)
                )
            if sample.mask is not None:
                item["mask"] = load_mask(str(sample.mask)) if decode_masks else str(sample.mask)
                if scheme is not None and decode_masks:
                    item["mask"] = remap(sample.source_id, item["mask"])
            if sample.boxes:
                item["boxes"] = torch.tensor(sample.boxes, dtype=torch.float32)
            if sample.points:
                item["points"] = torch.tensor(sample.points, dtype=torch.float32)

            return transform(item) if transform else item

        def class_weights(self) -> dict[str, Any]:
            """Inverse-frequency weights, ready for ``CrossEntropyLoss(weight=...)``."""
            raw = index.class_weights(self.samples)
            return {a.value: torch.tensor(w, dtype=torch.float32) for a, w in raw.items()}

    return MarineTorchDataset()


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate items whose per-sample shapes may differ.

    The default torch collate cannot handle the nested label/supervision dicts, and
    stacks images that may differ in size. Here, images stack only when their shapes
    agree and are otherwise left as a list — reef imagery is not uniformly sized, and a
    crash at batch time is a worse answer than an honest list.
    """
    torch = _require_torch()
    if not batch:
        return {}

    out: dict[str, Any] = {
        "source_id": [b["source_id"] for b in batch],
        "key": [b["key"] for b in batch],
    }

    for field in ("labels", "supervised"):
        axes = batch[0].get(field, {})
        out[field] = {a: torch.stack([b[field][a] for b in batch]) for a in axes}

    for field in ("image", "mask"):
        values = [b[field] for b in batch if field in b]
        if not values:
            continue
        if torch.is_tensor(values[0]) and len({tuple(v.shape) for v in values}) == 1:
            out[field] = torch.stack(values)
        else:
            out[field] = values

    for field in ("boxes", "points"):
        values = [b[field] for b in batch if field in b]
        if values:
            out[field] = values  # ragged by nature

    return out


__all__ = [
    "IGNORE_INDEX",
    "Axis",
    "collate",
    "default_image_loader",
    "default_mask_loader",
    "to_torch_dataset",
]
