"""TensorFlow adapter.

TensorFlow has no ``ignore_index`` convention, so the masked hierarchical loss has to be
applied explicitly. This adapter emits a boolean mask per axis alongside the labels and
provides :func:`masked_sparse_categorical_crossentropy` so the multi-source behaviour
matches the PyTorch path exactly rather than approximately.

Labels still carry ``-100`` at unsupervised positions for parity with the torch adapter,
but **never rely on that in a TF loss** — an out-of-range index there is undefined
behaviour, not a skip. Use the mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..labelindex import IGNORE_INDEX

if TYPE_CHECKING:  # pragma: no cover
    from ..builder import Dataset


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install 'marinedata[tf]'"
        ) from exc
    return tf


def to_tf_dataset(
    dataset: Dataset,
    *,
    split: str | None = None,
    decode_images: bool = True,
    image_size: tuple[int, int] | None = (512, 512),
    batch_size: int | None = None,
    shuffle_buffer: int | None = None,
    seed: int = 0,
):
    """Build a ``tf.data.Dataset``.

    Args:
        image_size: resize target. ``None`` keeps native size, which then requires
            ``batch_size=None`` (or a ragged pipeline) because reef imagery is not
            uniformly sized.
        batch_size: batching happens here so the resize/batch interaction stays
            explicit rather than failing deep in a training loop.
    """
    tf = _require_tf()

    samples = dataset.split_samples(split) if split else dataset.samples
    index = dataset.label_index
    axes = sorted(a.value for a in index.axes)

    if not samples:
        raise ValueError(f"No samples for split {split!r}")
    if decode_images and image_size is None and batch_size:
        raise ValueError(
            "image_size=None cannot be batched: images differ in size. "
            "Set image_size, or leave batch_size=None and batch downstream."
        )

    paths: list[str] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for sample in samples:
        if sample.image is None:
            continue
        encoded = index.encode(sample)
        paths.append(str(sample.image))
        label_rows.append([encoded.get(a_enum, IGNORE_INDEX) for a_enum in sorted(index.axes)])
        mask_rows.append([a_enum in sample.supervised for a_enum in sorted(index.axes)])

    if not paths:
        raise ValueError("No samples carry an image; a tf.data image pipeline needs one.")

    base = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(paths),
            tf.constant(label_rows, dtype=tf.int64),
            tf.constant(mask_rows, dtype=tf.bool),
        )
    )

    def _load(path, labels, mask):
        features: dict[str, Any] = {"key": path}
        if decode_images:
            image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            image = tf.image.convert_image_dtype(image, tf.float32)
            if image_size is not None:
                image = tf.image.resize(image, image_size)
            features["image"] = image
        return (
            features,
            {a: labels[i] for i, a in enumerate(axes)},
            {a: mask[i] for i, a in enumerate(axes)},
        )

    pipeline = base.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_buffer:
        pipeline = pipeline.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    if batch_size:
        pipeline = pipeline.batch(batch_size)
    return pipeline.prefetch(tf.data.AUTOTUNE)


def masked_sparse_categorical_crossentropy(y_true, y_pred, mask):
    """Cross-entropy that ignores unsupervised positions.

    The TensorFlow equivalent of PyTorch's ``ignore_index=-100``. Without it, a source
    that never annotated an axis trains that head against fabricated targets — which
    degrades the head quietly and shows up as "the model is bad at growth form" rather
    than as a data bug.

    Returns 0.0 when nothing in the batch supervises the axis, so a batch drawn entirely
    from a coarse source is a no-op rather than a NaN.
    """
    tf = _require_tf()

    mask = tf.cast(mask, tf.float32)
    safe_true = tf.where(mask > 0, tf.cast(y_true, tf.int64), tf.zeros_like(y_true, tf.int64))
    losses = tf.keras.losses.sparse_categorical_crossentropy(safe_true, y_pred, from_logits=True)
    total = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(tf.reduce_sum(losses * mask), total)


__all__ = ["masked_sparse_categorical_crossentropy", "to_tf_dataset"]
