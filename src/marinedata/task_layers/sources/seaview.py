"""SEAVIEW's excluded pickle: static opcode scan, then a restricted Unpickler inside a
network-isolated container — or images-only, per the charter's D-Z2.

``labelled_data/labelled_segmentation_data.pickle`` (6,956,023,722 B) sits outside every
staged SEAVIEW prefix and is excluded from the registry's ``seaview-survey-imagery`` size
count as an unsafe format. D-Z2 allows converting it to CSV *only if* a static opcode
scan (:func:`scan_globals`, ``pickletools.genops`` — no execution) shows every
``GLOBAL``/``STACK_GLOBAL`` target inside :data:`ALLOWED_MODULES`/:data:`ALLOWED_BUILTINS`
(pandas/numpy reconstruction, plain ``builtins`` containers, ``datetime``,
``collections.OrderedDict``). Any other target means the pickle stays unopened and
SEAVIEW ships images-only — this module never falls back to ``pickle.load``.

The scan is a static heuristic, not a full stack machine: ``GLOBAL`` (protocol <=3)
carries its "module name" as one opcode argument; ``STACK_GLOBAL`` (protocol 4+) pops two
preceding string pushes off the pickle VM's stack, so this scanner tracks the last two
string-literal opcodes seen (ignoring ``MEMOIZE``/frame opcodes, which don't touch the
stack) and pairs them with the next ``STACK_GLOBAL``. This is the same heuristic used by
disassembly-only pickle scanners; it cannot be fooled into under-reporting a global
(every ``STACK_GLOBAL`` needs exactly two immediately-preceding string pushes to resolve
to a real module/name), only into mis-attributing one to the wrong two strings if a
pickle deliberately interleaves unrelated string pushes — which would itself make the
two "strings" not resolve to a real, allowlisted module, and so still reject.
"""

from __future__ import annotations

import argparse
import pickletools
import subprocess
from pathlib import Path
from typing import BinaryIO

ALLOWED_MODULES = ("pandas", "numpy")
"""Any submodule under these (``pandas.core.frame``, ``numpy.core.multiarray``, …) is ok."""

ALLOWED_BUILTINS = {"list", "dict", "set", "tuple", "frozenset", "slice"}
ALLOWED_BUILTIN_MODULES = {"builtins", "__builtin__"}
"""``__builtin__`` is the Python-2 pickle-protocol-2 spelling of ``builtins`` — every
protocol-2 pickle containing e.g. a ``slice`` references it this way (verified 2026-09-25
against a real pandas 3.x DataFrame pickle); it is the same objects, not a wider grant."""
ALLOWED_EXACT = {("collections", "OrderedDict"), ("_codecs", "encode")}
"""``_codecs.encode`` is CPython's own pickle machinery for embedding bytes/str literals
in protocol>=2 pickles — present in virtually any such pickle, not data-package-specific."""
ALLOWED_DATETIME_MODULE = "datetime"

_STRING_OPS = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
    "SHORT_BINSTRING",
    "BINSTRING",
}


class UnsafeGlobal(RuntimeError):
    """A GLOBAL/STACK_GLOBAL target outside the allowlist — SEAVIEW stays images-only."""


def _is_allowed(module: str, name: str) -> bool:
    top = module.split(".", 1)[0]
    if top in ALLOWED_MODULES:
        return True
    if module in ALLOWED_BUILTIN_MODULES and name in ALLOWED_BUILTINS:
        return True
    if (module, name) in ALLOWED_EXACT:
        return True
    return module == ALLOWED_DATETIME_MODULE


def scan_globals(stream: BinaryIO) -> list[tuple[str, str]]:
    """Every ``(module, name)`` named by a GLOBAL/STACK_GLOBAL opcode, in file order.

    No pickle bytes are ever handed to ``pickle.load``/``Unpickler`` here — this is
    disassembly only (``pickletools.genops``), never execution.
    """
    found: list[tuple[str, str]] = []
    pending: list[str] = []
    for opcode, arg, _pos in pickletools.genops(stream):
        if opcode.name in _STRING_OPS and isinstance(arg, str):
            pending.append(arg)
            pending = pending[-2:]
        elif opcode.name == "GLOBAL" and isinstance(arg, str):
            module, _, name = arg.partition(" ")
            found.append((module, name))
        elif opcode.name == "STACK_GLOBAL":
            if len(pending) == 2:
                found.append((pending[0], pending[1]))
            pending = []
    return found


def check_allowlist(globals_found: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the disallowed ``(module, name)`` pairs (empty means the scan is clean)."""
    return [g for g in globals_found if not _is_allowed(*g)]


def convert_in_container(
    pickle_path: Path,
    out_dir: Path,
    *,
    image: str = "python:3.12-slim",
) -> subprocess.CompletedProcess[str]:
    """Run the restricted-Unpickler conversion inside a network-isolated container.

    Only the pickle (read-only) and one output directory are mounted; the container has
    no network (``--network none``) beyond the initial, already-authorized image pull,
    and its filesystem is read-only apart from ``/out``. Never call this unless
    :func:`check_allowlist` on the same file returned an empty list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "-v",
        f"{pickle_path}:/in.pkl:ro",
        "-v",
        f"{out_dir}:/out",
        image,
        "sh",
        "-c",
        "pip install --quiet --no-index --no-deps pandas numpy 2>/dev/null; "
        f"python3 -c {_shell_quote(_RESTRICTED_UNPICKLE_SRC)}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


_RESTRICTED_UNPICKLE_SRC = """
import io, pickle, sys

ALLOWED_MODULES = ("pandas", "numpy")
ALLOWED_BUILTINS = {"list", "dict", "set", "tuple", "frozenset", "slice"}
ALLOWED_BUILTIN_MODULES = {"builtins", "__builtin__"}
ALLOWED_EXACT = {("collections", "OrderedDict"), ("_codecs", "encode")}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        top = module.split(".", 1)[0]
        if top in ALLOWED_MODULES or module == "datetime" or (module, name) in ALLOWED_EXACT:
            return super().find_class(module, name)
        if module in ALLOWED_BUILTIN_MODULES and name in ALLOWED_BUILTINS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"disallowed global {module}.{name}")


with open("/in.pkl", "rb") as fh:
    obj = RestrictedUnpickler(fh).load()
obj.to_csv("/out/seaview_points.csv", index=False)
print("wrote", len(obj), "rows")
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="static opcode scan only (no download to disk)")
    scan.add_argument("--bucket", required=True)
    scan.add_argument("--key", required=True)
    scan.add_argument("--remote", default="rs-hel1")
    args = p.parse_args(argv)
    if args.cmd == "scan":
        from ...s3_upload import client_from_rclone

        client = client_from_rclone(args.remote)
        body = client.get_object(Bucket=args.bucket, Key=args.key)["Body"]
        found = scan_globals(body)
        bad = check_allowlist(found)
        print(f"globals found: {len(found)}, disallowed: {len(bad)}")
        for g in bad[:20]:
            print("DISALLOWED", g)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
