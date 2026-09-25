"""Zip member bytes, including Deflate64 (method 9) — WP-6j, D-AH (2).

CPython's :mod:`zipfile` refuses method 9 (``NotImplementedError: That compression
method is not supported``); the Kakadu fish-AI zips use it. Members are inflated with
the maintained ``inflate64`` wheel (py7zr's decoder) by reading the raw member bytes
at the local header — public ``ZipInfo`` fields only, no patching of ``zipfile``
internals, so it works on any seekable file object a ``ZipFile`` was opened on (a
local spool or an HTTP Range reader). Size and CRC-32 are checked like ``zf.read``.
"""

from __future__ import annotations

import struct
import zipfile
import zlib

DEFLATE64 = 9
_LOCAL_HEADER = 30
_LOCAL_SIG = b"PK\x03\x04"
_CHUNK = 1 << 20


def read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """``zf.read(info)``, except Deflate64 members are inflated via ``inflate64``."""
    if info.compress_type != DEFLATE64:
        return zf.read(info)
    if info.flag_bits & 0x1:
        raise NotImplementedError(f"{info.filename}: encrypted Deflate64 member")
    import inflate64  # lazy: only Deflate64 archives need the wheel

    fp = zf.fp
    if fp is None:
        raise ValueError("read_member: ZipFile is closed")
    fp.seek(info.header_offset)
    header = fp.read(_LOCAL_HEADER)
    if len(header) != _LOCAL_HEADER or header[:4] != _LOCAL_SIG:
        raise zipfile.BadZipFile(f"{info.filename}: bad local header")
    name_len, extra_len = struct.unpack("<2H", header[26:30])
    fp.seek(info.header_offset + _LOCAL_HEADER + name_len + extra_len)
    inflater = inflate64.Inflater()
    out = bytearray()
    remaining = info.compress_size
    while remaining > 0:
        chunk = fp.read(min(_CHUNK, remaining))
        if not chunk:
            raise zipfile.BadZipFile(f"{info.filename}: truncated Deflate64 member")
        remaining -= len(chunk)
        out += inflater.inflate(chunk)
    data = bytes(out)
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise zipfile.BadZipFile(f"{info.filename}: Deflate64 size/CRC-32 mismatch")
    return data
