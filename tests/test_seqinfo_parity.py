"""``parse_seqinfo`` no longer imports ``configparser``; it must still agree with the stdlib
reader it replaced on well-formed and malformed MOT ``seqinfo.ini`` text (P1.4b)."""

from __future__ import annotations

import configparser

import pytest

from marinedata.task_layers.sources.boxes_csv import parse_seqinfo

CASES = [
    "[Sequence]\nimWidth=1920\nimHeight=1080\n",
    "[Sequence]\nname=MOT17-02\nframeRate=30\nimWidth = 960\nimHeight : 544\nimExt=.jpg\n",
    "[Sequence]\nIMWIDTH=640\nimheight=480",
    "# comment\n; another\n[Sequence]\nimWidth=1\nimHeight=2\n",
    "[Sequence]\nimWidth=1920\n",
    "[Sequence]\nimWidth=abc\nimHeight=1080\n",
    "imWidth=1920\n[Sequence]\nimHeight=1\n",
    "[Sequence]\nimWidth=1\nimWidth=2\nimHeight=3\n",
    "[Sequence]\nimWidth=1\nimHeight=2\n[Sequence]\nx=1\n",
    "[Other]\nimWidth=1\nimHeight=2\n",
    "[Sequence]\nimWidth=1\nnot an option\nimHeight=2\n",
    "[DEFAULT]\nimHeight=7\n[Sequence]\nimWidth=5\n",
    "[Sequence]\nimWidth=1920\n  imHeight=1080\n",
    "garbage",
    "",
]


def _stdlib(text: str) -> tuple[int | None, int | None]:
    cp = configparser.ConfigParser()
    try:
        cp.read_string(text)
        sec = cp["Sequence"]
        return int(sec["imWidth"]), int(sec["imHeight"])
    except (configparser.Error, KeyError, ValueError):
        return None, None


@pytest.mark.parametrize("text", CASES)
def test_parse_seqinfo_matches_the_stdlib_ini_reader(text: str) -> None:
    assert parse_seqinfo(text) == _stdlib(text)
