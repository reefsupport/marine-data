"""The Dockerfiles build on the classic Docker builder: the release VM has no buildx.

PR #16 put ``ADD --checksum=... --chmod=0444`` in the release Dockerfile. Both flags need
BuildKit, and the VM's plain ``docker build`` stopped with "dockerfile parse error line 57: Unknown
flag: checksum". Nothing builds the image before the VM does, so these tests read the Dockerfiles
line by line, and run the script that replaced ``ADD --checksum`` against local files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from marinedata.gates import yunet_model

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = ("Dockerfile", "deploy/ingest-job/Dockerfile")
FETCH = "deploy/release-vm/fetch-pinned.py"

# The only flags the classic builder parses. Anything else after the keyword needs BuildKit:
# --checksum, --chmod, --link (ADD, COPY); --mount, --network, --security (RUN); and the newer
# --keep-git-dir, --exclude, --parents. `--chown` and `COPY --from` are fine.
CLASSIC_FLAGS = {"ADD": {"--chown"}, "COPY": {"--chown", "--from"}, "RUN": set()}
HEREDOC = re.compile(r"<<-?\s*[\"']?[A-Za-z_]")


def instructions(dockerfile: Path) -> list[str]:
    """The Dockerfile's instructions: comment lines dropped, backslash continuations joined."""
    joined: list[str] = []
    pending = ""
    for raw in dockerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
        else:
            joined.append(pending + line)
            pending = ""
    if pending:
        joined.append(pending.strip())
    return joined


def buildkit_only(instruction: str) -> list[str]:
    """What in one instruction the classic builder cannot parse (empty when it is fine)."""
    keyword, _, rest = instruction.partition(" ")
    keyword = keyword.upper()
    if keyword not in CLASSIC_FLAGS:
        return []
    found = []
    for word in rest.split():
        if not word.startswith("--"):
            break
        flag = word.partition("=")[0]
        if flag not in CLASSIC_FLAGS[keyword]:
            found.append(f"{keyword} {flag}")
    if HEREDOC.search(rest):
        found.append(f"{keyword} heredoc")
    return found


@pytest.mark.parametrize("name", DOCKERFILES)
def test_dockerfile_uses_no_buildkit_only_syntax(name):
    steps = instructions(ROOT / name)
    assert any(step.startswith("RUN ") for step in steps), f"{name}: read no RUN instruction"
    found = [f"{flag} in `{step[:60]}`" for step in steps for flag in buildkit_only(step)]
    assert not found, (
        f"{name} needs BuildKit, the release VM builds with the classic builder: {found}"
    )


@pytest.mark.parametrize(
    "step",
    [
        "ADD --checksum=sha256:abc https://example.org/m.onnx /opt/m.onnx",
        "ADD --chmod=0444 https://example.org/m.onnx /opt/m.onnx",
        "COPY --link . .",
        "COPY --chmod=0755 run.sh /run.sh",
        "COPY --chown=1:1 --parents src ./",
        "RUN --mount=type=cache,target=/root/.cache pip install x",
        "RUN --network=none true",
        "RUN --security=insecure true",
        "RUN <<EOF",
        "RUN cat <<-'EOF' > /etc/x",
        "COPY <<EOF /app/x",
        "ADD <<EOF /app/x",
    ],
)
def test_the_detector_flags_buildkit_only_instructions(step):
    assert buildkit_only(step)


@pytest.mark.parametrize(
    "step",
    [
        "COPY --from=uv /uv /usr/local/bin/uv",
        "COPY --chown=1:1 --from=build /out /out",
        "ADD --chown=1:1 a.tar.gz /srv/",
        "RUN uv sync --frozen --no-dev --extra ingest",
        "RUN python3 deploy/release-vm/fetch-pinned.py URL sha256:abc /x && chmod 0444 /x",
        "ENV A=1 B=2",
        "FROM python:3.12-slim AS base",
    ],
)
def test_the_detector_accepts_what_the_classic_builder_parses(step):
    assert buildkit_only(step) == []


def test_instructions_drop_comments_and_join_continuations(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# ADD --checksum=x is only named in this comment\n"
        "RUN a \\\n    # a comment between continuation lines\n    && b\n\n"
        "ADD --chmod=0444 \\\n    u \\\n    d\n"
    )
    assert instructions(dockerfile) == ["RUN a && b", "ADD --chmod=0444 u d"]


# the step that replaced `ADD --checksum`


def yunet_step() -> list[str]:
    (step,) = [s for s in instructions(ROOT / "Dockerfile") if FETCH in s]
    return shlex.split(step)


def test_the_yunet_step_fetches_the_pinned_model_and_stops_the_build_when_it_fails():
    model = f"/opt/models/{yunet_model.YUNET_FILE}"
    assert yunet_step() == [
        "RUN",
        "python3",
        FETCH,
        yunet_model.YUNET_URL,
        f"sha256:{yunet_model.YUNET_SHA256}",
        model,
        "&&",
        "chmod",
        "0444",
        model,
    ]
    assert f"ENV {yunet_model.ENV_VAR}={model}" in instructions(ROOT / "Dockerfile")


def test_the_fetch_script_is_in_the_build_context():
    assert (ROOT / FETCH).is_file()
    patterns = [
        line.strip().rstrip("/")
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    paths = [FETCH, *(str(parent) for parent in Path(FETCH).parents if str(parent) != ".")]
    assert [p for p in paths for pattern in patterns if fnmatch.fnmatch(p, pattern)] == []


def fetch(url: str, pin: str, dest: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(ROOT / FETCH), url, pin, str(dest)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def test_fetch_writes_the_file_when_the_digest_matches(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    dest = tmp_path / "opt" / "models" / "model.onnx"
    done = fetch(source.as_uri(), sha(b"weights"), dest)
    assert done.returncode == 0, done.stderr
    assert dest.read_bytes() == b"weights" and not list(tmp_path.rglob("*.part"))


def test_fetch_fails_and_writes_nothing_on_a_digest_mismatch(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"something else")
    dest = tmp_path / "opt" / "model.onnx"
    done = fetch(source.as_uri(), sha(b"weights"), dest)
    assert done.returncode != 0 and "sha256 mismatch" in done.stderr
    assert not dest.exists() and not list(tmp_path.rglob("*.part"))


def test_fetch_fails_when_the_url_cannot_be_read(tmp_path):
    dest = tmp_path / "model.onnx"
    done = fetch((tmp_path / "missing.bin").as_uri(), sha(b"weights"), dest)
    assert done.returncode != 0 and "cannot read" in done.stderr and not dest.exists()


@pytest.mark.parametrize("pin", ["0" * 64, "sha256:abc", f"md5:{'0' * 32}", f"sha256:{'A' * 64}"])
def test_fetch_rejects_a_malformed_pin(tmp_path, pin):
    done = fetch((tmp_path / "x.bin").as_uri(), pin, tmp_path / "y.onnx")
    assert done.returncode == 2 and "sha256:HEX" in done.stderr
