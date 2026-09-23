"""Registry loading and validation.

The registry is plain YAML under ``registry/`` so that adding a dataset is a reviewable
pull request rather than a code change. Validation is strict and happens at load: an
invalid entry fails the whole load rather than being skipped, because a silently
dropped source is indistinguishable from one that was never added.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

try:  # libyaml is ~4x faster and present in most environments
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover - pure-Python fallback
    from yaml import SafeLoader as _Loader

from .models import Licence, Profile, Source
from .schema import Crosswalk, LabelSchema
from .task import TaskKind, TaskSpec


class RegistryError(Exception):
    """Registry is malformed, inconsistent, or references something undefined."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_Loader)
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: invalid YAML — {exc}") from exc
    if data is None:
        raise RegistryError(f"{path}: file is empty")
    if not isinstance(data, dict):
        raise RegistryError(f"{path}: expected a mapping at top level, got {type(data).__name__}")
    return data


def _default_root() -> Path:
    """Packaged registry, falling back to the repo layout for editable installs."""
    packaged = Path(__file__).parent / "_registry"
    if packaged.is_dir():
        return packaged
    repo = Path(__file__).resolve().parents[2] / "registry"
    if repo.is_dir():
        return repo
    raise RegistryError(
        "Could not locate the registry directory. Pass an explicit path to Registry.load()."
    )


def _git_commit(path: Path) -> str | None:
    """The last commit that touched ``path``, or ``None`` outside a git checkout.

    A packaged (pip-installed) registry has no ``.git`` at all — that is a normal,
    expected case, not an error, so failures here are swallowed rather than raised.
    Scoped to ``path`` rather than bare ``HEAD``: a code-only commit elsewhere in the
    repo must not change what a lineage report claims about the registry's content.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


class Registry:
    """An immutable, validated view over the registry files.

    Construct via :meth:`load`. The constructor is not part of the public API — it
    takes already-validated objects and performs no checking.
    """

    def __init__(
        self,
        sources: dict[str, Source],
        licences: dict[str, Licence],
        profiles: dict[str, Profile],
        schemas: dict[str, LabelSchema] | None = None,
        crosswalks: dict[str, Crosswalk] | None = None,
        tasks: dict[str, TaskSpec] | None = None,
        commit: str | None = None,
    ) -> None:
        self._sources = dict(sources)
        self._licences = dict(licences)
        self._profiles = dict(profiles)
        self._schemas = dict(schemas or {})
        self._crosswalks = dict(crosswalks or {})
        self._tasks = dict(tasks or {})
        self.commit = commit
        """The registry directory's last commit, or ``None`` outside a git checkout.
        Threaded into every :class:`~marinedata.lineage.Lineage` this registry builds,
        so the audit record can actually re-derive what it claims to bind."""

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, root: str | Path | None = None) -> Registry:
        """Load and validate the whole registry. Raises on any inconsistency."""
        base = Path(root) if root is not None else _default_root()
        if not base.is_dir():
            raise RegistryError(f"Registry root does not exist: {base}")

        licences = cls._load_licences(base / "licences.yaml")
        profiles = cls._load_profiles(base / "profiles.yaml")
        sources = cls._load_sources(base / "sources", licences)
        schemas = cls._load_collection(base / "schemas", "schemas", LabelSchema)
        tasks = cls._load_collection(base / "tasks", "tasks", TaskSpec)
        crosswalks = cls._load_collection(base / "crosswalks", "crosswalks", Crosswalk)
        cls._check_references(sources, schemas, crosswalks)
        for task in tasks.values():
            if task.kind is not TaskKind.SUPERVISED:
                continue  # no schema to validate against
            if task.schema_id not in schemas:
                raise RegistryError(f"task '{task.id}' targets unknown schema '{task.schema_id}'")
            task.validate_against(schemas[task.schema_id])
        return cls(sources, licences, profiles, schemas, crosswalks, tasks, _git_commit(base))

    @staticmethod
    def _load_collection(base: Path, key: str, model: type) -> dict[str, object]:
        """Load a directory of YAML files each containing a list under ``key``."""
        out: dict[str, object] = {}
        if not base.is_dir():
            return out
        for path in sorted(base.rglob("*.yaml")):
            for entry in _read_yaml(path).get(key, []):
                try:
                    obj = model.model_validate(entry)
                except Exception as exc:
                    raise RegistryError(f"{path}: invalid {key[:-1]} — {exc}") from exc
                if obj.id in out:
                    raise RegistryError(f"{path}: duplicate {key[:-1]} id '{obj.id}'")
                out[obj.id] = obj
        return out

    @staticmethod
    def _check_references(
        sources: dict[str, Source],
        schemas: dict[str, object],
        crosswalks: dict[str, object],
    ) -> None:
        """A loader pointing at a missing schema should fail at load, not at epoch 1."""
        for src in sources.values():
            for images_source_id in src.images_from:
                if images_source_id not in sources:
                    raise RegistryError(
                        f"source '{src.id}' images_from references unknown source "
                        f"'{images_source_id}'"
                    )
            spec = src.loader
            if spec is not None:
                if spec.schema_id and spec.schema_id not in schemas:
                    raise RegistryError(
                        f"source '{src.id}' references unknown schema '{spec.schema_id}'"
                    )
                if spec.crosswalk_id and spec.crosswalk_id not in crosswalks:
                    raise RegistryError(
                        f"source '{src.id}' references unknown crosswalk '{spec.crosswalk_id}'"
                    )
            for ann in src.annotations:
                if ann.schema_id and ann.schema_id not in schemas:
                    raise RegistryError(
                        f"source '{src.id}' annotations.schema_id references unknown "
                        f"schema '{ann.schema_id}'"
                    )
        for walk in crosswalks.values():
            if walk.target_schema not in schemas:
                raise RegistryError(
                    f"crosswalk '{walk.id}' targets unknown schema '{walk.target_schema}'"
                )

    @staticmethod
    def _load_licences(path: Path) -> dict[str, Licence]:
        raw = _read_yaml(path).get("licences")
        if not raw:
            raise RegistryError(f"{path}: no 'licences' key")
        out: dict[str, Licence] = {}
        for entry in raw:
            lic = Licence.model_validate(entry)
            if lic.id in out:
                raise RegistryError(f"{path}: duplicate licence id '{lic.id}'")
            out[lic.id] = lic
        return out

    @staticmethod
    def _load_profiles(path: Path) -> dict[str, Profile]:
        raw = _read_yaml(path).get("profiles")
        if not raw:
            raise RegistryError(f"{path}: no 'profiles' key")
        out: dict[str, Profile] = {}
        for entry in raw:
            prof = Profile.model_validate(entry)
            if prof.id in out:
                raise RegistryError(f"{path}: duplicate profile id '{prof.id}'")
            out[prof.id] = prof
        return out

    @staticmethod
    def _load_sources(base: Path, licences: dict[str, Licence]) -> dict[str, Source]:
        if not base.is_dir():
            raise RegistryError(f"Sources directory does not exist: {base}")

        out: dict[str, Source] = {}
        for path in sorted(base.rglob("*.yaml")):
            doc = _read_yaml(path)
            for entry in doc.get("sources", []):
                resolved = Registry._resolve_licence(entry, licences, path)
                try:
                    src = Source.model_validate(resolved)
                except Exception as exc:
                    raise RegistryError(f"{path}: invalid source entry — {exc}") from exc
                if src.id in out:
                    raise RegistryError(f"{path}: duplicate source id '{src.id}'")
                out[src.id] = src

        if not out:
            raise RegistryError(f"No sources found under {base}")
        return out

    @staticmethod
    def _resolve_licence(
        entry: dict[str, Any], licences: dict[str, Licence], path: Path
    ) -> dict[str, Any]:
        """Expand a ``licence: <id>`` reference into the full licence object.

        Entries reference licences by id so that a licence correction propagates to
        every source at once rather than needing a sweep.
        """
        lic_ref = entry.get("licence")
        if not isinstance(lic_ref, str):
            return entry  # inline licence object, or missing — let pydantic report it
        if lic_ref not in licences:
            known = ", ".join(sorted(licences))
            raise RegistryError(
                f"{path}: source '{entry.get('id')}' references unknown licence "
                f"'{lic_ref}'. Known: {known}"
            )
        return {**entry, "licence": licences[lic_ref].model_dump()}

    # ── access ────────────────────────────────────────────────────────────

    @property
    def sources(self) -> tuple[Source, ...]:
        return tuple(self._sources.values())

    @property
    def profiles(self) -> tuple[Profile, ...]:
        return tuple(self._profiles.values())

    def source(self, source_id: str) -> Source:
        try:
            return self._sources[source_id]
        except KeyError:
            raise RegistryError(f"Unknown source '{source_id}'") from None

    def profile(self, profile_id: str) -> Profile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            known = ", ".join(sorted(self._profiles))
            raise RegistryError(f"Unknown profile '{profile_id}'. Known: {known}") from None

    @property
    def schemas(self) -> tuple[LabelSchema, ...]:
        return tuple(self._schemas.values())

    @property
    def crosswalks(self) -> tuple[Crosswalk, ...]:
        return tuple(self._crosswalks.values())

    @property
    def tasks(self) -> tuple[TaskSpec, ...]:
        return tuple(self._tasks.values())

    def task(self, task_id: str) -> TaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError:
            known = ", ".join(sorted(self._tasks)) or "(none loaded)"
            raise RegistryError(f"Unknown task '{task_id}'. Known: {known}") from None

    def projector_for(self, task_id: str):
        """Build the projector for a task, validated against its schema.

        Supervised tasks only — a self-supervised task has no vocabulary to project
        labels onto.
        """
        from .task import TaskProjector

        task = self.task(task_id)
        if task.kind is not TaskKind.SUPERVISED:
            raise RegistryError(
                f"task '{task.id}' is self-supervised and has no projector — it fixes "
                f"no target vocabulary, so nothing needs projecting"
            )
        return TaskProjector(task, self.label_schema(task.schema_id))

    def sources_for_task(self, task_id: str, *, contributing_only: bool = True):
        """Sources that can serve a task, with what each can actually contribute.

        A source is *eligible* when it supervises the task's axis through a crosswalk
        into the task's schema. It *contributes* only if some of its labels reach the
        task's classes — Coralscapes is eligible for a Caribbean genus task and supplies
        nothing to it, because its genera are Indo-Pacific.

        For a self-supervised task there is no crosswalk or axis to check: any source
        matching the task's ``modalities`` (or any modality, if unset) contributes —
        labelled or not, since a pretraining corpus wants the images regardless.

        Returns ``(Source, SourceFit)`` pairs. Set ``contributing_only=False`` to see the
        eligible-but-useless ones too, which is the interesting case when a task looks
        well-supported and is not.
        """
        from .task import SourceFit, fit_source

        task = self.task(task_id)

        if task.kind is not TaskKind.SUPERVISED:
            return tuple(
                (
                    source,
                    SourceFit(source_id=source.id, reachable=(), abstaining=(), unsupervised=True),
                )
                for source in self._sources.values()
                if not task.modalities or any(m in task.modalities for m in source.modalities)
            )

        projector = self.projector_for(task_id)
        out = []
        for source in self._sources.values():
            spec = source.loader
            if spec is None or not spec.crosswalk_id:
                continue
            walk = self.crosswalk(spec.crosswalk_id)
            if walk.target_schema != task.schema_id:
                continue
            if not any(task.axis in ann.supervises for ann in source.annotations):
                continue
            fit = fit_source(projector, walk, source.id)
            if fit.contributes or not contributing_only:
                out.append((source, fit))
        return tuple(out)

    def label_schema(self, schema_id: str) -> LabelSchema:
        try:
            return self._schemas[schema_id]
        except KeyError:
            known = ", ".join(sorted(self._schemas)) or "(none loaded)"
            raise RegistryError(f"Unknown schema '{schema_id}'. Known: {known}") from None

    def crosswalk(self, crosswalk_id: str) -> Crosswalk:
        try:
            return self._crosswalks[crosswalk_id]
        except KeyError:
            known = ", ".join(sorted(self._crosswalks)) or "(none loaded)"
            raise RegistryError(f"Unknown crosswalk '{crosswalk_id}'. Known: {known}") from None

    def harmonizer_for(self, source_id: str):
        """Build the harmonizer a source declares, or ``None`` if it declares none."""
        from .harmonize import Harmonizer

        spec = self.source(source_id).loader
        if spec is None or not spec.crosswalk_id:
            return None
        walk = self.crosswalk(spec.crosswalk_id)
        return Harmonizer(walk, self.label_schema(walk.target_schema))

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self) -> Iterator[Source]:
        return iter(self._sources.values())
