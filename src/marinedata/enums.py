"""Controlled vocabularies.

Every facet a caller can filter on is an enum, not a free string. Typos become
import-time errors instead of silently-empty query results — which, for a licence
gate, is the difference between "no data" and "wrong data".
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    """Licence tier. Ordering is *not* meaningful — use profiles, not comparisons.

    The tier answers "what class of permission do we hold?". The flags on
    :class:`~marinedata.models.LicenceFlags` answer "what specific acts are barred?".
    Both are required; neither subsumes the other.
    """

    OWN = "T0_OWN"
    """We hold the rights outright. No third-party restriction."""

    PERMISSIVE = "T1_PERMISSIVE"
    """CC0, CC-BY, Apache-2.0, MIT, BSD, US-Gov public domain. Shippable."""

    COPYLEFT = "T2_COPYLEFT"
    """CC-BY-SA, GPL. Shippable only if the derivative is released under the same terms."""

    NONCOMMERCIAL = "T3_NONCOMMERCIAL"
    """CC-BY-NC and variants. Research and internal use only."""

    TDM_ONLY = "T4_TDM_ONLY"
    """No licence grant. Lawfully accessible. Relies on a statutory text-and-data-mining
    exception (e.g. Art. 15o Auteurswet / DSM Art. 4). Jurisdiction-dependent and an
    affirmative defence rather than a permission — see docs/LEGAL.md."""

    PROHIBITED = "TX_PROHIBITED"
    """Provenance-defective or contract-blocked. Never usable, on any profile."""


class LegalBasis(str, Enum):
    """*Why* we are allowed to use this item. Recorded per source, surfaced in lineage."""

    OWN = "own"
    LICENCE = "licence"
    PERMISSION = "permission"
    """Written permission from the rightsholder, overriding the public licence."""
    TDM = "tdm"
    """Statutory TDM exception. Requires a counsel opinion reference on the profile."""
    UNKNOWN = "unknown"
    """Unresolved. Never usable — present so gaps are explicit rather than absent."""


class Redistribution(str, Enum):
    """Whether we may redistribute a verbatim copy of this source. Recorded per source,
    surfaced in lineage.

    Orthogonal to :class:`LegalBasis` and to ``LicenceFlags.no_derivatives`` — a source
    can permit reading data in place while forbidding redistribution, or permit
    redistributing an unmodified copy while forbidding derivative works from it. This
    field is a recorded position, never a storage or mirror filter.
    """

    OK = "ok"
    """The licence or an explicit grant permits redistributing an unmodified copy."""
    PROHIBITED = "prohibited"
    """Redistribution is barred outright — provenance-defective or contract-blocked."""
    UNKNOWN = "unknown"
    """Unresolved, or the basis does not establish a redistribution right. Default."""


class Capability(str, Enum):
    """What a source is *useful for*. The primary axis most users will query on."""

    # Benthic / coral
    BENTHIC_SEGMENTATION = "benthic-segmentation"
    BENTHIC_CLASSIFICATION = "benthic-classification"
    CORAL_INSTANCE_SEGMENTATION = "coral-instance-segmentation"
    BLEACHING_ASSESSMENT = "bleaching-assessment"
    CORAL_TAXONOMY = "coral-taxonomy"
    RESTORATION_MONITORING = "restoration-monitoring"

    # Fish & mobile fauna
    FISH_DETECTION = "fish-detection"
    FISH_CLASSIFICATION = "fish-classification"
    FISH_TRACKING = "fish-tracking"
    FISH_BIOMASS = "fish-biomass"
    INVERTEBRATE_DETECTION = "invertebrate-detection"
    MEGAFAUNA_ID = "megafauna-id"

    # Geometry
    THREE_D_RECONSTRUCTION = "3d-reconstruction"
    DEPTH_ESTIMATION = "depth-estimation"
    POSE_SLAM = "pose-slam"
    STRUCTURAL_COMPLEXITY = "structural-complexity"

    # Image quality
    IMAGE_ENHANCEMENT = "image-enhancement"
    SUPER_RESOLUTION = "super-resolution"

    # Remote sensing
    HABITAT_MAPPING_SATELLITE = "habitat-mapping-satellite"
    BATHYMETRY = "bathymetry"
    SEAGRASS_MAPPING = "seagrass-mapping"

    # Other modalities
    BIOACOUSTICS = "bioacoustics"
    EDNA_TAXONOMY = "edna-taxonomy"
    MARINE_DEBRIS = "marine-debris"

    # Cross-cutting
    GENERAL_PRETRAINING = "general-pretraining"
    """Self-supervised pretraining. Needs no labels — most image sources qualify."""
    VQA_CAPTIONING = "vqa-captioning"


class AnnotationKind(str, Enum):
    """The form of supervision carried, which determines what you can train."""

    NONE = "none"
    DENSE_MASK = "dense-mask"
    INSTANCE_MASK = "instance-mask"
    POINT_LABEL = "point-label"
    BBOX = "bbox"
    IMAGE_LABEL = "image-label"
    MULTILABEL = "multilabel"
    KEYPOINT = "keypoint"
    POLYLINE = "polyline"
    """Line annotations — e.g. scale bars, transect lines."""
    DEPTH_MAP = "depth-map"
    CAMERA_POSE = "camera-pose"
    POINT_CLOUD_LABEL = "point-cloud-label"
    TRACK = "track"
    AUDIO_EVENT = "audio-event"
    TABULAR = "tabular"


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    POINT_CLOUD = "point-cloud"
    MESH = "mesh"
    ORTHOMOSAIC = "orthomosaic"
    SATELLITE = "satellite"
    AUDIO = "audio"
    TABULAR = "tabular"
    SEQUENCE = "sequence"
    """Molecular sequence data (eDNA)."""


class Region(str, Enum):
    """Coarse biogeographic region.

    Deliberately coarse: the purpose is cross-region generalisation warnings, which is
    the field's most common deployment failure. Finer detail belongs in
    :attr:`~marinedata.models.Coverage.ecoregions` (MEOW names).
    """

    CARIBBEAN = "caribbean"
    WESTERN_ATLANTIC = "western-atlantic"
    EASTERN_ATLANTIC = "eastern-atlantic"
    RED_SEA = "red-sea"
    WESTERN_INDIAN = "western-indian"
    EASTERN_INDIAN = "eastern-indian"
    CORAL_TRIANGLE = "coral-triangle"
    CENTRAL_PACIFIC = "central-pacific"
    EASTERN_PACIFIC = "eastern-pacific"
    GBR_AUSTRALIA = "gbr-australia"
    MEDITERRANEAN = "mediterranean"
    TEMPERATE = "temperate"
    POLAR = "polar"
    DEEP_SEA = "deep-sea"
    GLOBAL = "global"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class Provenance(str, Enum):
    """Who produced the data. Orthogonal to licence: partner data may have no licence."""

    OWN = "own"
    PARTNER = "partner"
    PUBLIC = "public"
    AUTO = "auto"
    """Model-generated (pseudo-labels). Must never enter an evaluation split."""


class AccessMethod(str, Enum):
    HUGGINGFACE = "huggingface"
    ZENODO = "zenodo"
    S3 = "s3"
    HTTP = "http"
    GATED_FORM = "gated-form"
    """Registration or form gate — likely a contract. See LicenceFlags.contract_gated."""
    API = "api"
    REQUEST = "request"
    """Available only by contacting the maintainers."""
    SCRAPE = "scrape"
