"""WP-U10: the unified ``vqa`` / ``captions`` tables and their producers (offline, stubs)."""

# ruff: noqa: E501

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.annotation_schema import AnnotationSchemaError, validate_rows
from marinedata.task_layers import configs
from marinedata.task_layers import vqa_table as vt
from marinedata.task_layers.captions_table import (
    TEMPLATE_DETAIL,
    captions_from_frame,
    write_captions,
)
from marinedata.task_layers.producers import coralvqa_unified

SHA_A, SHA_B = "ab" * 32, "cd" * 32
MCQ = "How does the diver act?\nA. Touches the turtle\nB. Watches the turtle\nC. Feeds it\nPlease output only the letter."


def _fake(files: dict[str, bytes]):
    def fetch(key: str) -> bytes:
        try:
            return files[key]
        except KeyError:
            raise OSError(f"no such key {key}") from None

    return fetch


def _jsonl(*records: dict) -> bytes:
    return ("\n".join(json.dumps(r) for r in records) + "\n").encode()


def _turns(question: str, answer: str) -> list[dict]:
    return [
        {"from": "human", "value": f"<image>\n[vqa] {question}"},
        {"from": "gpt", "value": answer},
    ]


def test_sources_never_claim_human_for_machine_text():
    for spec in vt.VQA_SOURCES.values():
        assert spec.annotator in {"pseudo", "model"} and spec.annotator_detail.split(":")[0] in {
            "template",
            "vlm",
        }
        assert spec.licence_class in {"open", "restricted-nc", "internal-only", "unknown"}
        row = vt.vqa_row(spec=spec, ordinal=0, sha=SHA_A, question="q?", answer="a", qa_type="t")
        assert (
            row["annotator_type"] == spec.annotator
            and row["annotator_detail"] == spec.annotator_detail
        )
        assert json.loads(row["attrs"])["licence_class"] == spec.licence_class


def test_parse_options_needs_consecutive_lettered_choices():
    assert vt.parse_options(MCQ) == [
        "A. Touches the turtle",
        "B. Watches the turtle",
        "C. Feeds it",
    ]
    assert vt.parse_options("How many corals?") == []
    assert vt.parse_options("Q\nA. only one") == []
    assert vt.parse_options("Q\nA. one\nC. skipped B") == []


_INSTR = "\nPlease output only the letter and corresponding term (e.g., A. Mutualism), with no additional explanation."
MARINEEVT_FORMATS = {  # every layout seen in the staged MarineEVT question files (6,175 MC questions)
    "one choice per line": (MCQ, ["A. Touches the turtle", "B. Watches the turtle", "C. Feeds it"]),
    "inline, no instruction line": (
        "Why does the flatfish bury itself in the sand?\n"
        "A. To attract mates B. To avoid sunlight C. To ambush prey using camouflage D. To rest",
        ["A. To attract mates", "B. To avoid sunlight", "C. To ambush prey using camouflage",
         "D. To rest"],
    ),
    "inline + instruction line quoting an option": (
        "What is the next action of the octopus?\n"
        "A. It changes color B. It jets away rapidly using siphon propulsion C. It hides"
        "\nPlease output only the letter and corresponding term (e.g., A. It changes color), "
        "with no additional explanation.",
        ["A. It changes color", "B. It jets away rapidly using siphon propulsion", "C. It hides"],
    ),
    "inline, leading space, eight choices": (
        "What are the ecological relationship of those two species in the video?\n"
        " A. Mutualism B. Commensalism C. Parasitism D. Predation E. Competition F. Herbivory "
        "G. Amensalism H. Neutralism" + _INSTR,
        ["A. Mutualism", "B. Commensalism", "C. Parasitism", "D. Predation", "E. Competition",
         "F. Herbivory", "G. Amensalism", "H. Neutralism"],
    ),
    "inline, source typo (D. twice, trailing comma)": (
        "What is the current life stage of the species shown in the videos?\n"
        " A. Egg/Unborn B. Larval C. Juvenile D. Sub-adult D. Adult, E. Senescent \n" + _INSTR,
        ["A. Egg/Unborn", "B. Larval", "C. Juvenile", "D. Sub-adult D. Adult", "E. Senescent"],
    ),
    "inline with parenthesis markers": ("Q\nA) one B) two", ["A. one", "B. two"]),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(MARINEEVT_FORMATS))
def test_parse_options_reads_every_marineevt_layout(name):
    question, want = MARINEEVT_FORMATS[name]
    assert vt.parse_options(question) == want


def test_parse_options_inline_needs_an_a_marker_at_the_line_start():
    assert vt.parse_options("Is it plan B. Or plan C. maybe?") == []
    assert vt.parse_options("Q\nA. alone\nstem mentions B. x") == []
    assert vt.parse_options("Q\nB. one C. two") == []


def test_vqa_row_validates_and_rejects_bad_rows():
    spec = vt.VQA_SOURCES["coralvqa"]
    row = vt.vqa_row(
        spec=spec,
        ordinal=3,
        sha=SHA_A,
        question=" q? ",
        answer=" a ",
        qa_type="vqa",
        split="train",
        attrs={"answer_type": "short-answer"},
    )
    assert (row["question"], row["answer"], row["ann_id"], row["upstream_split"]) == (
        "q?",
        "a",
        "coralvqa:3",
        "train",
    )
    validate_rows("vqa", [row])
    with pytest.raises(AnnotationSchemaError):
        validate_rows("vqa", [{**row, "answer": ""}])
    with pytest.raises(AnnotationSchemaError):
        validate_rows("vqa", [{**row, "annotator_type": "robot"}])


def test_pending_rows_validate_against_a_placeholder_sha(tmp_path):
    spec = vt.VQA_SOURCES["uwbench"]
    row = vt.vqa_row(
        spec=spec, ordinal=1, sha=None, question="q?", answer="a", qa_type="object category"
    )
    pending = {
        **{k: v for k, v in row.items() if k != "image_sha256"},
        "image_key": "images_test_zip_X",
    }
    assert vt.validate_pending("vqa", [pending]) == []
    assert vt.validate_pending("vqa", [{**pending, "image_key": ""}])
    path = vt.pending_path(tmp_path, "vqa", "uwbench", spec.version)
    assert path.name.endswith(".pending.parquet") and vt.write_pending("vqa", path, [pending]) == 1
    cols = pq.read_table(path).column_names
    assert "image_key" in cols and "image_sha256" not in cols


def test_coralvqa_train_rows_resolve_sha_and_test_split_is_skipped():
    spec = vt.VQA_SOURCES["coralvqa"]
    base = f"sources/coralvqa/{spec.version}/"
    train = _jsonl(
        {"id": "CoralVQA", "image": "1.jpg", "conversations": _turns("How many genera?", "Three.")},
        {
            "id": "CoralVQA",
            "image": "9.jpg",
            "conversations": _turns("Which genus?", "Acropora"),
        },  # image not staged
        {
            "id": "CoralVQA",
            "image": "1.jpg",
            "conversations": [{"from": "human", "value": "<image>\n[vqa] No answer?"}],
        },
    )
    test = _jsonl({"question_id": 1, "image": "1.jpg", "text": "Q?", "category": "coral color"})
    files = {
        base + "CHECKSUMS.sha256": f"{SHA_A}  images/CoralVQA_Image_zip_1.jpg\n".encode(),
        base + coralvqa_unified.TRAIN: train,
        base + coralvqa_unified.TEST: test,
    }
    res = vt.staged_vqa(spec, fetch=_fake(files))
    assert len(res.rows) == 1 and res.pending == []
    row = res.rows[0]
    assert (row["image_sha256"], row["question"], row["answer"], row["qa_type"]) == (
        SHA_A,
        "How many genera?",
        "Three.",
        "vqa",
    )
    assert (
        row["ann_id"] == "coralvqa:0"
        and row["upstream_split"] == "train"
        and row["annotator_type"] == "pseudo"
    )
    assert res.skipped["image not staged"] == 1 and res.skipped["no question or answer"] == 1
    assert sum(v for k, v in res.skipped.items() if k.startswith("test split")) == 1
    validate_rows("vqa", res.rows)


def test_marineevt_video_qa_is_pending_with_options_and_flags():
    spec = vt.VQA_SOURCES["marineevt"]
    base = f"sources/marineevt/{spec.version}/"
    grp = "test_CasualReasoning_Human-SpeciesCasualDynamics"
    doc = {
        "data": [
            {
                "question": MCQ,
                "answer": "B. Watches the turtle",
                "question_format": "MultipleChoiceQuestion",
                "video_id": "VID1",
                "video_scene": "000023_000027",
                "dimension": "CasualReasoning",
                "subdimension": "Human-SpeciesCasualDynamics",
                "has_hallucination": False,
            },
            {
                "question": "Open?",
                "answer": "Yes",
                "question_format": "Open-endQuestion",
                "video_id": "VID2",
                "dimension": "d",
                "subdimension": "s",
            },  # no staged frame
            {"question": "Gone?", "answer": "Yes", "video_id": "VID1", "missing_video": True},
        ]
    }
    files = {
        base + "CHECKSUMS.sha256": (
            f"{SHA_A}  images/{grp}_videos_zip_videos_VID1_0abc.jpg\n{SHA_B}  images/{grp}_videos_zip_videos_VID1_frames_frame_000001.jpg\n"
        ).encode(),
        base + f"labels/files/{grp}_multi_turn_data_ver2.json": json.dumps(doc).encode(),
    }
    res = vt.staged_vqa(
        spec,
        fetch=_fake(files),
        lister=lambda prefix: sorted(k for k in files if k.startswith(prefix)),
    )
    assert res.rows == [] and len(res.pending) == 1
    row = res.pending[0]
    attrs = json.loads(row["attrs"])
    assert row["image_key"] == f"{grp}/VID1" and "image_sha256" not in row
    assert (
        attrs["answer_type"] == "multiple-choice"
        and attrs["frames_staged"] == 2
        and attrs["options"][1] == "B. Watches the turtle"
    )
    assert attrs["has_hallucination"] is False and attrs["licence_class"] == "open"
    assert (row["annotator_type"], row["upstream_split"], row["qa_type"]) == (
        "model",
        "test",
        "CasualReasoning/Human-SpeciesCasualDynamics",
    )
    assert (
        res.skipped["video has no staged frame"] == 1
        and res.skipped["no answer, no question or missing_video"] == 1
    )
    assert vt.validate_pending("vqa", res.pending) == []


def test_uwbench_pending_only_for_staged_images_and_ordinal_is_question_id():
    spec = vt.VQA_SOURCES["uwbench"]
    base = f"sources/uwbench/{spec.version}/"
    recs = {
        0: {
            "image_id": "AUV1490004.jpg",
            "question": "What is the large mollusk?",
            "ground_truth": "abalone",
            "question_id": "7",
            "type": "object category",
        },
        1: {
            "image_id": "Snake1150125.jpeg",
            "question": "Where?",
            "ground_truth": "center",
            "question_id": "8",
            "type": "object position",
        },  # not staged
    }
    files = {
        base + f"labels/files/unresolved-{n}.json": json.dumps(r).encode() for n, r in recs.items()
    }
    files[base + "images/images_test_zip_AUV1490004.jpg"] = b""
    res = vt.staged_vqa(
        spec,
        fetch=_fake(files),
        lister=lambda prefix: sorted(k for k in files if k.startswith(prefix)),
    )
    assert res.rows == [] and [r["ann_id"] for r in res.pending] == ["uwbench:7"]
    row = res.pending[0]
    assert row["image_key"] == "images_test_zip_AUV1490004" and row["upstream_split"] == "test"
    assert (
        row["annotator_type"],
        row["ann_license"],
        json.loads(row["attrs"])["licence_class"],
    ) == ("model", None, "internal-only")
    assert res.skipped["image not staged"] == 1 and res.seen == 2
    assert vt.validate_pending("vqa", res.pending) == []


def _wp13_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_sha256": [SHA_A, SHA_B],
            "caption_template": ["A reef photo.", ""],
            "caption_facts": ["{}", "{}"],
            "caption_vlm": [None, "A photo of a coral."],
            "vlm_model": [None, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"],
            "vlm_revision": [None, "main"],
            "prompt_sha256": [None, "deadbeef"],
            "seed": pd.array([None, 7], dtype="Int64"),
            "label_origin": ["derived", "model"],
            "consistency_flags": ["[]", '["depth"]'],
            "audit_verdict": [None, "consistent"],
        }
    )


def test_wp13_captions_map_to_pseudo_template_and_model_vlm(tmp_path):
    rows = captions_from_frame(
        _wp13_frame(),
        source_id="wp13-captions",
        source_version="v1",
        licence_class="open",
        splits={SHA_A: "train"},
    )
    assert [(r["image_sha256"], r["annotator_type"]) for r in rows] == [
        (SHA_A, "pseudo"),
        (SHA_B, "model"),
    ]  # empty template dropped
    assert rows[0]["annotator_detail"] == TEMPLATE_DETAIL and rows[0]["upstream_split"] == "train"
    assert rows[1]["annotator_detail"] == "vlm:mlx-community/Qwen2.5-VL-3B-Instruct-4bit@main"
    attrs = json.loads(rows[1]["attrs"])
    assert (
        attrs["prompt_sha256"] == "deadbeef"
        and attrs["seed"] == 7
        and attrs["consistency_flags"] == ["depth"]
        and attrs["licence_class"] == "open"
    )
    assert all(
        r["caption_type"] == "caption" and r["lang"] == "en" and r["annotator_type"] != "human"
        for r in rows
    )
    validate_rows("captions", rows)
    assert write_captions(tmp_path, "wp13-captions", "v1", rows).is_file()
    with pytest.raises(ValueError, match="not a WP-13 captions frame"):
        captions_from_frame(
            _wp13_frame().drop(columns=["audit_verdict"]), source_id="s", source_version="v"
        )


def test_configs_read_unified_vqa_and_captions_and_keep_the_legacy_fallback(tmp_path):
    spec = vt.VQA_SOURCES["coralvqa"]
    row = vt.vqa_row(
        spec=spec, ordinal=0, sha=SHA_A, question="q?", answer="a", qa_type="vqa", split="train"
    )
    vt.write_vqa(tmp_path, "coralvqa", spec.version, [row])
    pend = {
        **{k: v for k, v in row.items() if k != "image_sha256"},
        "ann_id": "coralvqa:1",
        "image_key": "k",
    }
    vt.write_pending("vqa", vt.pending_path(tmp_path, "vqa", "coralvqa", spec.version), [pend])
    write_captions(
        tmp_path,
        "wp13-captions",
        "v1",
        captions_from_frame(_wp13_frame(), source_id="wp13-captions", source_version="v1"),
    )
    vqa = configs.build_vqa_config(tmp_path)
    assert [
        (r["image_sha256"], r["sha256"], r["licence_class"], r["annotator_type"]) for r in vqa.rows
    ] == [(SHA_A, SHA_A, "restricted-nc", "pseudo")]  # pending never reaches the config
    caps = configs.build_captions_config(tmp_path)
    assert len(caps.rows) == 2 and {r["licence_class"] for r in caps.rows} == {"unknown"}
    assert "captions" in configs.CONFIG_IDS and set(configs.VQA_CONFIG_SOURCES) == set(
        vt.VQA_SOURCES
    )
    assert configs.build_captions_config(tmp_path / "nothing").rows == ()
    # no unified file: the D-Z2 tasklabels payload of coralvqa is read unchanged
    base = tmp_path / "legacy"
    (base / "_tasklabels" / "coralvqa").mkdir(parents=True)
    old = {
        "sha256": [SHA_B],
        "source_id": ["coralvqa"],
        "label_origin": ["human"],
        "question": ["lq"],
        "answer": ["la"],
        "qtype": ["vqa"],
        "split_hint": ["train"],
    }
    pq.write_table(pa.table(old), base / "_tasklabels" / "coralvqa" / "vqa.parquet")
    assert [(r["sha256"], r["question"]) for r in configs.build_vqa_config(base).rows] == [
        (SHA_B, "lq")
    ]


def test_uwbench_lists_only_images_and_labels_and_stops_at_limit():
    spec = vt.VQA_SOURCES["uwbench"]
    base = f"sources/uwbench/{spec.version}/"
    files = {
        base + f"labels/files/unresolved-{n}.json": json.dumps(
            {"image_id": f"A{n}.jpg", "question": "q?", "ground_truth": "a",
             "question_id": str(n), "type": "t"}
        ).encode()
        for n in range(600)
    }  # fmt: skip
    files |= {base + f"images/images_test_zip_A{n}.jpg": b"" for n in range(600)}  # all staged
    listed: list[str] = []
    fetched: list[str] = []

    def lister(prefix):
        listed.append(prefix)
        return sorted(k for k in files if k.startswith(prefix))

    def fetch(key):
        fetched.append(key)
        return files[key]

    res = vt.staged_vqa(spec, limit=10, fetch=fetch, lister=lister)
    assert listed == [base + "images/", base + "labels/files/"]  # never the whole prefix
    assert len(res.pending) == 10 and len(set(r["ann_id"] for r in res.pending)) == 10
    assert len(fetched) < 600 and res.seen == len(fetched)  # stopped after one batch
    ids = sorted(int(r["ann_id"].split(":")[1]) for r in res.pending)
    assert ids[-1] - ids[0] > 100  # spread over the label range, not the first files
    from marinedata.task_layers.producers import uwbench_vqa

    late = uwbench_vqa.staged(spec, limit=10, fetch=fetch, lister=lister, budget_s=-1)
    # the scan window is limit * SCAN spread labels, not all 600: the budget skips what is left of it
    assert late.pending == [] and late.skipped["read time budget reached"] == 10 * uwbench_vqa.SCAN


def test_uwbench_flaky_label_fetch_is_skipped_not_fatal():
    spec = vt.VQA_SOURCES["uwbench"]
    base = f"sources/uwbench/{spec.version}/"
    files = {
        base + "labels/files/unresolved-1.json": b"{}",
        base + "images/images_test_zip_X.jpg": b"",
    }

    def boom(key):
        raise TimeoutError(key)

    res = vt.staged_vqa(
        spec, fetch=boom, lister=lambda prefix: sorted(k for k in files if k.startswith(prefix))
    )
    assert res.pending == [] and res.skipped["label file fetch failed"] == 1
