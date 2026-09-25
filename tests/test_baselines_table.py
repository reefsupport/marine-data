"""Card table formatting (design §4.5)."""

from marinedata.eval.baselines.table import render_card_table


def test_render_card_table_has_header_and_id_ood_gap_row():
    rows = [
        {
            "task": "bleaching-condition",
            "backbone": "dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "config": "abc123",
            "splits": {
                "test": {"macro_f1": 0.73, "n_images": 400},
                "ood-geo-temperate-australasia": {"macro_f1": 0.61, "n_images": 120},
            },
        }
    ]
    md = render_card_table(rows)
    assert md.startswith("| task | split | metric | method")
    assert "bleaching-condition | test | macro_f1" in md
    assert "id-vs-ood-gap" in md
    assert "0.7 |" in md or "0.73" in md


def test_render_card_table_handles_no_ood_split():
    rows = [
        {
            "task": "source-id-domain",
            "backbone": "dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "config": "abc123",
            "splits": {"test": {"macro_f1": 0.99, "n_images": 400}},
        }
    ]
    md = render_card_table(rows)
    assert "id-vs-ood-gap" not in md
