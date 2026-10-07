"""WP-8d task-label producers: one module per source, each writing
``data/_tasklabels/<source_id>/<task>.parquet`` per the charter's D-Z2 fixed columns
(sha256, source_id, label_origin, plus the task payload). See
``docs/task-labels-producers.md`` for what each module does and why.
"""
