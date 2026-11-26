"""Evaluation harness: prepare clips, run configurations, score against human labels.

Workflow (each step is a module runnable with ``python -m``):

    prepare   download and trim the clips named in eval/manifest.json
    runner    run every configuration over every clip; export blind labelling files
    report    score the runs against human labels; write tables, CSVs and plot data

Ground truth is never produced by this package. It comes from people watching
the clips in eval/labeler/index.html, which shows them no system output.
"""
