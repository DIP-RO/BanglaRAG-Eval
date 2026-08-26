# Contributing

This repository contains research code and experimental artifacts for the Bangla RAG evaluation project.

## Branches

Do not work directly on `main`.

Use a separate branch for each feature, experiment, or documentation change.

Examples:

- `docs/research-protocol`
- `feature/annotation-schema`
- `feature/ragas-baseline`
- `experiment/pilot-001`
- `fix/metric-calculation`

## Commits

Keep commits focused on one change.

Use short messages that describe what changed.

Examples:

- `docs: update research protocol`
- `feat: add annotation schema`
- `exp: add pilot configuration`
- `fix: correct evaluation aggregation`
- `test: add metric tests`

Avoid combining unrelated changes in one commit.

## Pull Requests

Open a Pull Request when a change is ready for review.

A PR should briefly describe:

- what was changed;
- why the change was needed;
- how it was tested;
- which experiment or research question it supports.

Research-critical changes should be reviewed before being merged into `main`.

## Experiments

Every experiment should have a reproducible configuration and a unique experiment ID.

Record the dataset version, model/version, retrieval settings, evaluator settings, random seed where applicable, and relevant runtime information.

Do not overwrite previous experimental configurations.

## Data and Credentials

Do not commit:

- API keys;
- passwords or credentials;
- private datasets;
- restricted datasets;
- large generated outputs unless explicitly approved.

Check the licensing conditions of external datasets and models before redistribution.

## Research Practice

Results should not be changed or selectively reported simply to support a preferred hypothesis.

Unexpected, negative, and failed experiments should remain documented when they are relevant to the research process.
