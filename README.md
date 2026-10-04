# Reflex

Reflex provides immutable CPU-side contracts for runtime-defined decisions, deterministic prompt rendering, candidate scoring, abstention, and option-order comparisons. It does not load a model or make model-quality claims.

Requires Python 3.12+ and uv. Install the locked environment and run the core checks:

```sh
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

The scoring and tokenizer checks use hand-built CPU fixtures. They do not establish compatibility with a real model runtime.
