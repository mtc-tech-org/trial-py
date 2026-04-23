# Contributing to Trial

Thanks for your interest in contributing.

## Setup

```bash
git clone https://github.com/mtc-tech-org/trial-py.git
cd trial-py
pip install -e ".[all,dev]"
```

## Running tests

```bash
# Unit tests (no API key needed)
pytest tests/ -m "not integration"

# Integration tests
ANTHROPIC_API_KEY=sk-... pytest tests/ -m integration
```

With Docker:

```bash
docker compose run eval pytest tests/ -m "not integration"
```

## Making changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests — all unit tests must pass
4. Open a pull request

## What we're looking for

- New assertion types
- Additional provider adapters
- Bug fixes with a reproducing test case
- Documentation improvements

## What we're not looking for (yet)

- Plugin systems or registries
- Framework-specific integrations built into core
- Breaking changes to the public API without discussion

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- A minimal reproducible example
