# Contributing

We welcome contributions! This is an experimental research fork of LangChain
with new self-improving agent capabilities.

## Development Setup

```bash
pip install -e libs/core
pip install -e libs/langchain
```

## Code Style

- Follow LangChain's existing patterns (type hints, docstrings, Pydantic models)
- Use `typing_extensions` for `override` decorator
- All new public API must be exported in the relevant `__init__.py`
- New components in `langchain_core` should have minimal dependencies

## Testing

```bash
python -m pytest libs/langchain/tests/unit_tests/agents/test_meta_reflect_core.py -v
```

## Pull Request Process

1. Ensure tests pass
2. Update RESEARCH.md if your change is research-backed
3. Update README.md quickstart if applicable
4. Make sure `__all__` exports are complete in all modified `__init__.py` files
