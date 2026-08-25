# Tests

From the assistant root, with the venv active:

```bash
pytest              # default: offline unit tests
pytest -m live      # hits xAI TTS + official grok CLI (tiny spend)
pytest -k protocol  # one file / keyword
```

Default `addopts` in `pyproject.toml` excludes `live`. After changing tools, the CLI wrapper, or the Voice event schema, run `pytest` before relying on a voice session.
