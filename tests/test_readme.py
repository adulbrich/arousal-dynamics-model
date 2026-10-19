"""The README's Python examples run, in order, as one script."""

import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_examples_run() -> None:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)
    assert blocks, "README has no Python examples"
    namespace: dict[str, object] = {}
    exec(compile("\n".join(blocks), str(README), "exec"), namespace)  # noqa: S102
    assert "variant" in namespace
