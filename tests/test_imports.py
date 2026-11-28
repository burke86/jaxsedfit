import ast
from pathlib import Path

from jaxsedfit import FitConfig


def test_imports():
    assert FitConfig is not None


def test_package_has_no_jaxqsofit_imports():
    """Keep the reusable model dependency directed toward jaxsedfit."""
    package_dir = Path(__file__).parents[1] / "src" / "jaxsedfit"
    for source_path in package_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in node.names}
                assert "jaxqsofit" not in imported, source_path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".", 1)[0] != "jaxqsofit", source_path


def test_importing_jaxsedfit_enables_float64():
    import subprocess
    import sys

    code = "import jax, jaxsedfit, jax.numpy as jnp; assert jnp.asarray(1.0).dtype == jnp.float64"
    subprocess.run([sys.executable, "-c", code], check=True)
