"""`python -m episodic_workflow` entry point."""

from __future__ import annotations

if __package__:
    from .cli import main
else:
    # Running the checkout directory directly requires no install or PYTHONPATH.
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
