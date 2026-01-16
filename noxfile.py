"""Nox sessions for multi-version Python testing."""

import nox

nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install("pytest", "pytest-cov")

    # Run with coverage only for Python 3.12 (CI will use this for Codecov)
    if session.python == "3.12":
        session.run(
            "pytest",
            "tests/",
            "-v",
            "--cov=ccd2rdmol",
            "--cov-report=term-missing",
            "--cov-report=xml",
        )
    else:
        session.run("pytest", "tests/", "-v")
