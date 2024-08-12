import nox
import argparse

@nox.session(reuse_venv=True)
def docs(session: nox.session) -> None:
    """
    Build the documentation.
    Pass --nox-interactive to avoid blocking the terminal.
    First position argument is the target directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build targer, defualt: html."
    )
    parser.add_argument("output", nargs="?", help="Output directory.")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e .[docs]", "sphinx-autobuild")

    shared_args = (
        "-n", # nitpicky mode
        "-W", # turn warnings into errors
        "-T", # show full traceback on exception
        f"-b {args.builder}",
        "docs",
        args.output or "docs/_build/{args.builder}",
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)