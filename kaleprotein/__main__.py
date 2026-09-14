"""Command-line entry point for fetching standalone examples."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m kaleprotein")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-example", help="Download one example without cloning the library")
    download.add_argument("name", help="Example folder name, e.g. drugban_dti or mapdiff_inverse_folding")
    download.add_argument("--output", default="examples", help="Parent directory for the example (default: examples)")
    download.add_argument("--ref", help="Git branch, tag, or commit (default: installed version's release tag)")
    args = parser.parse_args(argv)

    from kaleprotein.utils import download_example

    try:
        path = download_example(args.name, output=args.output, ref=args.ref)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Downloaded example to {path}")
    print("Datasets, pretrained weights, and optional dependencies are not downloaded by this command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
