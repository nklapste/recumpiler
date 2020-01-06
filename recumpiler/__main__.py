#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main argparse entrypoint"""

import argparse
from recumpiler.mutators import recumpile_text


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Input")
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="Text to recumpile.")
    group.add_argument(
        "--repl", help="Accept stdin and create recumpiler REPL.", action="store_true"
    )
    return parser


def recumpile_repl():
    while True:
        text = input("recumpiler>")
        fucked_text = recumpile_text(text)
        print(fucked_text)


def main():
    args = get_parser().parse_args()
    if args.text:
        fucked_text = recumpile_text(args.text)
        print(fucked_text)
    if args.repl:
        recumpile_repl()


if __name__ == "__main__":
    main()
