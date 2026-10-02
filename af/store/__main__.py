"""``python -m af.store busy <store> [--clear NAME]``: list or clear batch markers."""

import sys

from af.store import busy

COMMANDS = {"busy": busy.main}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: python -m af.store {{{','.join(COMMANDS)}}} ...", file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
