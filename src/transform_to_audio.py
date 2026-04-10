import sys
from utility import cli_main_process


def main() -> int:
    result = cli_main_process()
    return result if isinstance(result, int) else 0


if __name__ == '__main__':
    sys.exit(main())
