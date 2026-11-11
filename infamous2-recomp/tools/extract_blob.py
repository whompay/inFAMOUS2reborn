"""Extract an exact byte range from a binary file.

Used for raw SPU policy modules embedded in a PPU ELF, which are not themselves
ELF images and therefore are intentionally skipped by extract_spu_images.py.
"""

import argparse
from pathlib import Path


def number(value: str) -> int:
    return int(value, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset", type=number, required=True)
    parser.add_argument("--length", type=number, required=True)
    args = parser.parse_args()

    data = args.input.read_bytes()
    end = args.offset + args.length
    if args.offset < 0 or args.length <= 0 or end > len(data):
        raise SystemExit(
            f"range 0x{args.offset:X}+0x{args.length:X} is outside "
            f"{args.input} (0x{len(data):X} bytes)"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data[args.offset:end])
    print(f"wrote {args.output}: 0x{args.length:X} bytes from 0x{args.offset:X}")


if __name__ == "__main__":
    main()
