"""
tests/run_tests.py — Zero-dependency test runner.

Discovers every tests/test_*.py module, runs every top-level `test_*` callable, and
reports pass/fail counts per module. Exists so the suite runs on a bare Python
install; the same test functions are plain asserts and run under pytest unchanged.

    python tests/run_tests.py            run everything
    python tests/run_tests.py alignment  run only modules matching 'alignment'
"""

import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    pattern = argv[0] if argv else ""
    files = sorted(p for p in (ROOT / "tests").glob("test_*.py") if pattern in p.stem)
    if not files:
        print(f"No test modules matching {pattern!r}")
        return 1

    total_pass = total_fail = 0
    failures: list[tuple[str, str, str]] = []

    for path in files:
        try:
            module = load_module(path)
        except Exception:
            print(f"\n{path.stem}\n  ✗ MODULE IMPORT FAILED")
            traceback.print_exc()
            total_fail += 1
            failures.append((path.stem, "<import>", traceback.format_exc(limit=3)))
            continue

        tests = [
            (name, obj)
            for name, obj in sorted(vars(module).items())
            if name.startswith("test_") and callable(obj)
        ]
        if not tests:
            continue

        print(f"\n{path.stem}")
        for name, fn in tests:
            try:
                fn()
                print(f"  ✓ {name}")
                total_pass += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                total_fail += 1
                failures.append((path.stem, name, traceback.format_exc(limit=4)))

    print(f"\n{'─' * 60}")
    print(f"{total_pass} passed, {total_fail} failed")
    if failures:
        print(f"\nFailure detail:")
        for mod, name, tb in failures:
            print(f"\n── {mod}::{name} ──\n{tb}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
