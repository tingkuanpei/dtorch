"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import argparse
import os
import unittest


class FileTrackingTestResult(unittest.TextTestResult):
    def startTest(self, test):
        try:
            module_name = test.__class__.__module__
            module = __import__(module_name, fromlist=["__file__"])
            test_file = os.path.basename(module.__file__)

            test_class_name = test.__class__.__name__
            test_method_name = test._testMethodName
            full_test_name = f"{test_class_name}.{test_method_name}"

            print(f"Executing test now: {test_file} -> {full_test_name}")
        except (ImportError, AttributeError):
            pass

        super().startTest(test)


SCOPE_CHOICES = ("base", "operators", "modules", "system_overhead", "all")
SCOPE_DIR_MAP = {
    "base": "",
    "operators": "operators",
    "modules": "modules",
    "system_overhead": "system_overhead",
}


def parse_scopes(argv: list[str] | None = None) -> list[str]:
    """Parse scope arguments from command line.

    Supports formats like:
      base operators          (multiple positional args)
      base+operators          (plus-separated)
      base,operators          (comma-separated)
      base+operators,modules   (mixed separators)

    Returns a list of canonical scope names.
    """
    parser = argparse.ArgumentParser(
        description="Run DTorch Python tests by scope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scopes (additive):\n"
            "  base       Tests directly under python/dtorch/test (no subdirectories)\n"
            "  operators  Tests under python/dtorch/test/operators\n"
            "  modules    Tests under python/dtorch/test/modules\n"
            "  system_overhead  Tests under python/dtorch/test/system_overhead\n"
            "  all        All tests (base + operators + modules + system_overhead)\n"
            "\nExamples:\n"
            "  %(prog)s                              # default: base + operators\n"
            "  %(prog)s base                         # base only\n"
            "  %(prog)s base operators modules       # all three\n"
            "  %(prog)s base+operators               # plus-separated\n"
            "  %(prog)s all                       # everything\n"
        ),
    )
    parser.add_argument(
        "scopes",
        nargs="*",
        metavar="SCOPE",
        help="Test scopes to run (default: base operators). Choices: %s" % ", ".join(SCOPE_CHOICES),
    )

    args = parser.parse_args(argv)

    # Normalize: split each arg on '+' or ',' to support combined forms
    raw: list[str] = []
    for token in args.scopes:
        # Split by '+' first, then by ',' in each chunk
        parts = []
        for chunk in token.split("+"):
            parts.extend(chunk.split(","))
        raw.extend(p.strip() for p in parts if p.strip())

    # Apply default if nothing given
    if not raw:
        raw = ["base", "operators"]

    # Validate choices
    invalid = [s for s in raw if s not in SCOPE_CHOICES]
    if invalid:
        parser.error("invalid scope(s): %s (choose from %s)" % (", ".join(invalid), ", ".join(SCOPE_CHOICES)))

    # Expand 'all' to every concrete scope
    if "all" in raw:
        return ["base", "operators", "modules", "system_overhead"]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for s in raw:
        if s not in seen:
            seen.add(s)
            result.append(s)

    return result


def main(argv: list[str] | None = None):
    scopes = parse_scopes(argv)
    testProjectRoot = os.path.dirname(__file__)

    runner = unittest.TextTestRunner()
    # runner = unittest.TextTestRunner(resultclass=FileTrackingTestResult)

    for scope in scopes:
        subdir = SCOPE_DIR_MAP[scope]
        path = os.path.join(testProjectRoot, subdir)
        loader = unittest.TestLoader()

        print("=" * 107)
        print(f"Running python3 tests from: {path}")
        print("=" * 107)
        print("\n")

        if scope == "base":
            # Only top-level test_*.py files — no subdirectory recursion (discover() recurses into packages)
            suite = unittest.TestSuite()
            for fname in sorted(os.listdir(path)):
                if fname.startswith("test_") and fname.endswith(".py"):
                    module_name = fname[:-3]  # strip .py
                    full_name = f"dtorch.test.{module_name}"
                    mod = __import__(full_name, fromlist=["__file__"])
                    suite.addTests(loader.loadTestsFromModule(mod))
        elif scope == "system_overhead":
            # system_overhead uses benchmark_*.py naming convention
            suite = loader.discover(path, pattern="benchmark_*.py")
        else:
            suite = loader.discover(path)

        result = runner.run(suite)
        print(f"RUN {result.testsRun} Tests. PASSED? {result.wasSuccessful()}")
        assert result.wasSuccessful()


# Usage:
#     python3 python/dtorch/test/run_all_test.py                               # default: base + operators
#     python3 python/dtorch/test/run_all_test.py base                          # base only
#     python3 python/dtorch/test/run_all_test.py base operators modules        # all three
#     python3 python/dtorch/test/run_all_test.py base+operators               # plus-separated
#     python3 python/dtorch/test/run_all_test.py base,modules               # comma-separated
#     python3 python/dtorch/test/run_all_test.py all                        # everything
# Run specific test: pytest -s python/dtorch/test/test_tensor.py::TestTensor::test_tensor_constructor_time
if __name__ == "__main__":
    main()
