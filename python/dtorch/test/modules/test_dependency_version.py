"""
Copyright 2026 The DTorch Authors. All rights reserved.

Author: Tingkuan Pei(contact: peitingkuan@163.com)
"""

import unittest

import transformers
import diffusers


class TestDependencyVersion(unittest.TestCase):
    def test_transformers_version(self):
        self.assertEqual(transformers.__version__, "4.53.1")

    def test_diffusers_version(self):
        self.assertEqual(diffusers.__version__, "0.34.0")


if __name__ == "__main__":
    unittest.main()
