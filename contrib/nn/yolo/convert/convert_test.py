"""Independent checks of the converter's weight layout and checkpoint encoding."""

import base64
import hashlib
import json
import struct
import unittest

import numpy as np
import torch

from convert import network_checkpoint


class NetworkCheckpointTests(unittest.TestCase):
    def test_kernel_layout_and_float32_bytes(self):
        conv = torch.nn.Conv2d(2, 3, 2)
        with torch.no_grad():
            conv.weight.copy_(torch.arange(24).reshape(3, 2, 2, 2))
            conv.bias.copy_(torch.tensor([-1.0, 0.0, 1.0]))
        encoded = network_checkpoint(torch.nn.Sequential(conv))
        self.assertEqual(encoded[:8], b"GOMLCP01")
        self.assertEqual(struct.unpack("<Q", encoded[8:16])[0], len(encoded) - 48)
        self.assertEqual(encoded[16:48], hashlib.sha256(encoded[48:]).digest())
        document = json.loads(encoded[48:])
        weight, bias = document["parameters"]
        self.assertEqual(weight["shape"], [8, 3])
        decoded = np.frombuffer(base64.b64decode(weight["data"]), dtype="<f4").reshape(
            8, 3
        )
        for channel in range(2):
            for row in range(2):
                for column in range(2):
                    for out in range(3):
                        self.assertEqual(
                            decoded[(channel * 2 + row) * 2 + column, out],
                            out * 8 + channel * 4 + row * 2 + column,
                        )
        np.testing.assert_array_equal(
            np.frombuffer(base64.b64decode(bias["data"]), dtype="<f4"), [-1, 0, 1]
        )

    def test_unsupported_grouped_dilated_or_biasless_convolution(self):
        for conv in (
            torch.nn.Conv2d(2, 2, 3, groups=2),
            torch.nn.Conv2d(2, 2, 3, dilation=2),
            torch.nn.Conv2d(2, 2, 3, bias=False),
        ):
            with self.subTest(conv=conv), self.assertRaises(ValueError):
                network_checkpoint(conv)

    def test_nonfinite_weights_and_unknown_modules_fail(self):
        conv = torch.nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            conv.weight.fill_(float("nan"))
        with self.assertRaises(ValueError):
            network_checkpoint(conv)
        with self.assertRaises(ValueError):
            network_checkpoint(torch.nn.ReLU())


if __name__ == "__main__":
    unittest.main()
