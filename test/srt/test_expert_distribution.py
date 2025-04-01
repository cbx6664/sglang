import glob
import os
import unittest

import requests
import torch
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestExpertDistribution(CustomTestCase):
    def setUp(self):
        # Clean up any existing expert distribution files before each test
        for f in glob.glob("expert_distribution_*.csv"):
            os.remove(f)

    def tearDown(self):
        # Clean up any expert distribution files after each test
        for f in glob.glob("expert_distribution_*.csv"):
            os.remove(f)

    def test_expert_distribution_record(self):
        for info in [
            dict(model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", mode_detail=False),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B", mode_detail=False),
            dict(model_path="Qwen/Qwen1.5-MoE-A2.7B", mode_detail=True),
        ]:
            with self.subTest(info=info):
                self._execute_core(**info)

    def _execute_core(self, model_path: str, mode_detail: bool):
        """Test expert distribution record endpoints"""
        os.environ["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DETAIL"] = "1" if mode_detail else "0"
        process = popen_launch_server(
            model_path,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
            ],
        )

        try:
            # Start recording
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/start_expert_distribution_record"
            )
            self.assertEqual(response.status_code, 200)

            # Make some requests to generate expert distribution data
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 3,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)

            # Stop recording
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/stop_expert_distribution_record"
            )
            self.assertEqual(response.status_code, 200)

            # Dump the recorded data
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/dump_expert_distribution_record"
            )
            self.assertEqual(response.status_code, 200)

            # Check data rows
            data = response.json()
            print(f"{data=}")

            if mode_detail:
                self.assertGreater(len(data), 0, "Should contain data rows")
            else:
                logical_count = torch.tensor(data['logical_count'])
                print(f"{logical_count=}")
                self.assertTrue(logical_count.sum() > 0)

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
