import unittest

import main as root_main
from scripts import run_recommend, run_v31_final_recommend


class CurrentEntrypointTests(unittest.TestCase):
    def test_root_and_stable_runner_use_current_v31_main(self):
        self.assertIs(root_main.main, run_v31_final_recommend.main)
        self.assertIs(run_recommend.main, run_v31_final_recommend.main)


if __name__ == "__main__":
    unittest.main()
