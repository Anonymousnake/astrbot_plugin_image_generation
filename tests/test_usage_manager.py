import os
import sys
import tempfile
import unittest
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ASTRBOT_ROOT_CANDIDATES = [
    Path(os.environ["ASTRBOT_ROOT"]) if os.environ.get("ASTRBOT_ROOT") else None,
    Path("/home/ubuntu/AstrBot"),
    Path(r"D:\Codex\AstrBot"),
]
for path in [PLUGIN_ROOT.parent, *ASTRBOT_ROOT_CANDIDATES]:
    if path and path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from astrbot_plugin_image_generation.core.config_manager import UsageSettings  # noqa: E402
from astrbot_plugin_image_generation.core.usage_manager import UsageManager  # noqa: E402


class UsageManagerTests(unittest.TestCase):
    def test_admin_bypass_does_not_record_daily_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UsageManager(
                tmpdir,
                UsageSettings(
                    enable_daily_limit=True,
                    daily_limit_count=1,
                    admin_bypass_limits=True,
                    rate_limit_seconds=0,
                ),
            )

            self.assertTrue(
                manager.check_rate_limit(
                    "group:unit",
                    is_admin=True,
                    requested_count=1,
                )
            )
            manager.settle_usage(
                "group:unit",
                is_admin=True,
                reserved_count=1,
                actual_count=1,
            )

            self.assertEqual(manager.get_usage_count("group:unit"), 0)

    def test_normal_user_records_daily_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = UsageManager(
                tmpdir,
                UsageSettings(
                    enable_daily_limit=True,
                    daily_limit_count=2,
                    admin_bypass_limits=True,
                    rate_limit_seconds=0,
                ),
            )

            self.assertTrue(
                manager.check_rate_limit(
                    "group:unit",
                    is_admin=False,
                    requested_count=1,
                )
            )
            manager.settle_usage(
                "group:unit",
                is_admin=False,
                reserved_count=1,
                actual_count=1,
            )

            self.assertEqual(manager.get_usage_count("group:unit"), 1)


if __name__ == "__main__":
    unittest.main()
