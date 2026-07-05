import asyncio
import os
import sys
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

from astrbot_plugin_image_generation.core.task_manager import (  # noqa: E402
    GenerationTaskStatus,
    TaskManager,
)


def _create_task(manager: TaskManager, task_id: str, coro):
    return manager.create_generation_task(
        coro,
        task_id=task_id,
        source="unit",
        unified_msg_origin="group:unit",
        prompt=f"prompt {task_id}",
        reference_image_count=0,
        requested_count=1,
        aspect_ratio="1:1",
        resolution="1024x1024",
    )


class TaskManagerQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        if hasattr(self, "manager"):
            await self.manager.cancel_all()

    async def test_accepts_one_running_one_queued_and_rejects_third(self) -> None:
        self.manager = TaskManager()
        first_started = asyncio.Event()
        first_release = asyncio.Event()
        second_started = asyncio.Event()
        second_release = asyncio.Event()

        async def first_job():
            first_started.set()
            await first_release.wait()

        async def second_job():
            second_started.set()
            await second_release.wait()

        result1 = _create_task(self.manager, "task-1", first_job())
        await asyncio.wait_for(first_started.wait(), timeout=1)
        result2 = _create_task(self.manager, "task-2", second_job())
        result3 = _create_task(self.manager, "task-3", second_job())

        self.assertTrue(result1.accepted)
        self.assertTrue(result2.accepted)
        self.assertFalse(result3.accepted)
        self.assertEqual(result3.message, "队列已满")
        self.assertIsNone(self.manager.get_generation_task("task-3"))
        self.assertEqual(result1.record.status, GenerationTaskStatus.RUNNING)
        self.assertEqual(result2.record.status, GenerationTaskStatus.QUEUED)
        self.assertFalse(second_started.is_set())

        first_release.set()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        self.assertEqual(result2.record.status, GenerationTaskStatus.RUNNING)
        second_release.set()

    async def test_cancel_queued_task_removes_it_without_stopping_running_task(self) -> None:
        self.manager = TaskManager()
        first_started = asyncio.Event()
        first_release = asyncio.Event()
        second_started = asyncio.Event()

        async def first_job():
            first_started.set()
            await first_release.wait()

        async def second_job():
            second_started.set()

        result1 = _create_task(self.manager, "task-1", first_job())
        await asyncio.wait_for(first_started.wait(), timeout=1)
        result2 = _create_task(self.manager, "task-2", second_job())

        ok, _ = self.manager.cancel_generation_task("task-2")

        self.assertTrue(ok)
        self.assertEqual(result1.record.status, GenerationTaskStatus.RUNNING)
        self.assertEqual(result2.record.status, GenerationTaskStatus.CANCELLED)
        self.assertFalse(second_started.is_set())

        first_release.set()
        await asyncio.sleep(0)
        self.assertEqual(result1.record.status, GenerationTaskStatus.SUCCEEDED)

    async def test_cancel_all_does_not_start_queued_tasks(self) -> None:
        self.manager = TaskManager()
        first_started = asyncio.Event()
        first_release = asyncio.Event()
        second_started = asyncio.Event()

        async def first_job():
            first_started.set()
            await first_release.wait()

        async def second_job():
            second_started.set()

        result1 = _create_task(self.manager, "task-1", first_job())
        await asyncio.wait_for(first_started.wait(), timeout=1)
        result2 = _create_task(self.manager, "task-2", second_job())

        await self.manager.cancel_all()
        await asyncio.sleep(0)

        self.assertEqual(result1.record.status, GenerationTaskStatus.CANCELLED)
        self.assertEqual(result2.record.status, GenerationTaskStatus.CANCELLED)
        self.assertFalse(second_started.is_set())


if __name__ == "__main__":
    unittest.main()
