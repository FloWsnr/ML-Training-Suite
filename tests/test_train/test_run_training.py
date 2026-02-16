"""Tests for run_training utility functions."""

from pathlib import Path

import pytest

from ml_suite.train.run_training import time_str_to_seconds, get_checkpoint_path


class TestTimeStrToSeconds:
    def test_basic(self):
        assert time_str_to_seconds("1:00:00") == 3600.0

    def test_minutes_only(self):
        assert time_str_to_seconds("0:30:00") == 1800.0

    def test_seconds_only(self):
        assert time_str_to_seconds("0:00:45") == 45.0

    def test_combined(self):
        assert time_str_to_seconds("2:30:15") == 2 * 3600 + 30 * 60 + 15


class TestGetCheckpointPath:
    def test_latest(self, tmp_path: Path):
        assert get_checkpoint_path(tmp_path, "latest") == tmp_path / "latest.pt"

    def test_best(self, tmp_path: Path):
        assert get_checkpoint_path(tmp_path, "best") == tmp_path / "best.pt"

    def test_epoch_number(self, tmp_path: Path):
        assert get_checkpoint_path(tmp_path, "5") == tmp_path / "epoch_0005" / "checkpoint.pt"

    def test_epoch_number_int(self, tmp_path: Path):
        assert get_checkpoint_path(tmp_path, 5) == tmp_path / "epoch_0005" / "checkpoint.pt"

    def test_invalid_name_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid checkpoint name"):
            get_checkpoint_path(tmp_path, "unknown")
