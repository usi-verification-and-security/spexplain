"""Tests for data-root resolution.  Like the rest of the suite, these read no
project data -- only files and directories the test itself creates."""

from __future__ import annotations

from pathlib import Path

import pytest

from explviz import paths
from explviz.paths import (ENV_VAR, data_path, data_root, describe_root,
                           explanations_path)


@pytest.fixture
def no_sibling(monkeypatch, tmp_path):
    """Point the sibling data/ candidate at somewhere that does not exist."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(paths, "SIBLING_DATA_DIR", tmp_path / "absent-data")
    return tmp_path


# ------------------------------------------------------------ resolution order

def test_falls_back_to_cwd_when_nothing_else_found(no_sibling):
    assert data_root() == Path(".")
    assert "current directory" in describe_root()


def test_sibling_data_dir_is_used_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)
    sibling = tmp_path / "data"
    sibling.mkdir()
    monkeypatch.setattr(paths, "SIBLING_DATA_DIR", sibling)
    assert data_root() == sibling
    assert "beside the package" in describe_root()


def test_env_var_wins_over_sibling(monkeypatch, tmp_path):
    sibling = tmp_path / "data"
    sibling.mkdir()
    override = tmp_path / "elsewhere"
    override.mkdir()
    monkeypatch.setattr(paths, "SIBLING_DATA_DIR", sibling)
    monkeypatch.setenv(ENV_VAR, str(override))
    assert data_root() == override
    assert ENV_VAR in describe_root()


def test_root_is_resolved_per_call(monkeypatch, tmp_path):
    """Creating the sibling folder later takes effect without reimporting."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    sibling = tmp_path / "data"
    monkeypatch.setattr(paths, "SIBLING_DATA_DIR", sibling)
    assert data_root() == Path(".")
    sibling.mkdir()
    assert data_root() == sibling


def test_sibling_is_anchored_on_the_package_not_the_cwd():
    """../data relative to explviz/, so it survives running from anywhere."""
    assert paths.SIBLING_DATA_DIR == paths.PACKAGE_DIR.parent / "data"
    assert paths.PACKAGE_DIR.name == "explviz"


# -------------------------------------------------------------------- lookups

def test_resolves_existing_file(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    (tmp_path / "datasets").mkdir()
    target = tmp_path / "datasets" / "points.csv"
    target.write_text("a,b\n1,2\n")
    assert data_path("datasets/points.csv") == target
    assert data_path("datasets", "points.csv") == target


def test_explanations_path_uses_the_explanations_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    (tmp_path / "explanations").mkdir()
    target = tmp_path / "explanations" / "itp.phi.txt"
    target.write_text("(and (<= x1 1))")
    assert explanations_path("itp.phi.txt") == target


def test_must_exist_false_skips_the_check(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert data_path("nowhere.csv", must_exist=False) == tmp_path / "nowhere.csv"
    assert explanations_path("no.phi.txt", must_exist=False) == \
        tmp_path / "explanations" / "no.phi.txt"


# ------------------------------------------------------------- error messages

def test_missing_file_message_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        data_path("datasets/absent.csv")
    msg = str(excinfo.value)
    assert "absent.csv" in msg
    assert ENV_VAR in msg                      # says how to change the root
    assert str(tmp_path) in msg                # says which root was used
    assert "data/" in msg                      # says about the sibling folder


def test_missing_file_message_without_env_var(no_sibling):
    with pytest.raises(FileNotFoundError) as excinfo:
        data_path("definitely/not/here.csv")
    msg = str(excinfo.value)
    assert "current directory" in msg
    assert ENV_VAR in msg
