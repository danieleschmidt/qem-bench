"""Tests for CLI functionality"""

import pytest
from qem_bench.cli import main, get_version, handle_command
import argparse


def test_get_version():
    """Test version retrieval"""
    version = get_version()
    assert isinstance(version, str)
    assert version is not None


def test_main_no_args(capsys):
    """Test main with no arguments shows help"""
    result = main([])
    captured = capsys.readouterr()
    assert result == 1
    assert "usage:" in captured.out.lower()


def test_main_version(capsys):
    """Test version flag"""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_benchmark_command():
    """Test benchmark command handling"""
    args = argparse.Namespace(command="benchmark", method="zne", backend="simulator")
    result = handle_command(args)
    assert result == 0


def test_generate_command():
    """Test generate command handling"""  
    args = argparse.Namespace(command="generate", type="qv", qubits=5)
    result = handle_command(args)
    assert result == 0


def test_unknown_command():
    """Test unknown command handling"""
    args = argparse.Namespace(command="invalid")
    result = handle_command(args)
    assert result == 1