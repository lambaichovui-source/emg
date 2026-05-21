import os
import tempfile
import pytest

from annotation_store import is_training_csv, SCHEMA_VERSION

def test_is_training_csv_missing_file():
    """Test that missing file returns False instead of crashing."""
    assert is_training_csv("this_file_does_not_exist_12345.csv") is False

def test_is_training_csv_valid_schema():
    """Test with a file containing the valid schema version."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(f"schema,{SCHEMA_VERSION}\n")
        f.write("some,other,data\n")
        temp_path = f.name

    try:
        assert is_training_csv(temp_path) is True
    finally:
        os.remove(temp_path)

def test_is_training_csv_valid_lowercase():
    """Test with a file containing 'ionm_training'."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("something,IONM_Training,something\n")
        temp_path = f.name

    try:
        assert is_training_csv(temp_path) is True
    finally:
        os.remove(temp_path)

def test_is_training_csv_invalid():
    """Test with a file that doesn't contain the expected strings."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("just,some,regular,data\n")
        f.write("nothing,special,here\n")
        temp_path = f.name

    try:
        assert is_training_csv(temp_path) is False
    finally:
        os.remove(temp_path)

def test_is_training_csv_empty_file():
    """Test with an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_path = f.name

    try:
        assert is_training_csv(temp_path) is False
    finally:
        os.remove(temp_path)
