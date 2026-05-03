"""Tests for local model backend — uses mocks, no real model downloads."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from pulse.local_model import LocalModelBackend, parse_structured_output


# --- parse_structured_output tests ---

def test_parse_direct_json():
    result = parse_structured_output('{"key": "value", "num": 1.5}')
    assert result == {"key": "value", "num": 1.5}


def test_parse_json_in_code_fence():
    text = 'Some preamble\n```json\n{"key": "value"}\n```\nMore text'
    result = parse_structured_output(text)
    assert result == {"key": "value"}


def test_parse_json_in_bare_fence():
    text = '```\n{"key": 42}\n```'
    result = parse_structured_output(text)
    assert result == {"key": 42}


def test_parse_embedded_json():
    text = 'The result is {"healthy_crop": 0.5, "disease": -0.3} based on analysis'
    result = parse_structured_output(text)
    assert result == {"healthy_crop": 0.5, "disease": -0.3}


def test_parse_no_json_returns_none():
    result = parse_structured_output("Just plain text with no JSON at all")
    assert result is None


def test_parse_empty_string():
    result = parse_structured_output("")
    assert result is None


# --- LocalModelBackend tests ---

def test_backend_not_loaded_by_default():
    backend = LocalModelBackend()
    assert not backend.is_loaded
    assert backend.model_id is None


def test_backend_loaded_with_injected_model():
    model = MagicMock()
    processor = MagicMock()
    backend = LocalModelBackend(model=model, processor=processor, model_id="test-model")
    assert backend.is_loaded
    assert backend.model_id == "test-model"


def test_generate_raises_when_not_loaded():
    backend = LocalModelBackend()
    import pytest
    with pytest.raises(RuntimeError, match="No local model loaded"):
        backend.generate_text_only("hello")
    with pytest.raises(RuntimeError, match="No local model loaded"):
        backend.generate_with_image(Image.new("RGB", (10, 10)), "hello")
