import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from activity_ml.inference import HARPredictor, load_feature_window_csv, predict_activity
from activity_ml.model import BiLSTMSelfAttentionClassifier


# --- Tests for load_feature_window_csv ---

def test_load_feature_window_csv_valid():
    csv_data = "feature1,feature2,feature3\n1.0,2.0,3.0\n4.0,5.0,6.0\n"
    file_like = io.StringIO(csv_data)
    feature_names = ["feature1", "feature2"]

    # Normally load_feature_window_csv takes a file path, but pandas.read_csv accepts file-like objects too.
    result = load_feature_window_csv(file_like, feature_names)

    expected = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

def test_load_feature_window_csv_missing_features():
    csv_data = "feature1,feature2\n1.0,2.0\n"
    file_like = io.StringIO(csv_data)
    feature_names = ["feature1", "feature3"]

    with pytest.raises(ValueError, match="window CSV is missing feature columns: \\['feature3'\\]"):
        load_feature_window_csv(file_like, feature_names)

def test_load_feature_window_csv_non_finite():
    # Use pandas string representation for inf/nan
    csv_data = "feature1,feature2\n1.0,inf\nNaN,2.0\n"
    file_like = io.StringIO(csv_data)
    feature_names = ["feature1", "feature2"]

    with pytest.raises(ValueError, match="window CSV feature values must be numeric and finite"):
        load_feature_window_csv(file_like, feature_names)


# --- Tests for HARPredictor ---

@pytest.fixture
def dummy_model():
    model = MagicMock(spec=BiLSTMSelfAttentionClassifier)
    # create a mock that returns something plausible for predict_proba
    def mock_predict_proba(tensor):
        # batch_size 1, num_classes 3
        # tensor is shape [1, window_size, input_size]
        return torch.tensor([[0.1, 0.7, 0.2]])
    model.predict_proba.side_effect = mock_predict_proba

    # Needs to be able to chain .to(device).eval()
    model.to.return_value = model
    model.eval.return_value = model
    return model

@pytest.fixture
def predictor(dummy_model):
    classes = ["walking", "running", "sitting"]
    feature_names = ["accel_x", "accel_y"]
    window_size = 5
    device = torch.device("cpu")
    return HARPredictor(dummy_model, classes, feature_names, window_size, device)

def test_har_predictor_predict_proba_valid(predictor, dummy_model):
    # Valid input: shape (5, 2)
    sequence = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ]

    probs = predictor.predict_proba(sequence)

    # Assert model was called properly
    dummy_model.predict_proba.assert_called_once()
    called_tensor = dummy_model.predict_proba.call_args[0][0]
    assert called_tensor.shape == (1, 5, 2)

    # Assert output
    np.testing.assert_array_almost_equal(probs, np.array([0.1, 0.7, 0.2]))

def test_har_predictor_predict_proba_invalid_shape(predictor):
    # Invalid shape: (4, 2) instead of (5, 2)
    sequence = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ]

    with pytest.raises(ValueError, match="expected feature window shape \\(5, 2\\), got \\(4, 2\\)"):
        predictor.predict_proba(sequence)

def test_har_predictor_predict_proba_non_finite(predictor):
    # Non-finite values
    sequence = [
        [1.0, np.inf],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [np.nan, 10.0]
    ]

    with pytest.raises(ValueError, match="feature window must contain only finite values"):
        predictor.predict_proba(sequence)

def test_har_predictor_predict_activity(predictor):
    sequence = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ]

    result = predictor.predict_activity(sequence)

    expected_probs = {
        "walking": 0.1,
        "running": 0.7,
        "sitting": 0.2
    }

    assert result["predicted_activity"] == "running"
    for k, v in result["probabilities"].items():
        assert np.isclose(v, expected_probs[k], atol=1e-5)

@patch("activity_ml.inference.load_checkpoint")
@patch("activity_ml.inference.BiLSTMSelfAttentionClassifier")
def test_har_predictor_from_checkpoint(mock_model_class, mock_load_checkpoint):
    mock_payload = {
        "model_config": {"input_size": 2, "hidden_size": 10},
        "state_dict": {"layer1.weight": torch.tensor([1.0])},
        "classes": ["walking", "running"],
        "feature_names": ["accel_x", "accel_y"],
        "window_size": 10
    }
    mock_load_checkpoint.return_value = mock_payload

    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance

    path = "fake_checkpoint.pt"
    predictor = HARPredictor.from_checkpoint(path, device="cpu")

    # Assert load_checkpoint called correctly
    mock_load_checkpoint.assert_called_once_with(path, torch.device("cpu"))

    # Assert model created and state_dict loaded
    mock_model_class.assert_called_once_with(**mock_payload["model_config"])
    mock_model_instance.load_state_dict.assert_called_once_with(mock_payload["state_dict"])

    assert predictor.classes == ("walking", "running")
    assert predictor.feature_names == ("accel_x", "accel_y")
    assert predictor.window_size == 10

@patch("activity_ml.inference.HARPredictor.from_checkpoint")
def test_predict_activity_wrapper(mock_from_checkpoint):
    mock_predictor = MagicMock()
    mock_predictor.predict_activity.return_value = {"predicted_activity": "running"}
    mock_from_checkpoint.return_value = mock_predictor

    sequence = [[1.0, 2.0]]
    path = "fake_checkpoint.pt"
    device = "cpu"

    result = predict_activity(sequence, path, device)

    # Assert HARPredictor instantiated and used correctly
    mock_from_checkpoint.assert_called_once_with(path, device)
    mock_predictor.predict_activity.assert_called_once_with(sequence)

    assert result == {"predicted_activity": "running"}
