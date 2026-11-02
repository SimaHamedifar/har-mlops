import numpy as np
import pandas as pd
import pytest
import torch

from activity_ml.preprocessing import (
    APPLIANCE_VALUES,
    LOCATION_VALUES,
    DOOR_SENSOR_VALUES,
    SENSOR_STATUS_VALUES,
    encode_features,
    LabelEncoder,
    build_windows,
    HARWindowDataset,
)

# Fixtures
@pytest.fixture
def sample_dataframe():
    """ Create synthetic sensor events. """

    return pd.DataFrame({
        "Hour_X": [0.0, 0.5, -0.5],
        "Hour_Y": [1.0, 0.5, -0.5],
        "time_delta": [0.0, 5.0, 10.0],
        "appliance": ["unknown"] * 3,
        "location": ["unknown"] * 3,
        "door_sensor": ["unknown"] * 3,
        "sensor_status": ["unknown"] * 3,
    })


@pytest.fixture
def sample_sequences():
    """
    Create 12 events belonging to two houses.
    House 01: 6 events
    House 02: 6 events
    """
    features = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
    labels = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    house_ids = np.array(["01"] * 6 + ["02"] * 6)
    return features, labels, house_ids


# Test 1: Feature encoding shape
def test_encode_features_shape(sample_dataframe):
    features = encode_features(sample_dataframe)
    expected_features = (
        3
        + len(APPLIANCE_VALUES)
        + len(LOCATION_VALUES)
        + len(DOOR_SENSOR_VALUES)
        + len(SENSOR_STATUS_VALUES)
    )
    assert features.shape == (3, expected_features)

# Test 2: Feature encoding data type
def test_encode_features_dtype(sample_dataframe):
    features = encode_features(sample_dataframe)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()

# Test 3: One-hot encoding
def test_one_hot_encoding(sample_dataframe):
    features = encode_features(sample_dataframe)
    appliance_start = 3
    appliance_end = appliance_start + len(APPLIANCE_VALUES)
    appliance_features = features[
        :,
        appliance_start:appliance_end
    ]
    assert np.all(appliance_features.sum(axis=1) == 1)
    unknown_index = APPLIANCE_VALUES.index("unknown")
    assert np.all(appliance_features[:, unknown_index] == 1)

# Test 4: Label encoding
def test_label_encoder():
    labels = [
        "Cooking",
        "Sleeping",
        "Cooking",
        "Eating",
    ]
    encoder = LabelEncoder.fit(labels)
    encoded = encoder.transform(labels)
    assert encoder.num_classes == 3
    assert encoded.dtype == np.int64
    assert encoded[0] == encoded[2]
    assert len(np.unique(encoded)) == 3


# Test 5: Unknown activity labels
def test_unknown_activity_label():
    encoder = LabelEncoder.fit([
        "Cooking",
        "Sleeping"
    ])
    with pytest.raises(ValueError):
        encoder.transform(["Working"])


# Test 6: Sliding window count
def test_sliding_window_count(sample_sequences):
    features, labels, house_ids = sample_sequences
    starts, lengths = build_windows(
        features,
        labels,
        house_ids,
        window_size=3,
        stride=1
    )
    # Each house contains 6 events.
    # Number of windows per house = 6 - 3 + 1 = 4.
    # Two houses -> 8 windows.
    assert len(starts) == 8
    assert np.all(lengths == 3)

# Test 7: Sliding window stride
def test_sliding_window_stride(sample_sequences):
    features, labels, house_ids = sample_sequences
    starts, lengths = build_windows(
        features,
        labels,
        house_ids,
        window_size=3,
        stride=2
    )
    # House 01: starts at 0 and 2.
    # House 02: starts at 6 and 8.
    np.testing.assert_array_equal(
        starts,
        np.array([0, 2, 6, 8])
    )


# Test 8: Household boundaries
def test_windows_do_not_cross_houses(sample_sequences):
    features, labels, house_ids = sample_sequences
    window_size = 3
    starts, _ = build_windows(
        features,
        labels,
        house_ids,
        window_size=window_size
    )
    for start in starts:
        window_houses = house_ids[
            start:start + window_size
        ]
        assert len(set(window_houses)) == 1

# Test 9: Invalid window size
def test_invalid_window_size(sample_sequences):
    features, labels, house_ids = sample_sequences
    with pytest.raises(ValueError):
        build_windows(
            features,
            labels,
            house_ids,
            window_size=0
        )
# Test 10: Inconsistent array lengths
def test_inconsistent_lengths(sample_sequences):
    features, labels, house_ids = sample_sequences
    with pytest.raises(ValueError):
        build_windows(
            features,
            labels[:-1],
            house_ids,
            window_size=3
        )


# Test 11: HARWindowDataset
def test_har_window_dataset(sample_sequences):
    features, labels, house_ids = sample_sequences
    window_size = 3
    starts, _ = build_windows(
        features,
        labels,
        house_ids,
        window_size=window_size
    )
    dataset = HARWindowDataset(
        features,
        labels,
        starts,
        window_size
    )
    assert len(dataset) == 8
    window, label = dataset[0]
    assert isinstance(window, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert window.shape == (3, 5)
    assert window.dtype == torch.float32
    assert label.dtype == torch.int64


# Test 12: Window label assignment
def test_window_label_assignment(sample_sequences):
    features, labels, house_ids = sample_sequences
    window_size = 3
    starts, _ = build_windows(
        features,
        labels,
        house_ids,
        window_size=window_size
    )
    dataset = HARWindowDataset(
        features,
        labels,
        starts,
        window_size
    )
    for index, start in enumerate(starts):
        _, label = dataset[index]
        expected_label = labels[
            start + window_size - 1
        ]
        assert label.item() == expected_label

# Test 13: Class counts
def test_class_counts(sample_sequences):
    features, labels, house_ids = sample_sequences
    starts, _ = build_windows(
        features,
        labels,
        house_ids,
        window_size=3
    )
    dataset = HARWindowDataset(
        features,
        labels,
        starts,
        window_size=3
    )
    counts = dataset.class_counts(num_classes=2)
    assert counts.sum() == len(dataset)
    assert len(counts) == 2