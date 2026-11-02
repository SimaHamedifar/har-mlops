import pytest
import torch
import torch.nn.functional as F

from activity_ml.model import BiLSTMSelfAttentionClassifier


# Test configuration
BATCH_SIZE = 4
WINDOW_SIZE = 10
INPUT_SIZE = 32
NUM_CLASSES = 22


# Pytest fixture: Create a model
@pytest.fixture
def model():

    model = BiLSTMSelfAttentionClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        num_attention_heads=4,
    )

    return model


# Test 1: Output shape
def test_model_output_shape(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    output = model(x)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)


# Test 2: Forward pass produces valid outputs
def test_forward_pass(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    output = model(x)
    assert torch.isfinite(output).all()

# Test 3: Prediction probabilities
def test_prediction_probabilities(model):
    model.eval()
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (BATCH_SIZE, NUM_CLASSES)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(BATCH_SIZE),
        atol=1e-6,
        rtol=1e-6,
    )

# Test 4: Backpropagation
def test_backpropagation(model):
    model.train()
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    labels = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(BATCH_SIZE,),
    )
    output = model(x)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


# Test 5: Invalid input dimensions
def test_invalid_input_shape(model):
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    with pytest.raises(ValueError):
        model(x)

# Test 6: Incorrect number of features
def test_incorrect_feature_size(model):
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, 20)
    with pytest.raises(ValueError):
        model(x)

# Test 7: Deterministic inference
def test_deterministic_inference(model):
    model.eval()
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    torch.testing.assert_close(output1, output2)


# Test 8: Attention weights
def test_attention_weights(model):
    model.eval()
    x = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE)
    with torch.no_grad():
        logits, attention_weights = model(x, return_attention=True)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert attention_weights.shape == (BATCH_SIZE, WINDOW_SIZE, WINDOW_SIZE)
    torch.testing.assert_close(
        attention_weights.sum(dim=-1),
        torch.ones(BATCH_SIZE, WINDOW_SIZE),
        atol=1e-5,
        rtol=1e-5,
    )

# Test 9: Invalid model configuration
def test_invalid_model_configuration():
    with pytest.raises(ValueError):
        BiLSTMSelfAttentionClassifier(
            input_size=32,
            num_classes=22,
            hidden_size=63,
            bidirectional=True,
            num_attention_heads=4,
        )