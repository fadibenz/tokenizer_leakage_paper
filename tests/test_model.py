from tokenizer_leakage.src.model import create_model
import torch
from transformers import LlamaForCausalLM, LlamaConfig


def test_create_model():

    """
    Tests the create_model function and basic properties of the created Llama model.
    """

    # Test case 1: Valid configuration and model creation
    print("Testing create_model with valid configuration...")

    run_config = {
        "d_model": 64,
        "d_ff": 256,
        "n_layers": 2,
        "num_heads": 8,
        "vocab_size": 1000,
        "context_length": 128
    }
    model = create_model(run_config)

    assert isinstance(model, LlamaForCausalLM), "Model should be an instance of LlamaForCausalLM"
    assert model.config.hidden_size == run_config["d_model"], "Hidden size mismatch"
    assert model.config.intermediate_size == run_config["d_ff"], "Intermediate size mismatch"
    assert model.config.num_hidden_layers == run_config["n_layers"], "Number of layers mismatch"
    assert model.config.num_attention_heads == run_config["num_heads"], "Number of attention heads mismatch"
    assert model.config.vocab_size == run_config["vocab_size"], "Vocab size mismatch"
    assert model.config.max_position_embeddings == run_config["context_length"], "Max position embeddings mismatch"
    assert model.config.rms_norm_eps == 1e-6, "RMS norm epsilon mismatch"

    # Test case 2: Model output shapes and logits
    batch_size = 2
    sequence_length = 50
    batch = torch.randint(0, run_config["vocab_size"], (batch_size, sequence_length))
    output = model(batch)

    assert hasattr(output, "logits"), "Model output should have a 'logits' attribute"
    assert isinstance(output.logits, torch.Tensor), "output.logits should be a torch.Tensor"

    expected_logits_shape = (batch_size, sequence_length, run_config["vocab_size"])
    assert output.logits.shape == expected_logits_shape, \
        f"Logits shape mismatch. Expected {expected_logits_shape}, got {output.logits.shape}"
    print(f"  Logits shape assertion passed. Shape: {output.logits.shape}")

    print("Testing weight initialization...")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {param_count:,}")

    for name, param in model.named_parameters():
        assert not torch.allclose(param, torch.zeros_like(param)), f"Parameter {name} is all zeros"

    print("  Weight initialization tests passed.")


def test_model_initialization():
    """Test that model weights are properly initialized and not pre-trained."""

    run_config = {
        "d_model": 64,
        "d_ff": 256,
        "n_layers": 2,
        "num_heads": 8,
        "vocab_size": 1000,
        "context_length": 128
    }

    model1 = create_model(run_config)
    model2 = create_model(run_config)

    # Parameters that should be deterministically initialized (not random)
    deterministic_params = {
        'layernorm.weight',
        'layernorm.bias',
        'norm.weight',
        'norm.bias'
    }

    # Test models have different random initializations (except for deterministic ones)
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter names should match: {name1} vs {name2}"

        if any(det_param in name1 for det_param in deterministic_params):
            assert torch.allclose(param1, param2), f"Deterministic parameter {name1} should be identical across models"
        else:
            assert not torch.allclose(param1, param2), f"Random parameter {name1} should differ between models"
