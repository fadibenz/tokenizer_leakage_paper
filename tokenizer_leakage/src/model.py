from transformers import LlamaForCausalLM, LlamaConfig

def create_model(run_config: dict) -> LlamaForCausalLM:
    """Creates a Llama-style model based on a configuration file."""
    if not run_config:
        raise ValueError(f"No config file was provided, you need to specify config file to create model")

    config =  LlamaConfig(
        hidden_size=run_config["d_model"],
        intermediate_size=run_config["d_ff"],
        num_hidden_layers=run_config["n_layers"],
        num_attention_heads=run_config["num_heads"],
        vocab_size=run_config["vocab_size"],
        max_position_embeddings= run_config["context_length"],
        rms_norm_eps= 1e-6
    )

    return LlamaForCausalLM(config)