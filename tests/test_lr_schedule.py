import numpy

from tokenizer_leakage.src.model import create_model
from tokenizer_leakage.src.utils import get_lr_scheduler
import torch

def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    model = create_model({
    "hidden_size": 232,
    "intermediate_size": 560,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "vocab_size":  4000,
    "context_length": 256 ,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_iters, cosine_cycle_iters, max_learning_rate, min_learning_rate)
    actual_lrs = []

    for _ in range(25):
        actual_lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))
