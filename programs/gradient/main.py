# coding=utf-8
# Copyright 2018 the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from functools import partial

from tensor_interval import *

from torch.profiler import profile, record_function, ProfilerActivity

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainingArguments,
    is_torch_available,
    set_seed,
)


RTOL = 1e-5
ATOL = 1e-5

if is_torch_available():
    from torch import nn

    from transformers import (
        AutoModelForCausalLM,
        Trainer,
    )


class StoreLossCallback(TrainerCallback):
    """
    Simple callback to store the loss.
    """

    def __init__(self):
        self.losses = []
        self.losses_interval = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])
        if "loss_interval" in logs:
            self.losses_interval.append(logs["loss_interval"])


def ForCausalLMLoss(
    logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False
):
    logits.init_interval()
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    if num_items_in_batch is None or disable_num_items_in_batch:
        loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="mean"
        )
    else:
        loss = nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="sum"
        )
        loss = loss / num_items_in_batch
    return loss


def test_gradient_accumulation_loss_alignment_with_model_loss():
    set_seed(42)
    import datasets

    model_name = "nickypro/tinyllama-15M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], max_length=16, padding="max_length", truncation=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_kwargs = {
        "report_to": "none",
        "logging_steps": 1,
        "max_steps": 5,
        "learning_rate": 3e-4,
        "disable_tqdm": True,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            **args_kwargs,
        )
        # train with base loss
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        base_loss_callback = StoreLossCallback()
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[base_loss_callback],
            data_collator=data_collator,
        )
        assert trainer.model_accepts_loss_kwargs
        trainer.train()

        args = TrainingArguments(
            tmp_dir,
            **args_kwargs,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=4,
        )

        # train with gradient accumulation
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        grad_accum_loss_callback = StoreLossCallback()
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[grad_accum_loss_callback],
            data_collator=data_collator,
        )
        assert trainer.model_accepts_loss_kwargs
        trainer.train()

        # train with broken loss
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        broken_loss_callback = StoreLossCallback()
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[broken_loss_callback],
            data_collator=data_collator,
        )
        # disable model_accepts_loss_kwargs so that "num_items_in_batch" is not passed to the model
        trainer.model_accepts_loss_kwargs = False
        trainer.train()

    # Calculate the difference between the base loss and the grad_accum loss
    diff_truth = [
        abs(base - grad)
        for base, grad in zip(
            base_loss_callback.losses, grad_accum_loss_callback.losses
        )
    ]
    diff_broken = [
        abs(base - grad)
        for base, grad in zip(base_loss_callback.losses, broken_loss_callback.losses)
    ]

    # all diff truth should be quite close
    assert max(diff_truth) < 0.01, f"Difference {max(diff_truth)} is not within 0.01"
    # max diff broken should be very off ("very off" is arbitrary, but as long as it's bigger than 0.1, it's fine)
    assert (
        max(diff_broken) > 0.7
    ), f"Difference {max(diff_broken)} is not greater than 0.7"

    # loss_base = sum(base_loss_callback.losses)
    # loss_broken = sum(broken_loss_callback.losses)

    # # mean/sum loss should not vary too much.
    # relative_diff = abs(loss_base - loss_broken) / max(loss_base, loss_broken)
    # assert relative_diff < 0.2, f"Relative difference {relative_diff} is not within 0.2"


def test_gradient_accumulation_loss_alignment_with_loss_func():
    set_seed(42)
    import datasets

    model_name = "roneneldan/TinyStories-33M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], max_length=16, padding="max_length", truncation=True
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    def compute_loss(
        logits, labels, vocab_size, num_items_in_batch, disable_num_items_in_batch=False
    ):
        return ForCausalLMLoss(
            logits["logits"],
            labels,
            vocab_size,
            num_items_in_batch,
            disable_num_items_in_batch,
        )

    loss_fn = partial(
        compute_loss,
        vocab_size=model.config.vocab_size,
        disable_num_items_in_batch=False,
    )

    base_loss_callback = StoreLossCallback()

    args_kwargs = {
        "report_to": "none",
        "logging_steps": 1,
        "max_steps": 5,
        "learning_rate": 3e-4,
        "disable_tqdm": True,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            **args_kwargs,
        )
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[base_loss_callback],
            compute_loss_func=loss_fn,
            data_collator=data_collator,
        )
        trainer.train()

    grad_accum_loss_callback = StoreLossCallback()
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            **args_kwargs,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=4,
        )
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[grad_accum_loss_callback],
            compute_loss_func=loss_fn,
            data_collator=data_collator,
        )
        trainer.train()

        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        broken_loss_callback = StoreLossCallback()
        loss_fn = partial(
            compute_loss,
            vocab_size=model.config.vocab_size,
            disable_num_items_in_batch=True,
        )
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[broken_loss_callback],
            compute_loss_func=loss_fn,
            data_collator=data_collator,
        )
        trainer.train()
        # Calculate the difference between the base loss and the grad_accum loss
        diff_truth = [
            abs(base - grad)
            for base, grad in zip(
                base_loss_callback.losses, grad_accum_loss_callback.losses
            )
        ]
        diff_broken = [
            abs(base - grad)
            for base, grad in zip(
                base_loss_callback.losses, broken_loss_callback.losses
            )
        ]

        # all diff truth should be quite close
        assert (
            max(diff_truth) < 0.01
        ), f"Difference {max(diff_truth)} is not within 0.01"

        # max diff broken should be very off
        assert (
            max(diff_broken) > 3
        ), f"Difference {max(diff_broken)} is not greater than 3"
        draw(
            base_loss_callback.losses,
            base_loss_callback.losses_interval,
            grad_accum_loss_callback.losses,
            broken_loss_callback.losses,
        )


def draw(base, base_interval, grad_accum, broken):
    import matplotlib.pyplot as plt

    # TODO: why /= 2?
    broken = [x / 2 for x in broken]
    plt.plot(base, label="base")
    base_interval_left = [x[0] for x in base_interval]
    base_interval_right = [x[1] for x in base_interval]
    plt.fill_between(
        range(len(base)), base_interval_left, base_interval_right, alpha=0.2
    )
    # plt.plot(grad_accum, label="grad_accum")
    plt.plot(broken, label="broken")
    plt.legend()
    plt.savefig("loss.png")


def correct_gradient_accumulation():
    set_seed(42)
    import datasets

    model_name = "nickypro/tinyllama-15M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:40]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], max_length=16, padding="max_length", truncation=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_kwargs = {
        "report_to": "none",
        "logging_steps": 1,
        "max_steps": 20,
        "learning_rate": 3e-4,
        "disable_tqdm": True,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            tmp_dir,
            **args_kwargs,
        )
        # train with base loss
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        base_loss_callback = StoreLossCallback()
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_dataset,
            callbacks=[base_loss_callback],
            data_collator=data_collator,
        )
        assert trainer.model_accepts_loss_kwargs
        trainer.train()
        print(base_loss_callback.losses)


test_gradient_accumulation_loss_alignment_with_loss_func()
# test_gradient_accumulation_loss_alignment_with_model_loss()


# GPTNeoForCausalLM(
#   (transformer): GPTNeoModel(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(2048, 768)
#     (h): ModuleList(
#       (0-3): 4 x GPTNeoBlock(
#         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): GPTNeoAttention(
#           (attention): GPTNeoSelfAttention(
#             (k_proj): Linear(in_features=768, out_features=768, bias=False)
#             (v_proj): Linear(in_features=768, out_features=768, bias=False)
#             (q_proj): Linear(in_features=768, out_features=768, bias=False)
#             (out_proj): Linear(in_features=768, out_features=768, bias=True)
#           )
#         )
#         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): GPTNeoMLP(
#           (c_fc): Linear(in_features=768, out_features=3072, bias=True)
#           (c_proj): Linear(in_features=3072, out_features=768, bias=True)
#           (act): NewGELUActivation()
#         )
#       )
#     )
#     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)

# CrossEntropyLoss(
#   (weight): None
#   (reduction): Mean()
#   (loss): CrossEntropyLoss()
#   (ignore_index): -100
#   (size_average): None
#   (total_weight): None
#   (reduction): Sum()
# )
