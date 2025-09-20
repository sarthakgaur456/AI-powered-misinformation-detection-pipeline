---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:1349
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Photos show Keanu Reeves hawking T-shirts with political messages.', 'These images were altered. The original photo shows actor Keanu Reeves in November 2017, and he was not holding a shirt. \u200b'],
    ['Video shows President Donald Trump admitting he became president because of a “rigged” 2024 election.', 'Social media posts shared a short clip of a longer video in which President Donald Trump said he will be president during the 2026 World Cup because the election was "rigged." | But Trump was referring to the 2020 election, which he lost to former President Joe Biden, and which he has falsely and relentlessly claimed was rigged. | The U.S., Canada and Mexico won a 2018 bid to co-host the 2026 World Cup. Trump was referring to the fact that had he won in 2020, he wouldn’t have been president during the 2026 event.'],
    ['“ABC issued an official statement confirming that Joy Behar and Whoopi Goldberg’s contracts will not be renewed.”', 'This claim originated on a self-described satire site. | An ABC spokesperson has said it isn’t true.'],
    ['President Donald Trump’s birthright citizenship executive order would affect former Vice President Kamala Harris’ citizenship status.', "President Donald Trump’s executive order restricting birthright citizenship does not apply to former Vice President Kamala Harris. | The executive order, if it survives legal challenges and is executed, is not retroactive and will take effect starting Feb. 19, 2025. | Kamala Harris was born Oct. 20, 1964. If both Harris’ parents were still on temporary visas and Trump’s order came six decade ago, perhaps Harris' citizenship would be in question."],
    ['In negotiating releases of Americans held abroad, “(I) gave the opposing country NOTHING — and never any cash.”', 'Former President Donald Trump’s administration released prisoners held by the U.S. government in exchange for freeing Americans held abroad. | In 2017, the Trump administration agreed to pay $2 million to North Korea for the release of a U.S. citizen detained there, but a White House official said the money was never paid.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Photos show Keanu Reeves hawking T-shirts with political messages.',
    [
        'These images were altered. The original photo shows actor Keanu Reeves in November 2017, and he was not holding a shirt. \u200b',
        'Social media posts shared a short clip of a longer video in which President Donald Trump said he will be president during the 2026 World Cup because the election was "rigged." | But Trump was referring to the 2020 election, which he lost to former President Joe Biden, and which he has falsely and relentlessly claimed was rigged. | The U.S., Canada and Mexico won a 2018 bid to co-host the 2026 World Cup. Trump was referring to the fact that had he won in 2020, he wouldn’t have been president during the 2026 event.',
        'This claim originated on a self-described satire site. | An ABC spokesperson has said it isn’t true.',
        "President Donald Trump’s executive order restricting birthright citizenship does not apply to former Vice President Kamala Harris. | The executive order, if it survives legal challenges and is executed, is not retroactive and will take effect starting Feb. 19, 2025. | Kamala Harris was born Oct. 20, 1964. If both Harris’ parents were still on temporary visas and Trump’s order came six decade ago, perhaps Harris' citizenship would be in question.",
        'Former President Donald Trump’s administration released prisoners held by the U.S. government in exchange for freeing Americans held abroad. | In 2017, the Trump administration agreed to pay $2 million to North Korea for the release of a U.S. citizen detained there, but a White House official said the money was never paid.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,349 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                       | label                                                           |
  |:--------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                           | float                                                           |
  | details | <ul><li>min: 17 characters</li><li>mean: 98.87 characters</li><li>max: 284 characters</li></ul> | <ul><li>min: 15 characters</li><li>mean: 320.59 characters</li><li>max: 837 characters</li></ul> | <ul><li>min: -1.0</li><li>mean: 0.02</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Photos show Keanu Reeves hawking T-shirts with political messages.</code>                                                | <code>These images were altered. The original photo shows actor Keanu Reeves in November 2017, and he was not holding a shirt. ​</code>                                                                                                                                                                                                                                                                                                                                                                                                               | <code>0.0</code> |
  | <code>Video shows President Donald Trump admitting he became president because of a “rigged” 2024 election.</code>             | <code>Social media posts shared a short clip of a longer video in which President Donald Trump said he will be president during the 2026 World Cup because the election was "rigged." \| But Trump was referring to the 2020 election, which he lost to former President Joe Biden, and which he has falsely and relentlessly claimed was rigged. \| The U.S., Canada and Mexico won a 2018 bid to co-host the 2026 World Cup. Trump was referring to the fact that had he won in 2020, he wouldn’t have been president during the 2026 event.</code> | <code>0.0</code> |
  | <code>“ABC issued an official statement confirming that Joy Behar and Whoopi Goldberg’s contracts will not be renewed.”</code> | <code>This claim originated on a self-described satire site. \| An ABC spokesperson has said it isn’t true.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.5
- Sentence Transformers: 5.1.0
- Transformers: 4.56.1
- PyTorch: 2.8.0+cu129
- Accelerate: 1.10.1
- Datasets: 4.1.1
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->