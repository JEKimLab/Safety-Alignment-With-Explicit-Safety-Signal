# Safety Alignment Can Be Not Superficial With Explicit Safety Signals [ICML 2025]

This is the official implementation for the paper "Safety Alignment Can Be Not Superficial With Explicit Safety Signals", which has been accepted by ICML 2025.

<!-- ⚠️ Due to time constraints during my summer internship, the code is being organized and uploaded gradually, and will be fully available before the conference. -->

## Dependencies Requirement

Please follow instructions in [REARME_base.md](https://github.com/JEKimLab/Safety-Alignment-With-Explicit-Safety-Signal/blob/main/README_base.md) to install dependencies. Please note this repo was build on the llama_recipes, which has change its name to [llama-cookbook](https://github.com/meta-llama/llama-cookbook)

## Update News

- 2025-07-12 Update Readme
- 2025-07-09 Add data and scripts
- 2025-05-07 Initiate the repo

## Prepare Instructions

> Transformers commit id: f51ac9e059a78049362803c1d606a2c6a8160ee4

1. Please replace files in /src/llama_recipes/models/symlink/* with Transformers' files (Uzip /src/llama_recipes/models/symlink/replace.zip first).

- utils.py -> src/transfromers/generation/utils.py
- modeling_llama.py -> src/transformers/models/llama/modeling_llama.py
- modeling_mistral.py -> src/transformers/models/mistral/modeling_mistral.py

2. Please add the following method to Transformers's file `src/transfromers/cache_utils.py`

    ```
    class DynamicCache(Cache):
        ...
        ...
        ...
        def delete_from_index(self, token_index: int):
            """
            Deletes the cache from the given token index. This is used in `generate` to remove the cache after the
            generation of a specific token.
            """
            for layer_idx in range(len(self)):
                if self.key_cache[layer_idx] != []:
                    self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :token_index, :]
                if self.value_cache[layer_idx] != []:
                    self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :token_index, :]
            self._seen_tokens = token_index
        ...
        ...
        ...
    ```

3. comment the following code in `src/transfromers/modeling_utils.py`

    ```
    if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
        old_lm_head = self.get_output_embeddings()
        if isinstance(old_lm_head, torch.nn.Embedding):
            new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
        else:
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
        if hasattr(old_lm_head, "_hf_hook"):
            hook = old_lm_head._hf_hook
            add_hook_to_module(new_lm_head, hook)
        old_lm_head_requires_grad = old_lm_head.weight.requires_grad
        new_lm_head.requires_grad_(old_lm_head_requires_grad)
        self.set_output_embeddings(new_lm_head)
    ```

4. Please check the file "src/llama_recipes/datasets/lima_dataset/lima_dataset.py" for data details.


## Training Instruction

```
bash scripts/train/run.sh

# Please note that HF_model_path_or_name should point to a valid model augmented with a CLS head.
bash scripts/convert/run.sh 
```

## Safety Evaluation Instruction

```
# Please check the scripts folder
# Example

bash scripts/eval/generation/adv_prefill.sh
bash scripts/eval/evaluation/zeval_adv_prefill.sh
# Please change the log file path to your own path.
bash scripts/eval/reevaluation/adv_prefill.sh
```

## Utility Evaluation Instruction

> Please use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for utility evaluation, commit id `3823cfec41c016378acbcc8616dd1ac92c15edd4`

> For perplexicy and mt-bench score, please check the scripts folder

## Citation

```
@article{li2024superficial,
  title={Superficial safety alignment hypothesis},
  author={Li, Jianwei and Kim, Jung-Eun},
  journal={arXiv preprint arXiv:2410.10862},
  year={2024}
}

@article{li2025safety,
  title={Safety alignment can be not superficial with explicit safety signals},
  author={Li, Jianwei and Kim, Jung-Eng},
  journal={arXiv preprint arXiv:2505.17072},
  year={2025}
}
```

