# Safety-Alignment-With-Explicit-Safety-Signal

⚠️ Due to time constraints during my summer internship, the code is being organized and uploaded gradually, and will be fully available before the conference.

## Requirement

Please follow instructions in [REARME_base.md](https://github.com/JEKimLab/Safety-Alignment-With-Explicit-Safety-Signal/blob/main/README_base.md) to install dependencies. Please note this repo was build on the llama_recipes, which has change its name to [llama-cookbook](https://github.com/meta-llama/llama-cookbook)

## Update

- 2025-05-07 Intiate the repo

## Instructions

> Transformers commit id: f51ac9e059a78049362803c1d606a2c6a8160ee4

1. Please replace files in /src/llama_recipes/models/symlink/* with Transformers' files.

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

### 