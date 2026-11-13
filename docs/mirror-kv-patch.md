# Mirror-KV patch for gemma-4 in llama.cpp (TS, 2 GPUs)

**Blocker without the patch:** `-sm tensor` on gemma-4 deterministically crashes upstream (C-17): global layers have `n_head_kv = 1 < n_devices`, and per-row qk-norm requires a full vector — head-split KV trips the assert in `handle_per_row` (ggml-backend-meta.cpp).

**Patch:** ~20 lines in `src/llama-model.cpp::get_tensor_config_impl` (inserted before "standard attention"). gemma-4 layers with `n_head_kv(il) < n_devices` get their attn_k/attn_v weights and KV buffers set to **MIRRORED** instead of axis-0 split. Then Kcur/Vcur are MIRRORED: qk-norm/rms(V)/RoPE pass without the assert; Q/O/FFN stay head-split with the same 2 allreduce per layer (96/token).

Build: TDM-GCC 10.3 + Vulkan SDK, commit d327bf1f → `runner-patched/`. Result — E3 v3 / C-21.

```cpp
// gemma-4 global layers have a single kv-head (< n_devices): per-row qk-norm/rms on head-split K/V would
//     trip the meta-split assert, so mirror the k-projection weights and KV cache for such layers
if (ud->model->arch == LLM_ARCH_GEMMA4 &&
        (tensor_name.find(".attn_k.weight") != std::string::npos || tensor_name.find(".attn_v.weight") != std::string::npos ||
            std::regex_match(tensor_name, pattern_kv_cache))) {
    uint32_t il = 0;
    if (tensor_name.substr(0, 4) == "blk.") {
        const size_t length_prefix = tensor_name.find('.', 4);
        GGML_ASSERT(length_prefix != std::string::npos);
        il = std::stoull(tensor_name.substr(4, length_prefix));
    } else if (tensor_name.substr(0, 6) == "cache_") {
        const size_t layer_index_start = tensor_name.find("_l", 6);
        GGML_ASSERT(layer_index_start != std::string::npos);
        il = std::stoull(tensor_name.substr(layer_index_start + 2));
    } else {
        GGML_ABORT("fatal error");
    }
    if (hparams.n_head_kv(il) < ud->n_devices) {
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }
}
```

**Consequence for the byte model:** global KV is not sharded → mirror-TS formula: `bytes/GPU = W/2 + c_swa/2 + S·16,384` (f16) — the growth slope doubles versus ideal head-split (∂/∂S = 16,384 B/pos instead of 8,192). The identity `mirror(S) == ideal(2S)` is pinned by a test.
