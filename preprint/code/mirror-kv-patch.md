# Патч mirror-KV для gemma-4 в llama.cpp (TS, 2 карты)

**Блокер без патча:** `-sm tensor` на gemma-4 детерминированно крашится в upstream (C-17): у глобальных слоёв `n_head_kv = 1 < n_devices`, а per-row qk-norm требует полный вектор — head-split KV бьёт в assert `handle_per_row` (ggml-backend-meta.cpp).

**Патч:** ~20 строк в `src/llama-model.cpp::get_tensor_config_impl` (вставка до «standard attention»). Слои gemma-4 с `n_head_kv(il) < n_devices` получают attn_k/attn_v-веса и KV-буферы **MIRRORED** вместо axis-0 split. Тогда Kcur/Vcur — MIRRORED: qk-norm/rms(V)/RoPE проходят без assert; Q/O/FFN остаются head-split с теми же 2 allreduce/слой (96/token).

Сборка: TDM-GCC 10.3 + Vulkan SDK, commit d327bf1f → `runner-patched/`. Результат — E3 v3 / C-21.

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

**Последствие для байт-модели:** глобальный KV не шардится → формула mirror-TS: `байты/карта = W/2 + c_swa/2 + S·16,384` (f16) — наклон роста удваивается против идеального head-split (∂/∂S = 16,384 B/pos вместо 8,192). Идентичность `mirror(S) == ideal(2S)` зафиксирована тестом.
