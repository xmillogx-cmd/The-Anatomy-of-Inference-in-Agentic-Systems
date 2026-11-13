"""Разбор заголовков safetensors локальных моделей -> архитектура (ground truth с диска).

CPU-only, stdlib. Читает только JSON-заголовок каждого shard'а (первые N байт),
веса не трогаем. Вывод компактный: на модель — size/dtypes/число ТЕКСТОВЫХ слоёй
(без vision tower) + shape тензоров layer 0 + embedding/lm_head.

Назначение: эталонные архитектурные параметры для math/inference_math.py и тестов.
"""
import json
import os
import pathlib
import re
import struct

import sys


def _models_root():
    # strip: cmd `set VAR=path && ...` добавляет в конец значения пробел, и путь с ним не проходит is_dir()
    p = os.environ.get("AOI_MODELS_DIR", "").strip()
    if not p or not pathlib.Path(p).is_dir():
        sys.exit(
            "Set AOI_MODELS_DIR to the directory containing model folders "
            "(each with *.safetensors shards). Got: %r" % (p,)
        )
    return pathlib.Path(p)


ROOT = _models_root()

TEXT_LAYER_RES = (
    re.compile(r"^model\.layers\.(\d+)\."),
    re.compile(r"^model\.language_model\.layers\.(\d+)\."),
)
LAYER0_PREFIXES = ("model.layers.0.", "model.language_model.layers.0.")
EMBED_MARKERS = ("embed_tokens", "lm_head", "wte", "embedding")


def text_layer_idx(tensors):
    idx = set()
    for n in tensors:
        for rx in TEXT_LAYER_RES:
            m = rx.match(n)
            if m:
                idx.add(int(m.group(1)))
                break
    return sorted(idx)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def main():
    out = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.safetensors"))
        if not files:
            continue
        tensors = {}
        dtypes = set()
        for fp in files:
            h = read_header(fp)
            tensors.update(h)
            for m in h.values():
                if isinstance(m, dict) and m.get("dtype"):
                    dtypes.add(m["dtype"])

        text_idx = text_layer_idx(tensors)
        l0 = {}
        for pref in LAYER0_PREFIXES:
            for n, meta in tensors.items():
                if isinstance(meta, dict) and n.startswith(pref):
                    l0[n[len(pref) :]] = meta["shape"]
        embeds = {
            n: meta["shape"]
            for n, meta in tensors.items()
            if isinstance(meta, dict)
            and "vision" not in n
            and any(k in n for k in EMBED_MARKERS)
        }
        out[d.name] = {
            "n_shards": len(files),
            "size_mb": round(sum(fp.stat().st_size for fp in files) / 2**20, 1),
            "dtypes": sorted(dtypes),
            "n_text_layers": (max(text_idx) + 1) if text_idx else None,
            "layer0": l0,
            "embeds": embeds,
        }
    print(json.dumps(out, indent=1))


def dump_names(model, limit):
    d = ROOT / model
    tensors = {}
    for fp in sorted(d.glob("*.safetensors")):
        tensors.update(read_header(fp))
    items = sorted(tensors.items())
    print(f"== {model}: {len(items)} тензоров, первые {min(limit, len(items))} ==")
    for name, meta in items[:limit]:
        if isinstance(meta, dict):
            print(f"{name}\t{meta.get('dtype')}\t{meta.get('shape')}")
        else:
            print(f"{name}\t<non-dict>")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "names":
        dump_names(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60)
    else:
        main()
