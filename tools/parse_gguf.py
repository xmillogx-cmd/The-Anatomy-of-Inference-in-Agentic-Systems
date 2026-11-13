"""Parse GGUF metadata of local models -> architecture (ground truth from disk).

CPU-only, stdlib. Reads only the KV section of metadata in the GGUF header (weights untouched).
Both formats supported:
  v1/v2: [u32 key_len][key utf8][u8 value_type][payload]
         types 0..11 per the classic specification.
  v3    : [u32 key_len][u32 res0][key utf8][u32 value_type][payload];
          strings (type 8) = [u32 len][u32 res0][bytes];
          scalars: t4/t5 = int LE (u32 slot), t6 = f32, t7 = u8/bool —
          payload EXACTLY 1 byte (not a u32 slot! this is where the desync was);
          type 9 = vector: [u32 subtag][u32 count][u32 res0][count x elem];
          subtag — same type codes as scalars: {5: u32 (4B), 6: f32 (4B),
          7: u8 (1B)}; subtag 8 = array of utf8 strings ([u32 len][u32 res0][bytes] per elem).

The v3 layout was verified by hex dump and full sequential parsing of
gemma-4-12B-it-Q4_K_M.gguf (2026-10-22): all 45 KV entries decode with exact
length match. Type table: t4 int x21, t5 int x1, t6 f32 x6,
t7 u8 x2 (bool), t8 utf8 x8, t9 vector x7; vector subtags: 5 (u32),
6 (f32), 7 (u8), 8 (array of utf8 strings).

Purpose: architecture parameters of GGUF fixtures for inference_math.py.
CLI: python parse_gguf.py [filter_substring] [--all | --tensors]
"""
import os
import pathlib
import struct
import sys


def _gguf_roots():
    raw = os.environ.get("AOI_GGUF_DIRS", "")
    # strip: cmd `set VAR=a;b && ...` appends a trailing space to the value, and the path with it fails is_dir()
    roots = [pathlib.Path(p.strip()) for p in raw.split(";") if p.strip()]
    live = [r for r in roots if r.is_dir()]
    if not live:
        sys.exit(
            "Set AOI_GGUF_DIRS to ';'-separated directories containing *.gguf. "
            "Got: %r" % (raw,)
        )
    return live


ROOTS = _gguf_roots()

# --- v1/v2 (classic specification) ---
V2_SCALAR_FMT = {1: "<B", 2: "<b", 3: "<B", 4: "<h", 5: "<H",
                 6: "<i", 7: "<I", 8: "<f", 9: "<q", 10: "<Q", 11: "<d"}

# --- v3 (empirically verified on local files) ---
V3_INT4 = {4, 5}                    # int LE, u32 slot (payload exactly 4 bytes)
V3_F32 = 6                          # f32 (payload 4 bytes)
V3_U8 = 7                           # u8/bool — payload EXACTLY 1 byte!
V3_UTF8 = 8
V3_VECTOR = 9
V3_VEC_UTF8 = 8                     # vector subtag: array of utf8 strings
V3_VEC_FMT = {5: "<I", 6: "<f", 7: "<B"}   # subtag -> struct format of the element


def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _pair_v1(f):
    klen = _u32(f)
    key = f.read(klen).decode("utf-8")
    vtype = struct.unpack("<B", f.read(1))[0]
    if vtype == 0:
        vlen = _u32(f)
        return key, f.read(vlen).decode("utf-8")
    fmt = V2_SCALAR_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"v1/v2: unknown value_type {vtype} (key {key})")
    size = struct.calcsize(fmt)
    return key, struct.unpack(fmt, f.read(size))[0]


def _pair_v3(f):
    klen = _u32(f)
    f.seek(4, 1)                      # u32 res0 before the key (always observed as 0)
    key = f.read(klen).decode("utf-8")
    vtype = _u32(f)
    if vtype == V3_UTF8:
        vlen = _u32(f)
        f.seek(4, 1)                  # u32 res0 after len (always observed as 0)
        return key, f.read(vlen).decode("utf-8")
    if vtype in V3_INT4:                     # int LE (u32 slot)
        return key, struct.unpack("<I", f.read(4))[0]
    if vtype == V3_F32:                       # f32
        return key, struct.unpack("<f", f.read(4))[0]
    if vtype == V3_U8:                        # u8/bool — payload exactly 1 byte
        return key, f.read(1)[0]
    if vtype == V3_VECTOR:
        subtag = _u32(f)
        count = _u32(f)
        f.seek(4, 1)                  # u32 res0 (always observed as 0)
        if subtag == V3_VEC_UTF8:     # array of utf8 strings
            vals = []
            for _ in range(count):
                ln = _u32(f)
                f.seek(4, 1)          # u32 res0 (always observed as 0)
                vals.append(f.read(ln).decode("utf-8"))
            return key, vals
        fmt = V3_VEC_FMT.get(subtag)
        if fmt is None:
            raise ValueError(f"v3: unknown vector subtag {subtag} (key {key})")
        char = fmt[1]                 # 'I' | 'f' | 'B'
        esize = 4 if char in "If" else 1
        raw = f.read(count * esize)
        return key, list(struct.unpack(f"<{count}{char}", raw))
    raise ValueError(f"v3: unknown value_type {vtype} (key {key})")


def parse_metadata(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path.name}: not GGUF")
        ver, nt, nk = struct.unpack("<IQQ", f.read(20))
        reader = _pair_v3 if ver >= 3 else _pair_v1
        out = {}
        for _ in range(nk):
            k, v = reader(f)
            out[k] = v
    return ver, nt, out


GGML_TYPE = {0: "f32", 1: "f16", 2: "q8_0", 3: "q4_0", 7: "q2_K", 8: "q3_K_S",
             9: "q3_K_M", 10: "q3_K_L", 11: "q4_K_S", 12: "q4_K_M",
             13: "q5_K_S", 14: "q5_K_M", 15: "q6_K"}


def list_tensors(path):
    """Tensor names, shapes and types (ground truth for Arch assumptions).

    v3 tensor header (verified by hex dump on 2026-10-22):
    [u32 name_len][u32 res0][name utf8][u32 n_dims][n_dims x u64 rows]
    [u8 ggml_type][pad 3B][u64 offset] — v1/v2 without res0/pad.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path.name}: not GGUF")
        ver, nt, nk = struct.unpack("<IQQ", f.read(20))
        reader = _pair_v3 if ver >= 3 else _pair_v1
        for _ in range(nk):
            reader(f)
        out = []
        for _ in range(nt):
            nlen = _u32(f)
            if ver >= 3:
                f.seek(4, 1)                  # u32 res0 (same writer pattern as KV pairs)
            name = f.read(nlen).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack(f"<{ndim}Q", f.read(8 * ndim))) if ndim else ()
            ttype = f.read(1)[0]
            if ver >= 3:
                f.seek(11, 1)                 # pad(3B) + u64 offset (not needed)
            else:
                f.seek(8, 1)                  # v1/v2 classic: straight to u64 offset
            out.append((name, shape, GGML_TYPE.get(ttype, f"t{ttype}")))
    return out


def fmt_val(v):
    if isinstance(v, list):
        head = ", ".join(str(x) for x in v[:8])
        more = " ..." if len(v) > 8 else ""
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
        if numeric:
            s = f"vector[{len(v)}] min={min(v)} max={max(v)} sum={sum(v)}: {head}{more}"
        else:
            s = f"vector[{len(v)}]: {head}{more}"
    elif isinstance(v, float):
        s = repr(round(v, 6))
    else:
        s = str(v)
    return s


def main(args):
    limit = [a for a in args if not a.startswith("--")] or None
    show_all = "--all" in args
    show_tens = "--tensors" in args
    files = []
    for root in ROOTS:
        files += sorted(root.rglob("*.gguf"))
    if limit:
        files = [p for p in files if any(s.lower() in str(p).lower() for s in limit)]
    for p in files:
        try:
            ver, nt, md = parse_metadata(p)
        except Exception as e:
            print(f"{str(p)}  ERROR: {type(e).__name__}: {e}")
            continue
        arch = md.get("general.architecture", "?")
        a = lambda k: md.get(f"{arch}.{k}", None)
        print(f"== {p.parent.name}/{p.name}  (GGUF v{ver}, tensors {nt})")
        print(f"   name={md.get('general.name', '?')!r}  arch={arch}")
        if show_tens:
            for tname, shape, ttype in list_tensors(p):
                dims = "x".join(str(x) for x in shape) or "-"
                print(f"   {tname}  [{dims}]  {ttype}")
        elif show_all:
            for k, v in md.items():
                print(f"   {k}: {fmt_val(v)}")
        else:
            for k in ("block_count", "embedding_length", "context_length",
                      "attention.head_count", "attention.head_count_kv",
                      "attention.head_size", "feed_forward_length"):
                v = a(k)
                if v is not None:
                    print(f"   {k}: {fmt_val(v)}")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else [])
