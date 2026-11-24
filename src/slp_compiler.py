"""
Generation 6 (Track 1): Straight-Line Program (SLP) Compiler with Algebraic CSE + DCE
and Flat C++17 / Python Code Emitter for Fused Butterfly OptTree.

Records the exact SSA DAG of build_butterfly_opttree_circuit(n), applies
common-subexpression elimination (CSE) and dead-code elimination (DCE),
and emits zero-overhead flat C++17 or Python straight-line programs.
"""

from __future__ import annotations

from src.general_n_butterfly_opttree import build_butterfly_opttree_circuit


class _SLPRecordWire:
    __slots__ = ("id", "depth", "rec")

    def __init__(self, wid: int, depth: int, rec: SLPRecorder) -> None:
        self.id = wid
        self.depth = depth
        self.rec = rec

    def __add__(self, other: _SLPRecordWire) -> _SLPRecordWire:
        a, b = (self.id, other.id) if self.id <= other.id else (other.id, self.id)
        return self.rec.emit("ADD", a, b, max(self.depth, other.depth) + 1)

    def __sub__(self, other: _SLPRecordWire) -> _SLPRecordWire:
        return self.rec.emit("SUB", self.id, other.id, max(self.depth, other.depth) + 1)

    def __mul__(self, other: _SLPRecordWire | int) -> _SLPRecordWire:
        if isinstance(other, _SLPRecordWire):
            a, b = (self.id, other.id) if self.id <= other.id else (other.id, self.id)
            return self.rec.emit("MUL", a, b, max(self.depth, other.depth) + 1)
        return self.rec.emit("SMUL", self.id, int(other), self.depth + 1)

    def __rmul__(self, other: int) -> _SLPRecordWire:
        return self.rec.emit("SMUL", self.id, int(other), self.depth + 1)


class SLPRecorder:
    """Records an arithmetic DAG with optional CSE and DCE."""

    def __init__(self, n: int, enable_cse: bool = True) -> None:
        self.n = n
        self.enable_cse = enable_cse
        self.ops: list[tuple[str, int, int]] = []
        self.table: dict[tuple[str, int, int], int] = {}
        self.depths: dict[int, int] = {i: 0 for i in range(n * n)}
        self.next_id = n * n

    def emit(self, op: str, a: int, b: int, depth: int) -> _SLPRecordWire:
        key = (op, a, b)
        if self.enable_cse and key in self.table:
            wid = self.table[key]
            return _SLPRecordWire(wid, self.depths[wid], self)
        wid = self.next_id
        self.next_id += 1
        self.ops.append((op, a, b))
        self.table[key] = wid
        self.depths[wid] = depth
        return _SLPRecordWire(wid, depth, self)

    def finalize(self, out_id: int) -> dict:
        """Apply dead-code elimination and re-index live SSA wires densely."""
        n2 = self.n * self.n
        used = {out_id}
        for idx in range(len(self.ops) - 1, -1, -1):
            wid = n2 + idx
            if wid in used:
                op, a, b = self.ops[idx]
                used.add(a)
                if op != "SMUL":
                    used.add(b)

        remap = {i: i for i in range(n2)}
        live_ops: list[tuple[str, int, int]] = []
        depths = {i: 0 for i in range(n2)}
        nxt = n2
        for idx, (op, a, b) in enumerate(self.ops):
            wid = n2 + idx
            if wid not in used:
                continue
            ra = remap[a]
            if op == "SMUL":
                rb = b
                d = depths[ra] + 1
            else:
                rb = remap[b]
                d = max(depths[ra], depths[rb]) + 1
            remap[wid] = nxt
            depths[nxt] = d
            live_ops.append((op, ra, rb))
            nxt += 1

        final_out = remap[out_id]
        muls = sum(1 for op, _, _ in live_ops if op == "MUL")
        adds = sum(1 for op, _, _ in live_ops if op in ("ADD", "SUB"))
        smuls = sum(1 for op, _, _ in live_ops if op == "SMUL")
        return {
            "n": self.n,
            "num_inputs": n2,
            "num_wires": nxt,
            "out_id": final_out,
            "ops": live_ops,
            "non_scalar_muls": muls,
            "adds_subs": adds,
            "scalar_muls": smuls,
            "total_gates": len(live_ops),
            "depth": depths[final_out],
        }


def compile_opttree_slp(n: int, p: int = (1 << 61) - 1) -> dict:
    """Compile Gen 5/6 Fused Butterfly OptTree for size n into a CSE+DCE optimized SLP."""
    _, ev = build_butterfly_opttree_circuit(n)
    rec = SLPRecorder(n, enable_cse=True)
    A = [[_SLPRecordWire(i * n + j, 0, rec) for j in range(n)] for i in range(n)]
    out = ev(A, p)
    return rec.finalize(out.id)


def emit_cpp_mod61(slp: dict, fn_name: str | None = None) -> str:
    """
    Emit a flat C++17 function computing Perm_n(A) mod P (P = 2^61 - 1)
    with zero loops, zero branching, and Mersenne-61 fast modular reduction.
    """
    n = slp["n"]
    if fn_name is None:
        fn_name = f"perm{n}_opttree_mod61"
    n2 = slp["num_inputs"]
    nw = slp["num_wires"]
    lines = [
        f"uint64_t {fn_name}(const uint64_t* A) {{",
        f"    static thread_local uint64_t w[{nw}];",
        f"    for (int i = 0; i < {n2}; ++i) w[i] = A[i];",
    ]
    for idx, (op, a, b) in enumerate(slp["ops"]):
        dst = n2 + idx
        if op == "ADD":
            lines.append(f"    w[{dst}] = add61(w[{a}], w[{b}]);")
        elif op == "SUB":
            lines.append(f"    w[{dst}] = sub61(w[{a}], w[{b}]);")
        elif op == "MUL":
            lines.append(f"    w[{dst}] = mul61(w[{a}], w[{b}]);")
        elif op == "SMUL":
            lines.append(f"    w[{dst}] = mul61(w[{a}], {b}ULL);")
    lines.append(f"    return w[{slp['out_id']}];")
    lines.append("}")
    return "\n".join(lines)


def emit_cpp_double(slp: dict, fn_name: str | None = None, p: int = (1 << 61) - 1) -> str:
    """
    Emit a flat C++17 function computing Perm_n(A) over IEEE-754 double
    (converting the single final modular inverse scalar into exact 1/N or -1/N).
    """
    n = slp["n"]
    N = 1 << (n - 1)
    inv_N_mod = pow(N, -1, p)
    neg_inv_N_mod = (-inv_N_mod) % p
    inv_N2_mod = pow(N * N, -1, p)
    neg_inv_N2_mod = (-inv_N2_mod) % p

    if fn_name is None:
        fn_name = f"perm{n}_opttree_f64"
    n2 = slp["num_inputs"]
    nw = slp["num_wires"]
    lines = [
        f"double {fn_name}(const double* A) {{",
        f"    static thread_local double w[{nw}];",
        f"    for (int i = 0; i < {n2}; ++i) w[i] = A[i];",
    ]
    for idx, (op, a, b) in enumerate(slp["ops"]):
        dst = n2 + idx
        if op == "ADD":
            lines.append(f"    w[{dst}] = w[{a}] + w[{b}];")
        elif op == "SUB":
            lines.append(f"    w[{dst}] = w[{a}] - w[{b}];")
        elif op == "MUL":
            lines.append(f"    w[{dst}] = w[{a}] * w[{b}];")
        elif op == "SMUL":
            if b == inv_N_mod:
                scale = f"(1.0 / {N}.0)"
            elif b == neg_inv_N_mod:
                scale = f"(-1.0 / {N}.0)"
            elif b == inv_N2_mod:
                scale = f"(1.0 / {N * N}.0)"
            elif b == neg_inv_N2_mod:
                scale = f"(-1.0 / {N * N}.0)"
            else:
                raise ValueError(f"Unexpected scalar constant {b}")
            lines.append(f"    w[{dst}] = w[{a}] * {scale};")
    lines.append(f"    return w[{slp['out_id']}];")
    lines.append("}")
    return "\n".join(lines)
