# RUN: %python -m pytest %s -q

# Copyright 2025 IBM Corp.
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

"""Unit tests for the ``_load_examples`` helpers in ``conftest``.

Two related mechanisms that turn ``VARIANTS`` declarations into registry entries:

**Sweep grammar** — ``params`` lists expand into a Cartesian product, one entry
per point, keyed as ``<variant>[k=v, ...]``:

  Plain values::

      "M": [128, 256, 512]   →  entries [M=128], [M=256], [M=512]

  Labelled tuples — for complex types whose repr would be unreadable in a key::

      "LAYOUT": [("stick", [...])]   →  entry [LAYOUT=stick]  (always shown)

  ``_normalise_param_list``, ``_expand_params``, ``_sweep_suffix``.

**Factory protocol** — a variant's ``"factory"`` key carries a ``VariantFactory``
whose hooks produce the fields that depend on the combination, so ``OP`` × ``DTYPE``
is one declaration rather than twelve. ``_apply_factory``, ``VariantFactory``.

A lit test rather than a pytest module: needs no dbo-opt and no device, and a test
that needs nothing should not sit in the suite that serializes the device. Same
direction as #71 for structural checks.
"""

import functools
from dataclasses import dataclass

import pytest

from conftest import (
    VariantFactory,
    _apply_factory,
    _expand_params,
    _normalise_param_list,
    _sweep_suffix,
)


# ---------------------------------------------------------------------------
# _normalise_param_list
#
# Converts a raw params list into a uniform list of (label, value) pairs.
# Plain values get str(value) as their auto-label; labelled tuples are
# validated and passed through unchanged.  Mixed lists are rejected.
# ---------------------------------------------------------------------------

def test_normalise_plain_ints():
    """Plain ints are auto-labelled with str(value)."""
    result = _normalise_param_list("M", [128, 256, 512])
    assert result == [("128", 128), ("256", 256), ("512", 512)]


def test_normalise_plain_str():
    """Plain strings are auto-labelled with the string itself."""
    result = _normalise_param_list("dtype", ["f32", "f16"])
    assert result == [("f32", "f32"), ("f16", "f16")]


def test_normalise_labelled_tuples():
    """Labelled tuples are returned as-is; labels are preserved."""
    values = [("small", 16), ("large", 512)]
    result = _normalise_param_list("N", values)
    assert result == [("small", 16), ("large", 512)]


def test_normalise_single_labelled_tuple():
    """A single labelled tuple is valid — the label triggers always-suffix behaviour."""
    values = [("tile64", 64)]
    result = _normalise_param_list("BLOCK", values)
    assert result == [("tile64", 64)]


def test_normalise_mixed_raises():
    """Mixing plain values and labelled tuples in one list is an error.

    The author must choose one form consistently — mixing makes the label
    contract ambiguous and is almost certainly a typo.
    """
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _normalise_param_list("M", [128, ("big", 512)])


def test_normalise_mixed_raises_includes_param_name():
    """The error message names the offending param so the author can find it."""
    with pytest.raises(ValueError, match=r"params\[.M.\]"):
        _normalise_param_list("M", [("a", 1), 2])


def test_normalise_mixed_raises_includes_kernel_name():
    """The error message also includes the kernel/variant name for context."""
    with pytest.raises(ValueError, match="myfixture"):
        _normalise_param_list("M", [("a", 1), 2], kernel_name="myfixture")


def test_normalise_non_string_label_raises():
    """Labels must be strings — int labels would produce confusing key suffixes."""
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [(42, 128)])


def test_normalise_wrong_tuple_length_raises():
    """Tuples must be exactly 2-element (label, value) — 3-element is rejected."""
    with pytest.raises(ValueError, match="must be .str, value."):
        _normalise_param_list("K", [("a", 1, 2)])


def test_normalise_empty_list():
    """An empty list normalises to an empty list (edge case, handled gracefully)."""
    result = _normalise_param_list("X", [])
    assert result == []


# ---------------------------------------------------------------------------
# _expand_params
#
# Produces the Cartesian product of all param lists, returning each
# combination as a dict mapping param name → (label, value).
# Also returns ``always_suffixed``: the set of param names whose original
# list contained at least one labelled tuple.
# ---------------------------------------------------------------------------

def test_expand_single_plain_value():
    """Single plain value → one combo, param not in always_suffixed.

    This is the existing single-element-list case; behaviour is unchanged.
    """
    combos, always_suffixed = _expand_params({"M": [64]})
    assert len(combos) == 1
    assert combos[0] == {"M": ("64", 64)}
    assert always_suffixed == set()


def test_expand_multi_plain_values():
    """Multiple plain values → one combo per value, none always-suffixed."""
    combos, always_suffixed = _expand_params({"M": [1, 2, 3]})
    assert len(combos) == 3
    assert combos[0] == {"M": ("1", 1)}
    assert combos[2] == {"M": ("3", 3)}
    assert always_suffixed == set()


def test_expand_single_labelled_tuple():
    """Single labelled tuple → one combo, param IS in always_suffixed.

    The label signals intent to always show this param in the key suffix,
    even with only one value — useful for complex types like layout dicts.
    """
    combos, always_suffixed = _expand_params({"BLOCK": [("t64", 64)]})
    assert len(combos) == 1
    assert combos[0] == {"BLOCK": ("t64", 64)}
    assert "BLOCK" in always_suffixed


def test_expand_multi_labelled_tuples():
    """Multiple labelled tuples → one combo per label, param in always_suffixed."""
    combos, always_suffixed = _expand_params({"N": [("a", 1), ("b", 2)]})
    assert len(combos) == 2
    assert combos[0] == {"N": ("a", 1)}
    assert combos[1] == {"N": ("b", 2)}
    assert "N" in always_suffixed


def test_expand_cartesian_product_two_params():
    """Two multi-value params produce a full Cartesian product.

    {"M": [1, 2], "K": [3, 4]} → 4 combos covering all (M, K) pairs.
    """
    combos, always_suffixed = _expand_params({"M": [1, 2], "K": [3, 4]})
    assert len(combos) == 4
    assert always_suffixed == set()
    pairs = {(c["M"][1], c["K"][1]) for c in combos}
    assert pairs == {(1, 3), (1, 4), (2, 3), (2, 4)}


def test_expand_cartesian_product_mixed_label_and_plain():
    """Labelled and plain params can be combined; only the labelled one is always-suffixed."""
    combos, always_suffixed = _expand_params({
        "M": [("small", 16), ("large", 256)],  # labelled
        "K": [32, 64],                          # plain
    })
    assert len(combos) == 4   # 2 M-values × 2 K-values
    assert "M" in always_suffixed
    assert "K" not in always_suffixed


def test_expand_mixed_raises():
    """Mixed labelled/plain list is rejected at expansion time (not silently accepted)."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _expand_params({"M": [1, ("big", 512)]})


def test_expand_empty_params():
    """Empty params dict → one empty combo (itertools.product behaviour)."""
    combos, always_suffixed = _expand_params({})
    assert combos == [{}]
    assert always_suffixed == set()


# ---------------------------------------------------------------------------
# _sweep_suffix
#
# Builds the "[k=v, ...]" suffix appended to registry keys.  A param
# appears in the suffix if it has more than one value OR is always-suffixed
# (was labelled).  Keys are sorted alphabetically for stability.
# ---------------------------------------------------------------------------

def _make_combo(params: dict) -> tuple[dict, dict, set]:
    """Normalise params and return (merged_normalised, first_combo, always_suffixed).

    Helper that wires together _expand_params and _normalise_param_list so
    individual suffix tests don't repeat the setup boilerplate.
    """
    combos, always_suffixed = _expand_params(params)
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}
    return merged, combos[0], always_suffixed


def test_suffix_no_swept_params():
    """Single plain value, not labelled → empty suffix (no brackets in key)."""
    merged, combo, always_suffixed = _make_combo({"M": [64]})
    assert _sweep_suffix(merged, combo, always_suffixed) == ""


def test_suffix_single_plain_multi_value():
    """Multi-value plain param → suffix uses str(value) as the label."""
    merged, combo, always_suffixed = _make_combo({"M": [64, 128]})
    # First combo has M=64; suffix should be "[M=64]"
    suffix = _sweep_suffix(merged, combo, always_suffixed)
    assert suffix == "[M=64]"


def test_suffix_single_labelled_single_value_always_shown():
    """Single labelled tuple → suffix is always present, uses custom label.

    This is the key use case for complex types like layout descriptors:
    the author supplies a short human-readable name for an otherwise
    unreadable repr.
    """
    merged, combo, always_suffixed = _make_combo({"BLOCK": [("tile64", 64)]})
    suffix = _sweep_suffix(merged, combo, always_suffixed)
    assert suffix == "[BLOCK=tile64]"


def test_suffix_labelled_multi_value_uses_custom_labels():
    """Multi-value labelled param → each combo uses its custom label, not repr(value)."""
    params = {"N": [("small", 16), ("large", 256)]}
    combos, always_suffixed = _expand_params(params)
    merged = {"N": _normalise_param_list("N", params["N"])}

    assert _sweep_suffix(merged, combos[0], always_suffixed) == "[N=small]"
    assert _sweep_suffix(merged, combos[1], always_suffixed) == "[N=large]"


def test_suffix_multiple_swept_params_sorted_alphabetically():
    """Multiple swept params appear in the suffix sorted alphabetically.

    Alphabetical order makes key names stable regardless of dict insertion
    order, which can vary between Python versions.
    """
    params = {"Z": [1, 2], "A": [10, 20]}
    combos, always_suffixed = _expand_params(params)
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}

    suffix = _sweep_suffix(merged, combos[0], always_suffixed)
    # "A" must come before "Z" in the suffix
    assert suffix.startswith("[A=")
    assert "Z=" in suffix
    assert suffix.index("A=") < suffix.index("Z=")


def test_suffix_plain_single_value_no_suffix():
    """Multiple plain single-value params → empty suffix (neither swept nor labelled)."""
    params = {"BLOCK": [64], "M": [128]}
    merged, combo, always_suffixed = _make_combo(params)
    assert _sweep_suffix(merged, combo, always_suffixed) == ""


def test_suffix_mixed_labelled_and_plain_multi():
    """Labelled single-value and plain multi-value both appear in the suffix.

    TAG is always-suffixed (labelled); N is multi-value (len > 1).
    Both should be present in every combo's suffix.
    """
    params = {"TAG": [("fast", 1)], "N": [4, 8]}
    combos, always_suffixed = _expand_params(params)
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}

    suffix0 = _sweep_suffix(merged, combos[0], always_suffixed)
    suffix1 = _sweep_suffix(merged, combos[1], always_suffixed)

    assert "TAG=fast" in suffix0 and "N=4" in suffix0
    assert "TAG=fast" in suffix1 and "N=8" in suffix1


# ---------------------------------------------------------------------------
# Integration: simulate the _load_examples key-generation pipeline
#
# These tests verify the end-to-end registry key format without touching
# the filesystem or requiring a real fixture.  They replicate the core
# loop logic from _load_examples using only the public helpers.
# ---------------------------------------------------------------------------

def _simulate_registry_keys(kernel_name: str, params: dict) -> list[str]:
    """Simulate key generation for a single variant in _load_examples.

    Given a variant name and its params dict, returns the list of registry
    keys that _load_examples would emit (one per Cartesian-product point).
    """
    combos, always_suffixed = _expand_params(params, kernel_name=kernel_name)
    merged = {k: _normalise_param_list(k, v) for k, v in params.items()}
    return [kernel_name + _sweep_suffix(merged, combo, always_suffixed)
            for combo in combos]


def test_integration_single_plain_no_suffix():
    """Single-value plain params → one entry with no suffix (backward compat)."""
    keys = _simulate_registry_keys("vector_add", {"N": [64], "BLOCK": [32]})
    assert keys == ["vector_add"]


def test_integration_multi_plain_suffix_with_value():
    """Multi-value plain param → one entry per value, suffix uses the value."""
    keys = _simulate_registry_keys("matmul", {"M": [128, 256]})
    assert keys == ["matmul[M=128]", "matmul[M=256]"]


def test_integration_labelled_single_always_suffix():
    """Single labelled tuple → one entry, suffix present with custom label.

    Typical use: a layout descriptor that is the only configuration but
    needs a readable name in the test ID.
    """
    keys = _simulate_registry_keys("softmax", {"BLOCK": [("tile32", 32)]})
    assert keys == ["softmax[BLOCK=tile32]"]


def test_integration_cartesian_two_labelled():
    """Two labelled params → Cartesian product with both labels in each key.

    Suffix params are sorted alphabetically (K before M here).
    """
    keys = _simulate_registry_keys("kernel", {
        "M": [("s", 16), ("l", 64)],
        "K": [("a", 4), ("b", 8)],
    })
    assert len(keys) == 4
    assert "kernel[K=a, M=s]" in keys
    assert "kernel[K=b, M=l]" in keys


def test_integration_mixed_error_propagates():
    """Mixed labelled/plain list raises before any keys are emitted."""
    with pytest.raises(ValueError, match="mixes labelled tuples"):
        _simulate_registry_keys("bad_kernel", {"X": [1, ("label", 2)]})


# ---------------------------------------------------------------------------
# _apply_factory / VariantFactory
# ---------------------------------------------------------------------------

def _apply(entry: dict, combo_values: dict) -> dict:
    """``_apply_factory`` on *entry* for a combination of plain values.

    Wraps each value back into the ``(label, value)`` pair form ``_expand_params``
    produces, which is what ``_apply_factory`` consumes.
    """
    combo = {k: (str(v), v) for k, v in combo_values.items()}
    _apply_factory(entry, combo, kernel_name="fix::variant")
    return entry


def test_absent_factory_leaves_the_entry_untouched():
    """No ``factory`` key → the entry is passed through byte for byte."""
    entry = {"kernel_fn": object(), "params": {"M": [64]}, "reference": len}
    before = dict(entry)
    _apply(entry, {"M": 64})
    assert entry == before


def test_hooks_reach_their_fields():
    """Each hook's return value lands on the field the table names for it."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, DTYPE, **_):
            return {"x_ptr": f"*{DTYPE}"}

        def reference(self, OP, **_):
            return f"oracle_{OP}"

        def inputs(self, DTYPE, **_):
            return f"inputs_{DTYPE}"

    entry = _apply({"factory": F()}, {"DTYPE": "fp16", "OP": "add"})
    assert entry["SIGNATURE"] == {"x_ptr": "*fp16"}
    assert entry["reference"] == "oracle_add"
    assert entry["inputs"] == "inputs_fp16"


def test_none_hook_leaves_its_field_unset():
    """``None`` leaves the field absent, so the protocol is opt-in per field.

    ``inputs`` is declared but declines for this combination; the undeclared
    ``reference`` inherits the ``None``-returning default.
    """

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **_):
            return {"x_ptr": "*fp32"}

        def inputs(self, DTYPE, **_):
            return None if DTYPE == "fp32" else "gen"

    entry = _apply({"factory": F()}, {"DTYPE": "fp32"})
    assert "SIGNATURE" in entry
    assert "inputs" not in entry
    assert "reference" not in entry


def test_hook_plus_literal_field_raises():
    """A hook and the literal field it produces is an error, not precedence.

    Which one won would otherwise be a fact about statement order in
    ``_apply_factory``.
    """

    @dataclass(frozen=True)
    class F(VariantFactory):
        def reference(self, **_):
            return "from_hook"

    with pytest.raises(ValueError, match="fix::variant.*'reference'.*reference"):
        _apply({"factory": F(), "reference": "literal"}, {"M": 64})


def test_undeclared_hook_tolerates_its_literal_field():
    """The collision is per hook, so a factory can supply one field and the
    variant another."""

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **_):
            return {"x_ptr": "*fp32"}

    entry = _apply({"factory": F(), "reference": "literal"}, {"M": 64})
    assert entry["reference"] == "literal"
    assert entry["SIGNATURE"] == {"x_ptr": "*fp32"}


def test_hooks_see_values_not_label_pairs():
    """Hooks get plain values; labels exist only to build registry keys.

    A labelled param would otherwise arrive as ``("stick", [...])`` and every
    hook would have to index ``[1]``.
    """
    seen = {}

    @dataclass(frozen=True)
    class F(VariantFactory):
        def signature(self, **combo):
            seen.update(combo)
            return {"x_ptr": "*fp32"}

    combo = {"DTYPE": ("f16", "fp16"), "LAYOUT": ("stick", [(1, "floordiv", 64)])}
    _apply_factory({"factory": F()}, combo, kernel_name="fix::variant")
    assert seen == {"DTYPE": "fp16", "LAYOUT": [(1, "floordiv", 64)]}


def test_bare_callable_on_factory_is_refused():
    """Not duck-typed: hooks are recognised by name on a ``VariantFactory``."""
    with pytest.raises(TypeError, match="must be a VariantFactory"):
        _apply({"factory": lambda **kw: None}, {"M": 64})


def test_kwargs_declaring_reference_is_passed_through_untouched():
    """Regression: a ``**kwargs``-declaring *oracle* is not a factory.

    ``fixtures/inter_tile_reduce/meta.py:86`` declares
    ``run_element_sum(inputs, BLOCK_M, BLOCK_N, NUM_N_TILES, **_kw)`` and binds the
    extras with ``functools.partial`` (``:317``). Any rule inferring
    combination-dependence from the shape of a callable — as ``extra_checks`` does
    — reads that as a factory, calls it with no ``inputs``, and collection dies
    with ``TypeError: run_element_sum() missing 1 required positional argument:
    'inputs'``. Measured, not hypothetical.
    """
    called = []

    def run_element_sum(inputs, BLOCK_M, BLOCK_N, NUM_N_TILES, **_kw):
        called.append(inputs)
        return inputs

    reference = functools.partial(
        run_element_sum, BLOCK_M=16, BLOCK_N=16, NUM_N_TILES=2)
    entry = _apply({"reference": reference}, {"BLOCK_M": 16, "BLOCK_N": 16})
    assert entry["reference"] is reference
    assert called == []
