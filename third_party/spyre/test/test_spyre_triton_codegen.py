# Copyright 2025 The Torch-Spyre Authors.
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

"""
Tests for the SpyreTritonKernel codegen path on CPU (no Spyre device).

Each test case:
  1. Compiles a PyTorch function via SpyreTritonScheduling on cpu.
  2. Reads the generated output_code.py from torch_compile_debug/.
  3. Parses out the Triton kernel source strings and the call() body.
  4. Runs structural assertions on the generated code.

Run with:
    .venv/bin/python -m pytest test_spyre_triton_codegen.py -v
"""

import importlib.util
import os
import sys
import re
import enum
import glob
import types
import shutil
import unittest

import pytest

# torch_spyre cannot be pip-installed in isolation (its _C extension requires
# the Spyre hardware SDK). Point TORCH_SPYRE_PATH at a local checkout of
# https://github.com/tnakaike/torch-spyre (dev/triton branch) to run these
# tests; the _C extension is pre-stubbed below so no hardware SDK is needed.
_torch_spyre_path = os.environ.get("TORCH_SPYRE_PATH")
if _torch_spyre_path:
    sys.path.insert(0, _torch_spyre_path)

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

if importlib.util.find_spec("torch_spyre") is None:
    pytest.skip(
        "torch_spyre not found — set TORCH_SPYRE_PATH to a local checkout of "
        "https://github.com/tnakaike/torch-spyre (dev/triton branch)",
        allow_module_level=True,
    )

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# ---------------------------------------------------------------------------
# Stubs — installed before any torch_spyre import.
# See test_inductor_triton_codegen.py for full rationale.
# ---------------------------------------------------------------------------

class DataFormats(enum.Enum):
    INVALID = 0; SEN169_FP16 = 1; IEEE_FP32 = 2; IEEE_FP16 = 3
    BFLOAT16 = 4; IEEE_INT32 = 5; IEEE_INT64 = 6; BOOL = 7
    SEN143_FP8 = 8; SEN152_FP8 = 9; SEN153_FP9 = 10; SEN18F_FP24 = 11
    SENINT2 = 12; SENINT4 = 13; SENINT8 = 14; SENINT16 = 15
    SENINT24 = 16; SENUINT2 = 17; SENUINT32 = 18

class SpyreTensorLayout:
    def __init__(self, dtype=None, sizes=None, strides=None):
        self.dtype = dtype; self.sizes = sizes or []; self.strides = strides or []

class ElementArrangement(enum.Enum):
    STANDARD = 0; EXX2 = 1; QFP8CH = 2; DL16_TO_FP32 = 3

def get_elem_in_stick(dtype):
    return {DataFormats.SEN169_FP16: 64, DataFormats.IEEE_FP16: 64,
            DataFormats.BFLOAT16: 64, DataFormats.IEEE_FP32: 32,
            DataFormats.IEEE_INT32: 32, DataFormats.IEEE_INT64: 16,
            DataFormats.SENINT8: 128}.get(dtype, 64)

def get_device_dtype(torch_dtype):
    import torch
    return {torch.float16: DataFormats.SEN169_FP16, torch.float32: DataFormats.IEEE_FP32,
            torch.bfloat16: DataFormats.BFLOAT16, torch.int32: DataFormats.IEEE_INT32,
            torch.int64: DataFormats.IEEE_INT64, torch.int8: DataFormats.SENINT8,
            torch.bool: DataFormats.BOOL}.get(torch_dtype, DataFormats.INVALID)

def encode_constant(v, d): return v
def spyre_empty_with_layout(*a, **k): raise NotImplementedError
def reinterpret_tensor(*a, **k): raise NotImplementedError
def reinterpret_tensor_with_layout(*a, **k): raise NotImplementedError

_C_stub = types.ModuleType("torch_spyre._C")
for _k, _v in dict(DataFormats=DataFormats, SpyreTensorLayout=SpyreTensorLayout,
                   ElementArrangement=ElementArrangement, get_elem_in_stick=get_elem_in_stick,
                   get_device_dtype=get_device_dtype, encode_constant=encode_constant,
                   spyre_empty_with_layout=spyre_empty_with_layout,
                   reinterpret_tensor=reinterpret_tensor,
                   reinterpret_tensor_with_layout=reinterpret_tensor_with_layout).items():
    setattr(_C_stub, _k, _v)
sys.modules["torch_spyre._C"] = _C_stub
sys.modules["torch_spyre._hooks"] = types.ModuleType("torch_spyre._hooks")

_ops_fallbacks_stub = types.ModuleType("torch_spyre.ops.fallbacks")
_ops_fallbacks_stub.fallback_ops = []
_ops_fallbacks_stub.warn_fallback = lambda *a, **k: None
sys.modules["torch_spyre.ops.fallbacks"] = _ops_fallbacks_stub

_ops_eager_stub = types.ModuleType("torch_spyre.ops.eager")
_ops_eager_stub.compile_once = lambda f: f
sys.modules["torch_spyre.ops.eager"] = _ops_eager_stub
sys.modules["torch_spyre._inductor.customops"] = types.ModuleType("torch_spyre._inductor.customops")

_lowering_stub = types.ModuleType("torch_spyre._inductor.lowering")
_lowering_stub.spyre_lowerings = {}
_lowering_stub.enable_spyre_lowerings = lambda: __import__("contextlib").nullcontext()
sys.modules["torch_spyre._inductor.lowering"] = _lowering_stub

import torch

_orig_register_kernel = torch.library.register_kernel
def _patched_register_kernel(op, device_types, func=None, *, lib=None, disable_dynamo=False):
    if isinstance(device_types, str):
        device_types = [device_types]
    if device_types is not None:
        device_types = [d for d in device_types if d != "spyre"]
        if not device_types:
            return func if func is not None else lambda f: f
    return _orig_register_kernel(op, device_types, func, lib=lib, disable_dynamo=disable_dynamo)
torch.library.register_kernel = _patched_register_kernel

from torch._inductor import config as inductor_config
from torch._inductor.codegen.common import register_backend_for_device, get_scheduling_for_device
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch_spyre._inductor_triton.spyre_triton_kernel import SpyreTritonKernel
from torch_spyre._inductor_triton.spyre_triton_scheduler import SpyreTritonScheduling

inductor_config.cpu_backend = "triton"
inductor_config.triton.native_matmul = True
inductor_config.force_disable_caches = True
get_scheduling_for_device("cpu")
register_backend_for_device("cpu", SpyreTritonScheduling, PythonWrapperCodegen)

# ---------------------------------------------------------------------------
# Stub triton compilation and kernel execution so the compile pipeline runs
# cleanly to completion (output_code.py written, debug log files closed) without
# needing an actual Spyre or GPU backend.
#
# Two stubs:
#   1. CachingAutotuner.precompile — skip triton.compile (no CPU backend).
#      Instead install a single no-op launcher so .run() doesn't loop.
#   2. CachingAutotuner.run — skip kernel execution entirely.
#      We only care about the generated source, not execution results.
#
# This prevents the "Triton compilation failed" error log and the ResourceWarning
# from unclosed debug log file handles (which happen when an exception aborts
# Inductor's context manager cleanup).
# ---------------------------------------------------------------------------

import logging as _logging
import triton as _triton
from torch._inductor.runtime.triton_heuristics import CachingAutotuner as _CachingAutotuner

_triton_heuristics_logger = _logging.getLogger("torch._inductor.runtime.triton_heuristics")
_triton_heuristics_logger.addFilter(
    lambda r: "Triton compilation failed" not in r.getMessage()
)

class _NoOpLauncher:
    config = type("cfg", (), {"found_by_coordesc": False, "kwargs": {}})()
    store_cubin = False
    def __call__(self, *args, **kwargs): pass

def _stub_precompile(self, *args, **kwargs):
    if not self.launchers:
        self.launchers = [_NoOpLauncher()]

def _stub_run(self, *args, stream=None, **kwargs):
    if not self.launchers:
        _stub_precompile(self)

_CachingAutotuner.precompile = _stub_precompile
_CachingAutotuner.run = _stub_run

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_DEBUG_ROOT = os.path.join(os.getcwd(), "torch_compile_debug")

_TRITON_BLOCK_RE = re.compile(
    r"async_compile\.triton\([^,]+,\s*'''(.*?)'''",
    re.DOTALL,
)
_JIT_FUNC_RE = re.compile(
    r"(@triton\.jit.*?^def \w+\(.*?)(?=\n@triton\.jit|\nclass |\nasync_compile|\Z)",
    re.DOTALL | re.MULTILINE,
)
_CALL_BODY_RE = re.compile(
    r"def call\(self, args\):(.*?)(?=\n    def |\nrunner|\Z)",
    re.DOTALL,
)


def _latest_output_code() -> str:
    matches = glob.glob(f"{_DEBUG_ROOT}/*/torchinductor/*/output_code.py")
    if not matches:
        raise FileNotFoundError("No output_code.py under torch_compile_debug/")
    return max(matches, key=os.path.getmtime)


class CompiledOutput:
    """Parsed view of one output_code.py.

    Attributes:
        path:           absolute path to the file
        source:         full file text
        kernel_sources: list of triton source strings (text inside async_compile.triton(..., '''...'''))
        jit_functions:  list of individual @triton.jit function texts extracted from kernel_sources
        call_body:      text of the Runner.call() method (buffer allocation, kernel launches, return)
    """

    def __init__(self, path: str):
        self.path = path
        with open(path) as f:
            self.source = f.read()
        self.kernel_sources: list[str] = _TRITON_BLOCK_RE.findall(self.source)
        self.jit_functions: list[str] = []
        for ks in self.kernel_sources:
            self.jit_functions.extend(_JIT_FUNC_RE.findall(ks))
        m = _CALL_BODY_RE.search(self.source)
        self.call_body: str = m.group(1) if m else ""

    def kernel_names(self) -> list[str]:
        return re.findall(r"^def (\w+)\(", "\n".join(self.jit_functions), re.MULTILINE)

    def signatures(self) -> dict[str, list[str]]:
        """Map kernel name -> list of parameter names (without type annotations)."""
        out = {}
        for fn in self.jit_functions:
            m = re.search(r"^def (\w+)\((.*?)\)", fn, re.MULTILINE)
            if m:
                params = [p.strip().split(":")[0].strip() for p in m.group(2).split(",")]
                out[m.group(1)] = params
        return out

    def spyre_grids(self) -> dict:
        """Extract spyre_grid / spyre_grids values from triton_meta."""
        import ast
        grids = {}
        for ks in self.kernel_sources:
            for m in re.finditer(r"'spyre_grids':\s*(\{[^}]+\})", ks):
                try: grids.update(ast.literal_eval(m.group(1)))
                except Exception: pass
            for m in re.finditer(r"'spyre_grid':\s*(\([^)]+\))", ks):
                try: grids["_"] = ast.literal_eval(m.group(1))
                except Exception: pass
        return grids

    def allocated_buffers(self) -> list[str]:
        return re.findall(r"(buf\d+)\s*=\s*empty_strided_cpu", self.call_body)

    def returned_buffers(self) -> list[str]:
        m = re.search(r"return \(([^)]+)\)", self.call_body)
        if not m:
            return []
        return [b.strip() for b in m.group(1).split(",") if b.strip()]


_VERBOSE = os.environ.get("TRITON_CODEGEN_VERBOSE", "0") == "1"


def compile_and_parse(fn, *inputs) -> CompiledOutput:
    """Compile fn through SpyreTritonScheduling and return a parsed CompiledOutput."""
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor")
    compiled(*inputs)  # runs cleanly; kernel execution is stubbed to a no-op
    return CompiledOutput(_latest_output_code())


# ---------------------------------------------------------------------------
# Tests — one class per op, compiled once in setUpClass.
# ---------------------------------------------------------------------------

class _OpTestBase(unittest.TestCase):
    """Base class: compile once per class, print kernel source if TRITON_CODEGEN_VERBOSE=1."""
    out: "CompiledOutput"

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(_DEBUG_ROOT, ignore_errors=True)
        cls.out = cls._compile()
        if _VERBOSE:
            print(f"\n{'='*60}\n{cls.__name__}\n{'='*60}")
            for i, src in enumerate(cls.out.kernel_sources):
                print(f"[kernel {i}]\n{src.strip()}\n")

    @classmethod
    def _compile(cls) -> "CompiledOutput":
        raise NotImplementedError


class TestAdd(_OpTestBase):
    @classmethod
    def _compile(cls):
        a = torch.randn(512, 256, dtype=torch.float16)
        b = torch.randn(512, 256, dtype=torch.float16)
        return compile_and_parse(lambda a, b: a + b, a, b)

    def test_produces_one_kernel(self):
        self.assertEqual(len(self.out.jit_functions), 1)

    def test_signature_has_two_inputs_one_output(self):
        params = next(iter(self.out.signatures().values()))
        self.assertIn("in_ptr0", params)
        self.assertIn("in_ptr1", params)
        self.assertIn("out_ptr0", params)

    def test_output_buffer_is_fp16(self):
        self.assertIn("torch.float16", self.out.call_body)

    def test_returns_one_buffer(self):
        self.assertEqual(len(self.out.returned_buffers()), 1)

    def test_kernel_body_has_addition(self):
        self.assertIn("tmp0 + tmp1", self.out.jit_functions[0])

    def test_loads_upcast_to_fp32(self):
        self.assertIn(".to(tl.float32)", self.out.jit_functions[0])

    def test_has_spyre_grid(self):
        self.assertTrue(len(self.out.spyre_grids()) > 0)


class TestRelu(_OpTestBase):
    @classmethod
    def _compile(cls):
        x = torch.randn(512, 256, dtype=torch.float16)
        return compile_and_parse(lambda x: torch.relu(x), x)

    def test_produces_one_kernel(self):
        self.assertEqual(len(self.out.jit_functions), 1)

    def test_uses_maximum(self):
        self.assertIn("triton_helpers.maximum", self.out.jit_functions[0])

    def test_one_input_one_output(self):
        params = next(iter(self.out.signatures().values()))
        self.assertIn("in_ptr0", params)
        self.assertNotIn("in_ptr1", params)
        self.assertIn("out_ptr0", params)


class TestFusedMulAdd(_OpTestBase):
    @classmethod
    def _compile(cls):
        a = torch.randn(512, 256, dtype=torch.float16)
        b = torch.randn(512, 256, dtype=torch.float16)
        return compile_and_parse(lambda a, b: a * b + b, a, b)

    def test_produces_one_kernel(self):
        self.assertEqual(len(self.out.jit_functions), 1)

    def test_kernel_body(self):
        body = self.out.jit_functions[0]
        self.assertIn("tmp0 * tmp1", body)
        self.assertIn("tmp2 + tmp1", body)

    def test_single_output(self):
        self.assertEqual(len(self.out.returned_buffers()), 1)


class TestSumReduction(_OpTestBase):
    @classmethod
    def _compile(cls):
        x = torch.randn(512, 256, dtype=torch.float16)
        return compile_and_parse(lambda x: x.sum(dim=1), x)

    def test_produces_one_kernel(self):
        self.assertEqual(len(self.out.jit_functions), 1)

    def test_has_tl_sum_and_reduction_axis(self):
        body = self.out.jit_functions[0]
        self.assertIn("tl.sum", body)
        self.assertIn("r0_numel", body)

    def test_output_shape_is_reduced(self):
        # (512,) written as "(512, )" in generated code
        self.assertIn("(512, )", self.out.call_body)


class TestSoftmax(_OpTestBase):
    @classmethod
    def _compile(cls):
        x = torch.randn(512, 256, dtype=torch.float16)
        return compile_and_parse(lambda x: torch.softmax(x, dim=1), x)

    def test_produces_five_jit_functions(self):
        # 4 noinline sub-kernels + 1 bundle entry
        self.assertEqual(len(self.out.jit_functions), 5)

    def test_bundle_has_four_sub_kernels(self):
        names = self.out.kernel_names()
        self.assertEqual(len([n for n in names if "kernel_" in n]), 4)

    def test_bundle_has_exp(self):
        self.assertIn("tl.exp", "\n".join(self.out.jit_functions))

    def test_bundle_has_two_reductions(self):
        all_bodies = "\n".join(self.out.jit_functions)
        self.assertEqual(all_bodies.count("tl.sum") + all_bodies.count("max2"), 2)

    def test_entry_calls_all_sub_kernels(self):
        entry = self.out.jit_functions[-1]
        for i in range(4):
            self.assertIn(f"kernel_{i}(", entry)

    def test_final_output_is_fp16(self):
        self.assertIn("torch.float16", self.out.call_body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
