import hashlib
import io
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# ---------------------------------------------------------------------------
# SpyreUtils
#
# ``utils`` and ``launcher_cls`` are NOT declared on DriverBase — it declares
# only is_active / map_python_to_cpp_type / get_current_target /
# get_active_torch_device / get_benchmarker / allocate_default_profile_scratch.
# They are an undeclared convention that ``compiler.py`` reaches for by name:
#
#   compiler.py:135  driver.active.utils.get_device_properties(dev)["max_shared_mem"]
#   compiler.py:445  driver.active.utils.unload_module(self.module)
#   compiler.py:464  driver.active.launcher_cls(self.src, self.metadata)
#   compiler.py:477  driver.active.utils.load_binary(name, kernel, shared, device)
#
# plus get_current_device() / get_current_stream(device) unconditionally at the
# top of JITFunction.run (jit.py:728). Every in-tree backend implements the same
# shape (CudaUtils / CudaLauncher), so this is the local spelling of an existing
# Triton role — but an upstream refactor moves the contract with no deprecation.
# ---------------------------------------------------------------------------

class SpyreUtils:
    """The ``driver.active.utils`` members Triton's runtime calls.

    Deliberately not a singleton. ``CudaUtils`` is one (``__new__`` plus a
    process-wide ``__init__``) only so that its ``driver.c`` is compiled once;
    there is nothing to compile here.
    """

    #: Sub-directory of ``knobs.cache.dir`` holding unpacked spyreCodeDirs.
    MODULE_CACHE = "spyre-modules"

    def get_device_properties(self, device):
        """One key, because one key is what Triton actually reads.

        ``CompiledKernel._init_handles`` uses ``max_shared_mem`` to bound
        ``metadata.shared`` (``compiler.py:467``). Spyre's LX scratchpad is not
        Triton shared memory — it is sized by ``SpyreOptions.lx_size`` and
        allocated by the scheduler — so 0 is reported against ``shared = 0`` and
        the check is a no-op rather than a false floor.

        CUDA reports seven keys. The other six are read only by
        ``python/triton/testing.py``; inventing them would make that module
        print nonsense instead of failing, so they stay absent.
        """
        del device
        return {"max_shared_mem": 0}

    def load_binary(self, name, kernel, shared, device):
        """Unpack the compiled artifact and return Triton's 5-tuple.

        ``kernel`` is the ZIP produced by ``SpyreBackend._make_spyrecode``. It is
        extracted, content-addressed, under ``knobs.cache.dir``, and the
        directory is returned as **both** the module and the function: there is
        no loaded-module / entry-point split to mirror, because on Spyre the
        program *is* the directory. ``module`` is what ``unload_module``
        receives; ``function`` is what reaches the launcher.

        The unpack is flat — ``spyrecode.json`` and ``init_binary.bin`` at the
        top with dbo-opt's ``debug/`` beside them — because ``prepare_kernel``
        opens exactly ``<dir>/spyrecode.json`` and the ``init_bin_file`` that
        names, both by name and with no directory scan anywhere.

        Keyed on the **artifact digest**, not on ``name``: ``name`` is ``""``
        (issue #104 — ``get_entry_func_name`` dyn_casts to ``tt.func`` after
        ``convert_functions`` has made it ``func.func``), and
        ``CompiledKernel.hash`` is not passed in.

        ``n_regs`` / ``n_spills`` are meaningless here and reported as 0.
        ``n_max_threads`` is 1, which keeps
        ``num_warps * warp_size > n_max_threads`` (1 * 1 > 1) false
        (``compiler.py:480``).

        ``prepare_kernel`` is deliberately **not** called here, despite being
        once-per-kernel work. It needs an initialized Spyre runtime and SIGSEGVs
        without one, and ``_init_handles`` runs at subscript time — before this
        path has seen a tensor, and allocating a device tensor is what
        initializes the runtime. It belongs in the launcher, built lazily and
        cached per CompiledKernel, which is per-kernel just the same and keeps
        loading device-free.
        """
        del name, shared, device  # see the docstring: none of the three is usable
        digest = hashlib.sha256(kernel).hexdigest()
        root = Path(knobs.cache.dir) / self.MODULE_CACHE / digest
        if not (root / "spyrecode.json").is_file():
            # Extract into a private directory and then move it into place, so a
            # concurrent load never observes a half-written spyreCodeDir.
            staging = root.parent / f"tmp.pid_{os.getpid()}_{uuid.uuid4().hex}"
            staging.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(kernel)) as archive:
                archive.extractall(staging)
            try:
                os.rename(staging, root)
            except OSError:
                # Another process won the race. Its copy is equivalent, because
                # the directory name is the digest of these very bytes.
                shutil.rmtree(staging, ignore_errors=True)
        return (str(root), str(root), 0, 0, 1)

    def unload_module(self, module):
        """Keep the unpacked directory. Load-bearing, not an omission.

        CUDA calls ``cuModuleUnload`` here; there is no Spyre analogue to undo.
        The directory is content-addressed, so keeping it is a cache rather than
        a leak, and the ``debug/dfir.mlir`` inside it is the only on-disk record
        of what ran.
        """
        del module


# ---------------------------------------------------------------------------
# SpyreLauncher
# ---------------------------------------------------------------------------

class SpyreLauncher:
    """Pure-Python launcher over the nine-positional ABI ``jit.py`` calls.

    Pure Python is not a departure: no in-tree backend generates per-kernel C
    (``make_launcher`` does not exist in this tree) and CUDA compiles one static
    ``driver.c`` once, process-wide.

    ``__init__`` must touch **no** option field that ``SpyreOptions`` does not
    declare. ``CudaLauncher`` reads ``global_scratch_size``,
    ``profile_scratch_*``, ``launch_cooperative_grid`` and ``launch_pdl`` as hard
    attributes; any of those here is an immediate AttributeError.

    Of the nine positionals (``jit.py:776``) only ``function`` and ``*args``
    carry information: ``stream`` is a stub, ``packed_metadata`` is ``()``
    (``SpyreBackend.pack_metadata``) and both hooks are ``None`` unless a
    profiler installed them.
    """

    def __init__(self, src, metadata):
        self.src = src
        self.metadata = metadata

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata,
                 launch_metadata, enter_hook, exit_hook, *args):
        del stream, packed_metadata, launch_metadata, enter_hook, exit_hook
        raise NotImplementedError(
            "Spyre kernel launch is not implemented yet. SpyreLauncher was "
            f"reached with {len(args)} argument(s), tile count "
            f"({gridX}, {gridY}, {gridZ}) and the compiled program at "
            f"{function}. The launch itself needs prepare_kernel / "
            "launch_jobplan from torch-spyre and a device."
        )


class SpyreDriver(DriverBase):
    """Spyre device driver.

    Sits on ``DriverBase`` rather than ``GPUDriver`` because
    ``GPUDriver.__init__`` hard-imports ``torch.cuda``
    (``python/triton/backends/driver.py:161``).

    Absent rather than stubbed: ``get_device_interface``,
    ``allocate_default_profile_scratch`` (inherits the raising base),
    ``get_empty_cache_for_benchmark``, ``clear_cache``, ``set_current_device``;
    ``get_benchmarker`` keeps raising. The cost, stated so it is a choice and not
    a discovery: ``triton.autotune``, ``do_bench``, Proton profiling and
    everything in ``python/triton/testing.py`` do not work.
    """

    def __init__(self) -> None:
        super().__init__()
        # Mirrors CudaDriver.__init__ (nvidia/backend/driver.py:340-342): the two
        # members Triton's runtime reaches for by name are assigned here.
        self.utils = SpyreUtils()
        self.launcher_cls = SpyreLauncher

    @classmethod
    def is_active(cls) -> bool:
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i32": "int32_t",
            "f16": "half",
            "fp8": "fp8",
        }
        return mapping.get(ty, ty)

    def get_current_target(self) -> GPUTarget:
        # warp_size = 1 keeps num_warps * warp_size at 1, which
        # SpyreUtils.load_binary's n_max_threads = 1 does not exceed.
        return GPUTarget(backend="spyre", arch=1, warp_size=1)

    def get_active_torch_device(self):
        return None

    def get_current_device(self) -> int:
        """Single device. Called unconditionally at the top of JITFunction.run
        (``jit.py:728``) and used to index the per-device compile caches."""
        return 0

    def get_current_stream(self, device) -> int:
        """Streams are not modelled. The value is threaded through to the
        launcher's ``stream`` positional, which ignores it."""
        del device
        return 0

    def get_benchmarker(self):
        raise NotImplementedError("Spyre does not support local benchmarking")
