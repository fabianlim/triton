import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import zipfile

from triton import knobs
from triton._utils import BITWIDTH_DICT
from triton.backends.compiler import BaseBackend, GPUTarget
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple
from types import ModuleType

# ---------------------------------------------------------------------------
# Fixed HBM address policy (issue #100)
#
# Pointer argument i is based at i * _SLOT_BYTES. This is the same assignment
# torch-spyre's Inductor path makes (SEGMENT_OFFSETS), so the two producers of
# a Spyre binary agree by construction rather than by coincidence. Slot 7 holds
# the program itself, which is why only 7 pointer arguments fit.
# ---------------------------------------------------------------------------
_SLOT_BYTES = 16 * 1024 ** 3
_MAX_POINTER_ARGS = 7


def _elem_bytes(pointee: str) -> int:
    """Bytes per element for a Triton pointee spelling ("fp16", "i32", ...).

    Derived from ``triton._utils.BITWIDTH_DICT`` so there is one authority for
    element widths. Unknown or sub-byte types raise rather than being guessed:
    a base address wrong by a factor of the element width reads unrelated HBM
    and returns plausible garbage with no diagnostic.
    """
    bits = BITWIDTH_DICT.get(pointee)
    if not bits or bits % 8:
        raise ValueError(
            f"pointer element type {pointee!r} has no usable byte width "
            f"(BITWIDTH_DICT gives {bits!r}); Spyre base addresses are element "
            "indices, so the width must be a whole number of bytes"
        )
    return bits // 8


def resolve_dbo_opt(required: bool = True) -> Optional[str]:
    """Absolute path to the ``dbo-opt`` named by ``knobs.spyre.dbo_opt``.

    A value containing a path separator is taken literally; a bare name is
    looked up on ``PATH``. Returns ``None`` instead of raising when
    ``required`` is False, which is what :meth:`SpyreBackend.hash` wants: a
    missing tool must still produce a cache key (and a distinct one), with the
    actionable error raised later by the stage that actually needs the tool.
    """
    tool = knobs.spyre.dbo_opt
    path = tool if os.sep in tool else shutil.which(tool)
    if path is None or not os.path.isfile(path):
        if not required:
            return None
        raise RuntimeError(
            f"dbo-opt not found: {tool!r}. Set TRITON_SPYRE_DBO_OPT (or "
            "knobs.spyre.dbo_opt) to the dbo-opt binary that should compile "
            "KTIR to SpyreCode."
        )
    return os.path.realpath(path)


def base_addresses_for(signature_types) -> Tuple[int, ...]:
    """Fixed base addresses, in ELEMENTS, for the pointer entries of a signature.

    ``signature_types`` is an iterable of Triton type spellings in argument
    order; entries starting with ``*`` are the pointer arguments and everything
    else (runtime scalars, ``constexpr``) is skipped. The i-th pointer gets
    ``i * _SLOT_BYTES`` — converted to an element index because that is the unit
    ``ktdp.construct_memory_view``'s ``$offset`` carries (see
    MaterializeBaseAddresses.cpp).

    Raises when the kernel has more than :data:`_MAX_POINTER_ARGS` pointers, or
    when a pointee type has no usable element width.
    """
    ptr_types = [ty for ty in signature_types
                 if isinstance(ty, str) and ty.startswith("*")]
    if len(ptr_types) > _MAX_POINTER_ARGS:
        raise ValueError(
            f"Spyre supports at most {_MAX_POINTER_ARGS} pointer arguments "
            f"(slot {_MAX_POINTER_ARGS} holds the program itself); this kernel "
            f"has {len(ptr_types)}: {ptr_types}"
        )
    # "*k" marks a const pointer and "*" a plain one; neither changes the
    # element width (see triton._utils.canonicalize_ptr_dtype).
    pointees = [ty[2:] if ty.startswith("*k") else ty[1:] for ty in ptr_types]
    return tuple(i * _SLOT_BYTES // _elem_bytes(pointee)
                 for i, pointee in enumerate(pointees))

# The TTIR→KTIR core pass sequence, as binding names on
# spyre.passes.ttir_to_ktdp.
_CORE_PIPELINE_PASSES = (
    "lower_descriptor_memory",
    "lower_scalar_load",
    "lower_compute_ops",
    "rewrite_descriptor_layout",
    "lower_inter_tile",
    "convert_functions",
)

# Passes that need more than the pass manager, as {pass name: SpyreOptions
# fields to forward}. Names must match the py::arg names on the binding in
# third_party/spyre/triton_spyre.cc and the field names on SpyreOptions, since
# they are passed as keyword arguments.
_PASS_OPTIONS = {
    "distribute_work": ("grid",),
    "materialize_base_addresses": ("base_addresses",),
    "rewrite_descriptor_layout": ("data_layout",),
    "convert_ttir_to_ktdp": ("data_layout",),
}


@dataclass
class SpyreOptions:
    # Per-axis partition of the Spyre hardware grid. One entry per
    # tl.program_id axis the kernel reads; prod(grid) is the total
    # physical core count. Default covers the common 1D-on-32-cores
    # case. A 2D kernel with grid = (16, 2) would partition the same
    # 32 cores as 16x2 across axes x and y.
    grid: Tuple[int, ...] = (32,)

    # Fixed HBM base addresses for the kernel's pointer arguments, positionally:
    # entry i is the address for the i-th `index` argument of the lowered
    # function, which ConvertFunctions produced from the i-th !tt.ptr argument.
    # When set, MaterializeBaseAddresses replaces those arguments with
    # arith.constant and drops them from the signature, for feeding the dataflow
    # scheduler. Empty (default) leaves the signature untouched.
    #
    # Values are ELEMENT indices, not byte addresses — ktdp.construct_memory_view's
    # offset feeds MemRef.base_ptr, and ktir-cpu computes the byte position as
    # base_ptr * bytes_per_elem(dtype). No scaling is applied on the way through.
    base_addresses: Tuple[int, ...] = ()

    # Optional correctness patches to splice into the TTIR→KTIR pipeline, as
    # {fix pass name: core pass it runs after}. Both are binding names on
    # spyre.passes.ttir_to_ktdp.
    #
    #   required_fixes = {"convert_elementwise_to_linalg": "lower_compute_ops"}
    #
    # Choose the anchor by what the fix depends on. A pass that repairs IR its
    # anchor produces must run after it, and anchoring earlier silently does
    # nothing — unalias_linalg_outs anchors on convert_elementwise_to_linalg for
    # exactly that reason. Other fixes may instead need to land before a later
    # consumer.
    #
    # The anchor must be one of _CORE_PIPELINE_PASSES. Any other name — a typo,
    # or a plausible-looking "distribute_work" — is silently ignored and the fix
    # never runs. A missing pass *binding* raises; a bad *anchor* does not.
    required_fixes: Mapping[str, str] = field(default_factory=dict)
    lx_size: int = 2 * 1024 * 1024  # 2 MB scratchpad per core
    # HBM data layout: "device" (stickified row-major physical strides) or
    # "host" (strides derived from logical strides via the coordinate map).
    data_layout: str = "device"
    # Required by Triton code generator
    sanitize_overflow: bool = False
    debug: bool = False
    allowed_dot_input_precisions: tuple = ("ieee",)

    # --- Fields Triton's runtime requires but Spyre has no analogue for ---
    #
    # Both are fictions, chosen so the ceiling checks in
    # CompiledKernel._init_handles become no-ops rather than false floors. The
    # third value in the same set, n_max_threads, is reported by
    # SpyreUtils.load_binary in driver.py:
    #
    #   num_warps  -- compared as `num_warps * warp_size > n_max_threads`
    #                 (python/triton/compiler/compiler.py:480). With
    #                 warp_size = 1 from SpyreDriver.get_current_target() and
    #                 n_max_threads = 1 from SpyreUtils.load_binary, 1*1 > 1 is
    #                 false. Spyre has no warps.
    #   shared     -- compared as `shared > max_shared_mem` (:467), and
    #                 SpyreUtils.get_device_properties reports 0, so 0 > 0 is
    #                 false. Spyre's LX scratchpad is not Triton shared memory;
    #                 it is sized by lx_size and allocated by the scheduler.
    #
    # An upstream refactor of _init_handles breaks these silently, hence the
    # explicit note here and at each reported value.
    num_warps: int = 1
    shared: int = 0

    # JITFunction.run injects instrumentation_mode into kwargs unconditionally
    # (python/triton/runtime/jit.py:725) and _pack_args rejects any launch kwarg
    # that is not a field of the parsed options (:711), so every launch fails
    # that check unless the field exists. Spyre runs no instrumentation passes;
    # the value is accepted and ignored.
    instrumentation_mode: str = ""

    def __post_init__(self):
        # Normalize list → tuple for hashability / dataclass equality.
        if isinstance(self.grid, list):
            self.grid = tuple(self.grid)
        if isinstance(self.base_addresses, list):
            self.base_addresses = tuple(self.base_addresses)
        # RewriteDescriptorLayout treats any value other than "device" as
        # "host", so an unrecognized string would silently pick a layout
        # rather than fail.
        if self.data_layout not in ("device", "host"):
            raise ValueError(
                f"data_layout must be 'device' or 'host', got {self.data_layout!r}")

    def hash(self):
        key = "_".join(f"{name}-{val}" for name, val in sorted(self.__dict__.items()))
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SpyreBackend(BaseBackend):
    """Spyre AI accelerator backend for Triton.

    Compiles Triton TTIR to KTIR (KTDP dialect IR) for the IBM Spyre accelerator.
    """

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "spyre"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # Set here, not in add_stages: CompiledKernel constructs a *fresh*
        # backend via make_backend() (python/triton/compiler/compiler.py:418),
        # so an attribute assigned on the compiling instance is not the one it
        # reads. It also decides bytes-vs-text per artifact (:427) — the file
        # whose extension matches binary_ext is read as bytes.
        self.binary_ext = "spyrecode"

    def hash(self) -> str:
        """Backend identity folded into the on-disk cache key.

        Includes the ``dbo-opt`` binary and the device description, because both
        change the emitted artifact while leaving the source, the signature and
        the options untouched. Without them, repointing or rebuilding dbo-opt
        silently reuses a stale binary. NVIDIA closes the same hole by folding
        ``get_ptxas_version(arch)`` into its options
        (``third_party/nvidia/backend/compiler.py:574``).
        """
        return "-".join([
            f"spyre-{self.target.arch}",
            self._dbo_opt_identity(),
            self._device_identity(),
            f"dbo_debug-{int(bool(knobs.spyre.dbo_debug))}",
        ])

    @staticmethod
    def _dbo_opt_identity() -> str:
        """``dbo-opt`` identity: resolved path plus size and mtime.

        dbo-opt has no meaningful ``--version`` (it reports the LLVM version it
        was linked against, which is identical across rebuilds), so size+mtime
        of the resolved path is what actually distinguishes two builds. A
        missing tool gets its own key rather than raising — see resolve_dbo_opt.
        """
        path = resolve_dbo_opt(required=False)
        if path is None:
            return f"dbo_opt-missing-{knobs.spyre.dbo_opt}"
        st = os.stat(path)
        return f"dbo_opt-{path}-{st.st_size}-{st.st_mtime_ns}"

    @staticmethod
    def _device_identity() -> str:
        """Digest of the device description, or an explicit "no file" marker.

        ``knobs.spyre.device`` is optional: unset, dbo-opt falls back to its own
        default under ``$DEEPTOOLS_PATH``. In that case there is no file to
        digest, so we say so instead of hashing an empty string as though it
        were a device — a real device file must never collide with "default".
        """
        device = knobs.spyre.device
        if not device:
            return "device-dbo_opt_default"
        path = Path(device)
        if not path.is_file():
            # Report it rather than hash nothing; _make_spyrecode raises with a
            # usable message when it actually needs the file.
            return f"device-{device}-missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return f"device-{device}-{digest}"

    def parse_options(self, options: dict) -> SpyreOptions:
        return SpyreOptions(**{k: v for k, v in options.items()
                               if k in SpyreOptions.__dataclass_fields__})

    def compile_time_launch_options(self, grid, specialization) -> dict:
        """Contribute the grid and the base addresses as *compile* inputs.

        On Spyre both are baked into the artifact, so they must be part of the
        options before the cache key is computed. Neither can be produced by
        ``parse_options``: the grid is consumed by ``JITFunction.run``'s own
        keyword-only parameter and never reaches ``**options``, and the
        addresses are derived from the specialized signature, which only exists
        once the argument binder has run. See the hook's default in
        ``python/triton/backends/compiler.py`` and its single call site in
        ``python/triton/runtime/jit.py``.
        """
        extra = {"base_addresses": base_addresses_for(ty for ty, _ in specialization)}
        if grid is None:
            # warmup=True: there is no grid to bake, leave the option default.
            return extra
        if callable(grid):
            raise ValueError(
                "Spyre bakes the grid into the compiled artifact, so a callable "
                "grid cannot be used: the tile count must be known before the "
                "kernel is compiled. Pass a concrete tuple, e.g. kernel[(32,)]."
            )
        extra["grid"] = tuple(grid)
        return extra

    def add_stages(self, stages: dict, options: SpyreOptions, language=None) -> None:
        stages["ttir"] = lambda src, metadata: self._make_ttir(src, metadata, options)
        stages["ktir"] = lambda src, metadata: self._make_ktir(src, metadata, options)
        stages["spyrecode"] = lambda src, metadata: self._make_spyrecode(src, metadata, options)

    def load_dialects(self, context) -> None:
        from triton._C.libtriton import spyre
        spyre.load_dialects(context)

    def get_codegen_implementation(self, options):
        """Return codegen hooks queried by the Triton frontend.

        Called from ``python/triton/compiler/compiler.py::compile_to_ttir``
        (line 304). The returned dict is threaded through to the
        language builder and consumed by individual frontend helpers;
        today only ``min_dot_size`` is required, enforced around the
        ``tl.dot`` shape check in
        ``python/triton/language/semantic.py`` (lines 1453-1458).

        ``min_dot_size(lhs_type, rhs_type)`` returns ``(min_M, min_N,
        min_K)`` lower bounds that a ``tl.dot`` call's operand shapes
        must satisfy. Upstream backends pick:

          - NVIDIA (``third_party/nvidia/backend/compiler.py:19``) —
            ``(1, 1, 16)`` for fp16/bf16, ``(1, 1, 32)`` for int8/fp8.
            Only K is constrained; small M/N are padded into tensor
            cores.
          - AMD (``third_party/amd/backend/compiler.py:16``) —
            ``(1, 1, 1)``, falling back to FMA for configurations not
            natively supported by its matrix cores.
          - Interpreter (``python/triton/runtime/interpreter.py:290``) —
            ``(1, 1, 1)``.

        Spyre has no ``tl.dot`` shape floor today (``linalg.matmul``
        handles arbitrary tile sizes), so we return ``(1, 1, 1)``
        matching AMD and the interpreter. Revisit this if a future
        KTIR matmul path needs a minimum.
        """
        return {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    def pack_metadata(self, metadata):
        return ()

    def _make_ttir(self, mod, metadata, options):
        """Run standard Triton TTIR optimization passes."""
        from triton._C.libtriton import ir, passes

        pm = ir.pass_manager(mod.context)
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "make_ttir")

        metadata["stage"] = "ttir"
        return mod

    def _make_ktir(self, mod, metadata, options):
        """Lower optimized TTIR to KTIR using C++ MLIR passes.

        Pipeline steps, in the order they are added below:

        _CORE_PIPELINE_PASSES, each optionally followed by fixes anchored to it via
        options.required_fixes:
          - LowerDescriptorMemory: tt.descriptor_load/store/gather/scatter -> ktdp.*
          - LowerScalarLoad: scalar tt.load (+ addptr chain) -> ktdp.* rank-0 read
          - LowerComputeOps: tt.reduce/broadcast/expand_dims -> linalg/tensor
            + dead op sweep
          - RewriteDescriptorLayout: logical tensor descriptors -> physical
            (stick-tiled) layout from tt.spyre_tensor_layout annotations
          - LowerInterTile: tt.inter_tile_reduce -> ktdp.inter_tile_produce + delivery
          - ConvertFunctions: tt.func/return -> func.func/return, !tt.ptr -> index
            (last of the core passes — the memory passes above consume !tt.ptr
            args via getBasePtrAsIndex)

        then:
          - DistributeWork: tt.get_program_id -> ktdp.get_compute_tile_id
          - canonicalize + CSE
        """
        from triton._C.libtriton import ir, passes, spyre

        fixes = options.required_fixes
        ttir_to_ktdp = spyre.passes.ttir_to_ktdp

        def add_pass(name):
            # Passes are exposed as add_<name>, taking the pass manager plus any
            # SpyreOptions fields named in _PASS_OPTIONS. A missing binding
            # means a requested fix would silently never run, so raise.
            adder = getattr(ttir_to_ktdp, f"add_{name}", None)
            if adder is None:
                raise ValueError(
                    f"no pass binding 'add_{name}' on "
                    "spyre.passes.ttir_to_ktdp; declare it in "
                    "third_party/spyre/triton_spyre.cc and rebuild libtriton"
                )
            adder(pm, **{opt: getattr(options, opt)
                         for opt in _PASS_OPTIONS.get(name, ())})

        pm = ir.pass_manager(mod.context)
        if fixes:
            # Compose pass-by-pass so each fix lands after its anchor.
            for core_pass in _CORE_PIPELINE_PASSES:
                add_pass(core_pass)
                for fix, anchor in fixes.items():
                    if anchor == core_pass:
                        add_pass(fix)
        else:
            add_pass("convert_ttir_to_ktdp")
        add_pass("distribute_work")
        # Clean up redundant arithmetic (fold muli x,1; simplify cast chains)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        pm.run(mod, "make_ktir")

        metadata["name"] = mod.get_entry_func_name()
        metadata["stage"] = "ktir"
        return mod

    def _make_spyrecode(self, mod, metadata, options):
        """Lower KTIR to a loadable Spyre binary by running ``dbo-opt``.

        Returns the spyreCodeDir as **ZIP bytes**. A compile stage yields one
        artifact, but a spyreCodeDir is two files (``spyrecode.json`` +
        ``init_binary.bin``) plus dbo-opt's ``debug/`` tree, so the archive is
        the single artifact and ``SpyreUtils.load_binary`` unpacks it. Layout
        inside the ZIP is flat — spyreCodeDir's own contents at the root, with
        ``debug/`` as a subdirectory — because ``prepare_kernel`` opens
        ``<dir>/spyrecode.json`` and the ``init_bin_file`` it names, both by
        name and with no directory scan.

        Two steps:

        1. ``MaterializeBaseAddresses``, which replaces the pointer arguments
           with ``arith.constant`` and drops them from the signature. The
           dataflow scheduler requires a zero-argument entry function. This runs
           here rather than in ``_make_ktir`` so the cached ``.ktir`` artifact
           keeps the argument-passing calling convention.
        2. ``dbo-opt --from-ktir --kEmitSpyreCode``, whose scheduler +
           codegen stages write the spyreCodeDir. ``--kEmitSpyreCode`` is a pass
           pipeline that has to be requested explicitly; ``--export-dir`` alone
           makes dbo-opt exit 0 having written nothing.

        The scheduler additionally requires the compute to be a ``linalg`` op
        with an unaliased ``outs``, which this stage does not arrange: it is a
        property of the KTIR handed to it, produced by the
        ``convert_elementwise_to_linalg`` / ``unalias_linalg_outs`` entries of
        ``SpyreOptions.required_fixes``. Without them dbo-opt rejects the
        ``ktdp.load`` operand, because the memref keeps a dynamic
        ``strided<..., offset: ?>`` layout until a ``linalg`` consumer pins it.
        """
        from triton._C.libtriton import ir, passes, spyre

        if options.base_addresses:
            pm = ir.pass_manager(mod.context)
            spyre.passes.ttir_to_ktdp.add_materialize_base_addresses(
                pm, base_addresses=list(options.base_addresses))
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            pm.run(mod, "make_spyrecode")

        dbo_opt = resolve_dbo_opt()
        device = knobs.spyre.device
        if device and not Path(device).is_file():
            raise FileNotFoundError(
                f"knobs.spyre.device / TRITON_SPYRE_DEVICE points at {device!r}, "
                "which does not exist. Leave it unset to let dbo-opt use its "
                "own default device under $DEEPTOOLS_PATH."
            )

        with tempfile.TemporaryDirectory(prefix="spyrecode_") as tmp:
            tmp = Path(tmp)
            ktir_path = tmp / "kernel.ktir"
            ktir_path.write_text(str(mod))
            export_dir = tmp / "export"
            export_dir.mkdir()

            argv = [dbo_opt, "--from-ktir", "--kEmitSpyreCode",
                    f"--export-dir={export_dir}"]
            if device:
                argv.append(f"--device={device}")
            argv.append(str(ktir_path))

            env = dict(os.environ)
            if knobs.spyre.dbo_debug:
                env["DBO_DEBUG"] = "1"
            # dbo-opt prints the optimized module on stdout; only the files it
            # writes under --export-dir matter here.
            result = subprocess.run(argv, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"dbo-opt failed (exit {result.returncode}):\n"
                    f"  argv: {' '.join(argv)}\n{result.stderr}"
                )

            code_dir = export_dir / "spyreCodeDir"
            # dbo-opt can exit 0 having written nothing, so check rather than
            # trust the exit status.
            missing = [name for name in ("spyrecode.json", "init_binary.bin")
                       if not (code_dir / name).is_file()]
            if missing:
                raise RuntimeError(
                    f"dbo-opt exited 0 but did not write {', '.join(missing)} "
                    f"under {code_dir}\n  argv: {' '.join(argv)}\n{result.stderr}"
                )

            members = [(path, path.relative_to(code_dir).as_posix())
                       for path in sorted(code_dir.rglob("*")) if path.is_file()]
            debug_dir = export_dir / "debug"
            members += [(path, f"debug/{path.relative_to(debug_dir).as_posix()}")
                        for path in sorted(debug_dir.rglob("*")) if path.is_file()]

            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path, name in members:
                    # Fixed timestamp and mode: the artifact digest is the key
                    # SpyreUtils.load_binary unpacks under, so identical inputs
                    # must give identical bytes. Real mtimes would make every
                    # recompile look like a new binary.
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = 0o644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())

        metadata["stage"] = "spyrecode"
        return buffer.getvalue()
