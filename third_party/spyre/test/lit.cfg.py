import importlib.util
import os
import shutil

import lit.formats
import lit.llvm

llvm_config = lit.llvm.llvm_config

config.name = "SpyreTriton"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.spyre_obj_root, "test")

llvm_config.with_system_environment([
    "HOME", "INCLUDE", "LIB", "TMP", "TEMP",
    "TRITON_SPYRE_DBO_OPT", "TRITON_SPYRE_DEVICE",
    # For the device tests. The launch happens in the test process itself, so
    # that process is the one that has to import torch_spyre: PYTHONPATH says
    # where it is and LD_LIBRARY_PATH is what its _C.so dlopens against. Both are
    # forwarded rather than reconstructed -- which torch-spyre tree this machine
    # uses is not ours to invent, and ld.so reads LD_LIBRARY_PATH when the test
    # process is exec'd, so only what lit passes through reaches it. See
    # ``python/lit.local.cfg``, which appends the test tree's own entries to
    # whatever arrives here.
    "PYTHONPATH", "LD_LIBRARY_PATH", "DEEPTOOLS_PATH",
    # Machine setup for the device runtime, exported by the host's Spyre
    # environment (/etc/ibm/spyre). Not reconstructable here and not ours to
    # invent -- AIU_WORLD_SIZE is a topology fact, and the runtime aborts with
    # RAS::CONFIGURATION::InvalidAIUWorldSizeVar rather than defaulting. Dropping
    # FLEX_DEVICE is worse than an error: the runtime silently falls back to a mock
    # device, which then rejects FLEX_COMPUTE=SENTIENT.
    "AIU_WORLD_SIZE", "FLEX_COMPUTE", "FLEX_DEVICE",
])

# A ``dbo-opt`` feature, so a test needing the backend compiler says
# ``REQUIRES: dbo-opt`` and lit reports Unsupported when there is none.
#
# That is the whole reason to gate at this level: these tests run pytest, and
# pytest exits 0 when it skips, so a gate living only inside the test shows up as
# an ordinary Passed -- a run that compiled nothing would look exactly like one
# that did.
#
# The real resolver is resolve_dbo_opt() in backend/compiler.py. This is not a
# second opinion but the same rule re-spelled -- a value containing a path
# separator is taken literally, a bare name is looked up on PATH -- because
# importing the backend here would pull triton into lit's config phase. Keep the
# two in step: if they ever disagree, lit runs a file whose tests then skip, and
# the skip is invisible again.
_dbo_opt = os.environ.get("TRITON_SPYRE_DBO_OPT", "dbo-opt")
_dbo_path = _dbo_opt if os.sep in _dbo_opt else shutil.which(_dbo_opt)
if _dbo_path and os.path.isfile(_dbo_path):
    config.available_features.add("dbo-opt")

# A ``spyre-device`` feature for the tests that actually launch. The launch is
# in-process, so it names no interpreter: what has to be true is a property of the
# machine and of the path this very interpreter resolves.
#
# Two halves, both cheap enough to pay on every lit run:
#
#   * a numbered /dev/vfio group. /dev/vfio/vfio is the container and exists
#     whether or not a card is bound, so the digit-named entries are the ones that
#     mean hardware.
#   * torch_spyre resolvable, checked with find_spec rather than an import. lit
#     inherits the PYTHONPATH it forwards above, so this resolves on the same path
#     the test process will search, and it costs one finder walk -- where a real
#     import pulls in torch and dlopens _C.so, seconds spent on every run
#     including the ones that launch nothing.
#
# What the two buy: the feature is false in both the states that matter, and each
# is reachable. CI has neither half. On a machine with hardware, dropping
# torch-spyre from PYTHONPATH turns it off -- which is what makes this a gate that
# can be exercised rather than one that is merely believed.
#
# What they do not buy, deliberately: neither proves the device is healthy or free.
# A wedged or already-open device should fail loudly with the runtime's own
# diagnostic, not vanish into Unsupported, which would read as "no device here" on
# a machine that has one.
#
# SERIALIZATION. A Spyre device admits one opener, and with an in-process launch
# the opener is the pytest process itself, for its whole lifetime -- not a child it
# spawns and reaps. lit runs test files in parallel, one worker per file, so what
# keeps this safe is that exactly ONE lit file requires this feature. A second
# device-requiring *file* is what breaks it (a second test inside that file is
# fine), and the fix at that point is an fcntl.flock on a well-known path taken
# around the whole test session, not around a call.
_vfio = "/dev/vfio"
_have_vfio_group = os.path.isdir(_vfio) and any(
    name.isdigit() for name in os.listdir(_vfio))
if _have_vfio_group and importlib.util.find_spec("torch_spyre") is not None:
    config.available_features.add("spyre-device")

llvm_config.use_default_substitutions()

config.spyre_tools_dir = os.path.join(config.spyre_obj_root, "bin",
                                      "spyre-triton-opt")
tool_dirs = [config.spyre_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["spyre-triton-opt", "FileCheck"], tool_dirs)

config.substitutions = [
    (key, '"%s"' % val if val and " " in val and not val.startswith('"') else val)
    for key, val in config.substitutions
]
