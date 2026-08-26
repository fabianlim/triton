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
    # For the device tests: which interpreter can import torch_spyre, and the
    # library path its _C.so needs. LD_LIBRARY_PATH has to be forwarded rather
    # than reconstructed, because ld.so reads it when the CHILD is exec'd and only
    # what pytest inherits reaches that child.
    "TRITON_SPYRE_LAUNCH_PYTHON", "LD_LIBRARY_PATH", "DEEPTOOLS_PATH",
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

# A ``spyre-device`` feature for the tests that actually launch. Deliberately only
# a check that the launch interpreter exists and is executable -- NOT that a device
# is present and healthy. A machine with the interpreter but a wedged device should
# fail loudly with the runtime's own diagnostic, not vanish into Unsupported, which
# would read as "no device here" on a machine that has one.
#
# There is no default: a bare name would be some interpreter without torch_spyre,
# and the whole point is that this is a different venv from the one running the
# suite. Unset means no device testing.
_launch_python = os.environ.get("TRITON_SPYRE_LAUNCH_PYTHON")
if _launch_python and os.access(_launch_python, os.X_OK):
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
