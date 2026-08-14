from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from types import ModuleType


@dataclass(frozen=True)
class GPUTarget(object):
    # Target backend, e.g., cuda, hip
    backend: str
    # Target architecture, e.g., 90 (for cuda compute capability), gfx940 (for hip)
    arch: Union[int, str]
    warp_size: int


class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1


class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget):
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        raise NotImplementedError

    @abstractmethod
    def parse_options(self, options: dict) -> object:
        """
        Converts an `options` dictionary into an arbitrary object and returns it.
        This function may contain target-specific heuristics and check the legality of the provided options
        """
        raise NotImplementedError

    @abstractmethod
    def add_stages(self, stages: dict, options: object) -> None:
        """
        Populates `stages` dictionary with entries of the form:
        ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
        The value of each entry may populate a `metadata` dictionary.
        Stages will be run sequentially (in inseriton order) and can communicate using `metadata`.
        All stages are expected to return a `str` object, except for the last stage which returns
        a `bytes` object for execution by the launcher.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dialects(self, context):
        """
        Load additional MLIR dialects into the provided `context`
        """
        raise NotImplementedError

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations
        """
        raise NotImplementedError

    # --- START --- added for spyre
    def compile_time_launch_options(self, grid, specialization) -> Dict:
        """Extra `parse_options` inputs that are only knowable at launch time.

        Most backends have none, and return `{}`: the launch grid is a runtime
        parameter that never reaches the compiler. A backend that instead bakes
        it into the compiled artifact needs it among the options *before* the
        cache key is computed, and this is the only place that can supply it —
        the grid is consumed by `JITFunction.run`'s own keyword-only parameter,
        so it never reaches the `**options` the argument binder collects, and
        `parse_options` runs after the key is computed.

        `grid` is whatever `kernel[grid]` carried (`None` for a warmup);
        `specialization` is the per-argument `(type, specialization)` list
        `create_function_from_signature` built. `specialization` is offered for
        a backend that wants to key an option off the argument types, but note
        that it is already part of the cache key on its own — so anything purely
        derived from it can equally be derived later, inside the backend, with
        no risk of a wrong key.

        Called from `JITFunction.run`; the returned keys must be fields of the
        object `parse_options` produces, or `_pack_args` will reject them.
        """
        return {}

    # --- END --- added for spyre
    @staticmethod
    def parse_attr(desc):
        assert isinstance(desc, str)
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 16]]
        return ret

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        if arg % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        if arg.data_ptr() % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""
