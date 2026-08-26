#!/usr/bin/env python3
"""Launch a compiled spyreCodeDir on the Spyre device. Run me in a torch-spyre venv.

This is the device half of a device test, and it is a separate *process* on
purpose. Two reasons, both structural rather than stylistic:

* ``torch_spyre/_C.so`` is dlopen'd at import and ``ld.so`` reads
  ``LD_LIBRARY_PATH`` at **exec**, so the library path must already be in place
  when the interpreter starts. No in-process arrangement can arrange that.
* A Spyre device opens exclusively per process, so the pytest parent must not
  hold one. It does not: it only spawns this.

Consequently this file must not import ``triton``, and the interpreter that runs
it is not the one that runs the test suite. The handoff is a directory plus
``.npy`` files.

It launches and writes the output; it does not judge. Comparison belongs in the
test, where a mismatch gets a real assertion message and the fixture's own
tolerances.

    device_runner.py --dir DIR --pointers x_ptr,y_ptr,output_ptr \\
                     --output output_ptr

``--dir`` holds ``spyrecode/`` (flat: ``spyrecode.json`` beside
``init_binary.bin``) and one ``<name>.npy`` per pointer. The output lands in
``<output>.out.npy``. ``--pointers`` is in KERNEL ORDER, which is load-bearing:
the address-correction flit is built by walking the launch arguments positionally,
so a reordering silently patches the wrong segments.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=pathlib.Path)
    ap.add_argument("--pointers", required=True,
                    help="comma-separated, in kernel argument order")
    ap.add_argument("--output", required=True,
                    help="which pointer is the output")
    ap.add_argument("--stop-after", choices=["layout", "prepare", "launch"])
    args = ap.parse_args()
    pointers = args.pointers.split(",")
    spyrecode = args.dir / "spyrecode"

    import torch
    import torch_spyre

    # Assert, do not set. torch_spyre.__init__ does setdefault(..., "1"), so the
    # correct value normally arrives as an import side effect of another package --
    # but an inherited BUNDLE_SYMBOLIC_ARGS=0 would win silently and change the
    # address mode. prepare_kernel.cpp reads it as
    #     bind_io_addresses_ = (env == nullptr || env != "1")
    # so only the exact string "1" means "addresses arrive via the correction
    # flit". Unset is not neutral -- it binds addresses, which is wrong for the
    # symbolic artifact this backend emits by default.
    symbolic = os.environ.get("BUNDLE_SYMBOLIC_ARGS")
    if symbolic != "1":
        print(f"BUNDLE_SYMBOLIC_ARGS={symbolic!r}, need '1' to launch a symbolic "
              "artifact", file=sys.stderr)
        return 2

    torch_spyre._autoload()

    # prepare_kernel SIGSEGVs without an initialized runtime, and a device
    # allocation is what initializes it.
    torch.zeros(1, device="spyre")

    host = {name: np.load(args.dir / f"{name}.npy") for name in pointers}

    # Layout of the first input, reported for diagnosis rather than asserted: the
    # test cannot see it, and when a comparison fails on a wider shape this is the
    # first thing worth knowing. fp16 sticks are 64 elements, so device_size is
    # stick-major and need not match the host shape.
    first = torch.from_numpy(host[pointers[0]].copy()).to("spyre")
    layout = getattr(torch_spyre._C, "get_spyre_tensor_layout", None)
    info = {}
    if layout is not None:
        got = layout(first)
        info = {a: str(getattr(got, a)) for a in ("device_size", "stride_map")
                if hasattr(got, a)}
    print(json.dumps({"stage": "layout", "host_shape": list(first.shape), **info}))
    if args.stop_after == "layout":
        return 0

    # The directory is flat. torch-spyre's own runner appends "/spyreCodeDir"
    # because dbo-opt's export nests it; SpyreUtils.load_binary flattens, so this
    # is passed as-is.
    plan = torch_spyre._C.prepare_kernel(str(spyrecode))
    steps = [str(plan.get_step_type(i)) for i in range(plan.num_steps())]
    print(json.dumps({"stage": "prepare", "steps": steps}))
    if args.stop_after == "prepare":
        return 0

    # Kernel order. The output is zeroed so that a kernel which never writes
    # cannot be mistaken for one that wrote the right answer.
    tensors = []
    for name in pointers:
        t = torch.from_numpy(host[name].copy())
        if name == args.output:
            t.zero_()
        tensors.append(t.to("spyre"))
    torch_spyre._C.launch_jobplan(plan, tensors)
    if hasattr(torch_spyre._C, "synchronize"):
        torch_spyre._C.synchronize()
    state = (str(torch_spyre._C.get_device_state())
             if hasattr(torch_spyre._C, "get_device_state") else "unknown")
    print(json.dumps({"stage": "launch", "device_state": state}))
    if args.stop_after == "launch":
        return 0

    # There is no D2H step in a symbolic artifact -- the output tensor's own
    # device memory is the only channel back, so this is the read.
    out = tensors[pointers.index(args.output)].cpu().numpy()
    np.save(args.dir / f"{args.output}.out.npy", out)
    print(json.dumps({"stage": "done", "output": f"{args.output}.out.npy",
                      "nonzero": int(np.count_nonzero(out)), "size": int(out.size)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
