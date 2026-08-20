# RUN: %python %s | FileCheck %s

"""The fixed segment policy, exercised through the backend's own helper.

A Python lit test rather than a .mlir one because the policy is arithmetic over a
signature, not a rewrite: there is no pass to run and no IR to check. Printing the
tuples and matching them keeps the expected values next to the inputs, which a
table of asserts does not.
"""
from backend.compiler import _segment_addresses

# Segment i is based at i * 16 GiB, expressed in ELEMENTS, so for f16 the byte
# offset divides by 2. These are the same three constants torch-spyre's Inductor
# path assigns for a 3-buffer fp16 kernel (SEGMENT_OFFSETS).
# CHECK: fp16 x3: (0, 8589934592, 17179869184)
print("fp16 x3:", _segment_addresses(["*f16", "*f16", "*f16"]))

# The element count per segment depends on the width: same 16 GiB, fewer f32s.
# CHECK-NEXT: f32 x2: (0, 4294967296)
print("f32 x2:", _segment_addresses(["*f32", "*f32"]))
# CHECK-NEXT: i8 x2: (0, 17179869184)
print("i8 x2:", _segment_addresses(["*i8", "*i8"]))

# Each pointer uses ITS OWN width. A single global element stride would misplace
# the narrower type in a mixed-precision kernel.
# CHECK-NEXT: mixed: (0, 8589934592, 34359738368)
print("mixed:", _segment_addresses(["*f32", "*f16", "*i8"]))

# Only ``*``-prefixed entries are pointers; a runtime scalar consumes no segment,
# and positions stay dense so segment i and pointer i agree by construction.
# CHECK-NEXT: with scalars: (0, 8589934592)
print("with scalars:", _segment_addresses(["*f16", "i32", "*f16", "index"]))

# Segment 7 holds the program, so seven pointers is the ceiling.
# CHECK-NEXT: seven: 7
print("seven:", len(_segment_addresses(["*f16"] * 7)))
try:
    _segment_addresses(["*f16"] * 8)
except ValueError as e:
    # CHECK-NEXT: eight rejected: at most 7 pointer arguments
    print("eight rejected:", str(e).split(";")[0].split("Spyre supports ")[1])

# A width that is not a whole number of bytes has no honest element index.
for bad in ("*float128", "*!tt.ptr<f16>", "*i1"):
    try:
        _segment_addresses([bad])
    except ValueError:
        # CHECK: rejected: *float128
        # CHECK-NEXT: rejected: *!tt.ptr<f16>
        # CHECK-NEXT: rejected: *i1
        print("rejected:", bad)
