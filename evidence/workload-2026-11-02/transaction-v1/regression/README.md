# Bounded host-input regression

The first source-free installed native validation ran28methods with API/GPU validation. One method errored: MLX0.32 rejected the negative strides of a reversed NumPy matrix before the intended library fallback. This was a genuine adapter defect, not a failed performance study. The report/log and exact previous planner/test/pin are preserved here. No candidate timing or reserved confirmation input was used.

The correction explicitly copies negative-stride, read-only and unaligned host inputs, preserving dtype, before MLX staging. These unprofiled requests use the complete library fallback. The corrected method also exercises read-only, unaligned and broadcast host storage. The successful28-method report is in the parent folder. The failed source archive is historical evidence, not an installable accepted artifact.
