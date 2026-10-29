// Four SIMD groups cooperate on one wide row, reusing loaded register values.
const uint row = threadgroup_position_in_grid.x;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const long base = long(row) * long(inp_strides[0]);
threadgroup float partial[4];
threadgroup uint bad[4];
float values[(D + 127) / 128];
float peak = 1.0e-12f;
uint invalid = 0;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    uint col = tid + k * 128;
    float v = col < D ? inp[base + long(col) * long(inp_strides[1])] : 0.0f;
    values[k] = v;
    peak = max(peak, abs(v));
    invalid |= uint(!isfinite(v));
}
float p = simd_max(peak);
uint b = simd_sum(invalid);
if (lane == 0) { partial[sg] = p; bad[sg] = b; }
threadgroup_barrier(mem_flags::mem_threadgroup);
peak = max(max(partial[0], partial[1]), max(partial[2], partial[3]));
invalid = bad[0] | bad[1] | bad[2] | bad[3];
// All reads of partial must complete before the same storage is reused.
threadgroup_barrier(mem_flags::mem_threadgroup);
float squares = 0.0f;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    values[k] = values[k] / peak;
    squares += values[k] * values[k];
}
float s = simd_sum(squares);
if (lane == 0) partial[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float total = partial[0] + partial[1] + partial[2] + partial[3];
float denominator = max(sqrt(total), 1.0e-12f / peak);
for (uint k = 0; k < (D + 127) / 128; ++k) {
    uint col = tid + k * 128;
    if (col < D) out[row * D + col] = invalid ? NAN : values[k] / denominator;
}
