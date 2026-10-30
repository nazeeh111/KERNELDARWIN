// One SIMD group owns one row. Values stay in registers through both reductions.
// Input strides come from MLX, so slices/broadcasts do not require a hidden copy.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
const long base = long(row) * long(inp_strides[0]);
float values[(D + 31) / 32];
float peak = 1.0e-12f;
uint invalid = 0;
for (uint k = 0; k < (D + 31) / 32; ++k) {
    uint col = lane + k * 32;
    float v = col < D ? inp[base + long(col) * long(inp_strides[1])] : 0.0f;
    values[k] = v;
    peak = max(peak, abs(v));
    invalid |= uint(!isfinite(v));
}
peak = simd_max(peak);
invalid = simd_sum(invalid);
float squares = 0.0f;
for (uint k = 0; k < (D + 31) / 32; ++k) {
    values[k] = values[k] / peak;
    squares += values[k] * values[k];
}
float denominator = max(sqrt(simd_sum(squares)), 1.0e-12f / peak);
for (uint k = 0; k < (D + 31) / 32; ++k) {
    uint col = lane + k * 32;
    if (col < D) out[row * D + col] = invalid ? NAN : values[k] / denominator;
}
