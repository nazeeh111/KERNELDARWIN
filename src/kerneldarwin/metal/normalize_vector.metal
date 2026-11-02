// Packed reads for contiguous four-element blocks, with a scalar strided path.
// Each SIMD group owns one row; no shared memory or inter-group synchronization.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
const long base = long(row) * long(inp_strides[0]);
float4 values[(D + 127) / 128];
float peak = 1.0e-12f;
uint invalid = 0;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    uint col = lane * 4 + k * 128;
    float4 v = float4(0.0f);
    // Packed-vector loads accept any float-aligned backing offset.
    if (inp_strides[1] == 1 && col + 3 < D) {
        v = kd_load4(inp + base + col);
    } else {
        for (uint j = 0; j < 4; ++j) {
            if (col + j < D) v[j] = inp[base + long(col + j) * long(inp_strides[1])];
        }
    }
    values[k] = v;
    peak = max(peak, max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w))));
    invalid |= uint(any(!isfinite(v)));
}
peak = simd_max(peak);
invalid = simd_sum(invalid);
float squares = 0.0f;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    values[k] = values[k] / peak;
    squares += dot(values[k], values[k]);
}
float denominator = max(sqrt(simd_sum(squares)), 1.0e-12f / peak);
for (uint k = 0; k < (D + 127) / 128; ++k) {
    uint col = lane * 4 + k * 128;
    for (uint j = 0; j < 4; ++j) {
        if (col + j < D) out[row * D + col + j] = invalid ? NAN : values[k][j] / denominator;
    }
}
