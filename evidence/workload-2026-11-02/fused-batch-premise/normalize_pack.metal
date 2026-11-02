// One SIMD group owns each document or query row. Three query groups are
// normalized directly into one output, without an intermediate concatenation.
const uint row = thread_position_in_grid.x / 32u;
const uint lane = thread_index_in_simdgroup;
const uint qr = row >= N ? row - N : 0u;
float values[(D + 31) / 32];
float peak = 1.0e-12f;
uint invalid = 0u;
for (uint slot = 0; slot < (D + 31) / 32; ++slot) {
    const uint col = lane + slot * 32u;
    float value = 0.0f;
    if (col < D) {
        if (row < N) {
            value = docs[long(row) * long(docs_strides[0]) + long(col) * long(docs_strides[1])];
        } else if (qr < Q0) {
            value = q0[long(qr) * long(q0_strides[0]) + long(col) * long(q0_strides[1])];
        } else if (qr < Q0 + Q1) {
            value = q1[long(qr - Q0) * long(q1_strides[0]) + long(col) * long(q1_strides[1])];
        } else {
            value = q2[long(qr - Q0 - Q1) * long(q2_strides[0]) + long(col) * long(q2_strides[1])];
        }
    }
    values[slot] = value;
    peak = max(peak, abs(value));
    invalid |= uint(!isfinite(value));
}
peak = simd_max(peak);
invalid = simd_max(invalid);
float squares = 0.0f;
for (uint slot = 0; slot < (D + 31) / 32; ++slot) {
    values[slot] /= peak;
    squares += values[slot] * values[slot];
}
const float denominator = max(sqrt(simd_sum(squares)), 1.0e-12f / peak);
if (lane == 0u) invalid_rows[row] = invalid;
for (uint slot = 0; slot < (D + 31) / 32; ++slot) {
    const uint col = lane + slot * 32u;
    if (col < D) {
        const float value = invalid ? NAN : values[slot] / denominator;
        if (row < N) normalized_docs[row * D + col] = value;
        else normalized_queries[qr * D + col] = value;
    }
}
