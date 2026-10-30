// A complete 128-thread group owns one query and four document scores.
// Normalize the query once into shared memory; four SIMD32 groups reuse it.
// The caller guarantees finite, pre-normalized float32 document vectors.
const uint query = thread_position_in_grid.y;
const uint tile = threadgroup_position_in_grid.x;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint document = tile * 4 + sg;
const long query_base = long(query) * long(queries_strides[0]);
threadgroup float normalized_query[D];
threadgroup float partial[4];
threadgroup uint bad[4];
float values[(D + 127) / 128];
float peak = 1.0e-12f;
uint invalid = 0;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    const uint column = tid + k * 128;
    const float value = column < D
        ? queries[query_base + long(column) * long(queries_strides[1])]
        : 0.0f;
    values[k] = value;
    peak = max(peak, abs(value));
    invalid |= uint(!isfinite(value));
}
const float simd_peak = simd_max(peak);
const uint simd_invalid = simd_sum(invalid);
if (lane == 0) {
    partial[sg] = simd_peak;
    bad[sg] = simd_invalid;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
peak = max(max(partial[0], partial[1]), max(partial[2], partial[3]));
invalid = bad[0] | bad[1] | bad[2] | bad[3];
// Every thread must finish reading the peaks before any SIMD overwrites them.
threadgroup_barrier(mem_flags::mem_threadgroup);
float squares = 0.0f;
for (uint k = 0; k < (D + 127) / 128; ++k) {
    values[k] = values[k] / peak;
    squares += values[k] * values[k];
}
const float simd_squares = simd_sum(squares);
if (lane == 0) partial[sg] = simd_squares;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = partial[0] + partial[1] + partial[2] + partial[3];
const float denominator = max(sqrt(total), 1.0e-12f / peak);
for (uint k = 0; k < (D + 127) / 128; ++k) {
    const uint column = tid + k * 128;
    if (column < D) {
        normalized_query[column] = invalid ? NAN : values[k] / denominator;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// No document-tail branch may skip an earlier threadgroup barrier.
if (document < uint(docs_shape[0])) {
    const long document_base = long(document) * long(docs_strides[0]);
    float dot = 0.0f;
    for (uint column = lane; column < D; column += 32) {
        const float document_value = docs[
            document_base + long(column) * long(docs_strides[1])];
        dot += normalized_query[column] * document_value;
    }
    const float score = simd_sum(dot);
    if (lane == 0) {
        scores[query * uint(docs_shape[0]) + document] = invalid ? NAN : score;
    }
}
