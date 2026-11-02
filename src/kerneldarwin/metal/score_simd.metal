// One complete SIMD32 group owns one (query, document) output pair.
// Normalize a raw query in registers while accumulating its scaled dot product.
// The caller guarantees finite, pre-normalized float32 document vectors.
const uint document = thread_position_in_grid.x / 32;
const uint query = thread_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const long query_base = long(query) * long(queries_strides[0]);
const long document_base = long(document) * long(docs_strides[0]);
float values[(D + 31) / 32];
float peak = 1.0e-12f;
uint invalid = 0;
for (uint k = 0; k < (D + 31) / 32; ++k) {
    const uint column = lane + k * 32;
    const float value = column < D
        ? queries[query_base + long(column) * long(queries_strides[1])]
        : 0.0f;
    values[k] = value;
    peak = max(peak, abs(value));
    invalid |= uint(!isfinite(value));
}
peak = simd_max(peak);
invalid = simd_sum(invalid);
float squares = 0.0f;
float dot = 0.0f;
for (uint k = 0; k < (D + 31) / 32; ++k) {
    const uint column = lane + k * 32;
    const float scaled = values[k] / peak;
    const float document_value = column < D
        ? docs[document_base + long(column) * long(docs_strides[1])]
        : 0.0f;
    squares += scaled * scaled;
    dot += scaled * document_value;
}
const float denominator = max(sqrt(simd_sum(squares)), 1.0e-12f / peak);
const float score = simd_sum(dot) / denominator;
if (lane == 0) {
    scores[query * uint(docs_shape[0]) + document] = invalid ? NAN : score;
}
