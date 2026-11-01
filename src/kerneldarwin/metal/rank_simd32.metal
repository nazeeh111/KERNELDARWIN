// One SIMD group selects one row's exact stable top five. No shared storage.
// Grid Q*32, SIMD width32; K=5, 5<=N<=2048, Q<=128. Padding/removed keys are0.
const uint row = thread_position_in_grid.x / 32u;
const uint lane = thread_index_in_simdgroup;
const long base = long(row) * long(scores_strides[0]);
const long stride = long(scores_strides[1]);
uint keys[(N + 31) / 32];
uint invalid = 0u;
for (uint slot = 0; slot < (N + 31) / 32; ++slot) {
    const uint col = lane + slot * 32u;
    uint key = 0u;
    if (col < N) {
        const uint bits = as_type<uint>(scores[base + long(col) * stride]);
        invalid |= uint((bits & 0x7f800000u) == 0x7f800000u);
        key = kd_top5_key(bits);
    }
    keys[slot] = key;
}
invalid = simd_max(invalid);
if (invalid != 0u) {
    // Uniform per-SIMD return; other rows share no storage or barriers.
    if (lane == 0u) {
        for (uint j = 0; j < K; ++j) {
            ids[row * K + j] = 0xffffffffu;
            values[row * K + j] = as_type<float>(0x7fc00000u);
        }
    }
    return;
}
for (uint j = 0; j < K; ++j) {
    uint local_key = 0u;
    uint local_id = 0xffffffffu;
    for (uint slot = 0; slot < (N + 31) / 32; ++slot) {
        const uint key = keys[slot];
        const uint col = lane + slot * 32u;
        if (key != 0u && (key > local_key || (key == local_key && col < local_id))) {
            local_key = key;
            local_id = col;
        }
    }
    const uint winner_key = simd_max(local_key);
    const uint winner_id = simd_min(local_key == winner_key ? local_id : 0xffffffffu);
    // N>=K and positive finite keys guarantee a valid, unique winner.
    if (lane == 0u) {
        ids[row * K + j] = winner_id;
        values[row * K + j] = as_type<float>(as_type<uint>(scores[base + long(winner_id) * stride]));
    }
    if (lane == winner_id % 32u) keys[winner_id / 32u] = 0u;
}
