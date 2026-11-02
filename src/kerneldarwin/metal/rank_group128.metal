// Untimed prototype. Requires float32 scores[Q, N], Q <= 128, 5 <= N <= 2048,
// template K == 5, grid (Q * 128, 1, 1), and threadgroup (128, 1, 1).
// Apple SIMD width must be 32. Header supplies kd_top5_key(uint).
// scores_strides are actual signed element strides, including zero/negative.
const uint row = threadgroup_position_in_grid.x;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const long base = long(row) * long(scores_strides[0]);
const long col_stride = long(scores_strides[1]);
threadgroup uint bad[4];
threadgroup uint wave_keys[4];
threadgroup uint wave_ids[4];
uint keys[(N + 127) / 128];
uint invalid = 0u;

// Each real score is loaded once for its key. Key zero denotes padding/removal;
// every finite score has a positive key. Original bits are reloaded on output.
for (uint slot = 0; slot < (N + 127) / 128; ++slot) {
    const uint col = tid + slot * 128u;
    uint key = 0u;
    if (col < N) {
        const uint bits = as_type<uint>(scores[base + long(col) * col_stride]);
        invalid |= uint((bits & 0x7f800000u) == 0x7f800000u);
        key = kd_top5_key(bits);
    }
    keys[slot] = key;
}
const uint wave_bad = simd_max(invalid);
if (lane == 0u) bad[sg] = wave_bad;
// Publish all four invalid flags before any thread decides to return.
threadgroup_barrier(mem_flags::mem_threadgroup);
invalid = bad[0] | bad[1] | bad[2] | bad[3];
if (invalid != 0u) {
    // Uniform whole-threadgroup return. bad[] is never reused.
    if (tid == 0u) {
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
    for (uint slot = 0; slot < (N + 127) / 128; ++slot) {
        const uint key = keys[slot];
        const uint col = tid + slot * 128u;
        if (key != 0u &&
            (key > local_key || (key == local_key && col < local_id))) {
            local_key = key;
            local_id = col;
        }
    }
    const uint wave_key = simd_max(local_key);
    const uint wave_id = simd_min(local_key == wave_key
                                 ? local_id : 0xffffffffu);
    if (lane == 0u) {
        wave_keys[sg] = wave_key;
        wave_ids[sg] = wave_id;
    }
    // Publish each SIMD group's best (key, index) pair before all-group reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint winner_key = 0u;
    uint winner_id = 0xffffffffu;
    for (uint group = 0; group < 4u; ++group) {
        const uint key = wave_keys[group];
        const uint index = wave_ids[group];
        if (key > winner_key || (key == winner_key && index < winner_id)) {
            winner_key = key;
            winner_id = index;
        }
    }
    // N >= K leaves a real winner on every iteration. All threads agree on it.
    if (tid == 0u) {
        ids[row * K + j] = winner_id;
        values[row * K + j] =
            as_type<float>(as_type<uint>(scores[base + long(winner_id) * col_stride]));
    }
    if (tid == winner_id % 128u) keys[winner_id / 128u] = 0u;
    // Finish every shared-pair read before the next iteration overwrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
