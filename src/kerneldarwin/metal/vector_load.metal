// MLX places very small inputs in the constant address space. packed_float4
// has float alignment, including a column slice whose backing offset is 4 bytes.
inline float4 kd_load4(const device float* p) {
    return float4(reinterpret_cast<const device packed_float4*>(p)[0]);
}
inline float4 kd_load4(const constant float* p) {
    return float4(reinterpret_cast<const constant packed_float4*>(p)[0]);
}
