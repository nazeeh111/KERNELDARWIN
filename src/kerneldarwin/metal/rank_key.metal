// Match pinned argsort(-scores): native tiny comparison keys tie as zero.
// Values are never converted: the selected original bits are copied to output.
inline uint kd_top5_key(uint bits) {
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude < 0x00800000u) return 0x80000000u;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
