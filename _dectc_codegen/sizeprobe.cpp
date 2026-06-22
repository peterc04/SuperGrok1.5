#include <cstdio>
#include <cstdint>
#include <cstddef>
// Mirror __nv_bfloat16 as a 2-byte POD (same size/alignment as CUDA's).
struct bf16 { unsigned short x; };
#define SG_TUNED_DEC_DW_STAGE 1   // default
struct alignas(8) DecDwSpec {
    const bf16* dY;
    const bf16* X;
    int Nout; int Kin; int K;
    int grad_off;
    const bf16* dY_bias;
    int bias_off;
#if SG_TUNED_DEC_DW_STAGE
    bf16* dYt;
    bf16* Xt;
    int64_t t_off;
#endif
};
// no alignas on DecDwSpec in real code; let compiler pick. Recreate WITHOUT alignas:
struct DecDwSpecReal {
    const bf16* dY;
    const bf16* X;
    int Nout; int Kin; int K;
    int grad_off;
    const bf16* dY_bias;
    int bias_off;
#if SG_TUNED_DEC_DW_STAGE
    bf16* dYt;
    bf16* Xt;
    int64_t t_off;
#endif
};
template<int kDecRingStages>
struct DecTcSmem {
    static constexpr int kDecAtomsPerSlot = 2; // min(2,2)
    static constexpr int TILE_N = 128;
    alignas(16) bf16 sA[kDecRingStages * kDecAtomsPerSlot * 64 * 16];
    alignas(16) bf16 sB[kDecRingStages * TILE_N * 16];
    float red[256];
    DecDwSpecReal spec[9];
};
int main(){
    printf("sizeof(DecDwSpecReal)=%zu alignof=%zu\n", sizeof(DecDwSpecReal), alignof(DecDwSpecReal));
    printf("S=2: sizeof(DecTcSmem)=%zu (%.3f KB) alignof=%zu\n", sizeof(DecTcSmem<2>), sizeof(DecTcSmem<2>)/1024.0, alignof(DecTcSmem<2>));
    printf("S=3: sizeof(DecTcSmem)=%zu (%.3f KB) alignof=%zu\n", sizeof(DecTcSmem<3>), sizeof(DecTcSmem<3>)/1024.0, alignof(DecTcSmem<3>));
    printf("S=4: sizeof(DecTcSmem)=%zu (%.3f KB) alignof=%zu\n", sizeof(DecTcSmem<4>), sizeof(DecTcSmem<4>)/1024.0, alignof(DecTcSmem<4>));
    printf("48KB=%d  STATIC CAP\n", 48*1024);
    return 0;
}
