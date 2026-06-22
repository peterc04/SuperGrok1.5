#include <cstdio>
#include <cstdint>
#include <algorithm>
// DecDwSpec at DW_STAGE=1 (default) is 72 bytes (verified vs device layout below).
static const int kDwSpecBytes = 72;
// DecTcSmem layout (bytes):
//  sA: kDecRingStages * kDecAtomsPerSlot * 64 * 16 * sizeof(bf16=2)
//  sB: kDecRingStages * TILE_N * 16 * 2
//  red: 256*4 = 1024
//  spec: 9 * 72 = 648
// alignof 16 (the alignas(16) on sA). Trailing pad to multiple of 16.
long smem(int TILE_M, int TILE_N, int IL, int S){
    int kAtomsM = TILE_M/64;
    int kDecAtomsPerSlot = std::min(kAtomsM, IL);
    long sA = (long)S*kDecAtomsPerSlot*64*16*2;
    long sB = (long)S*TILE_N*16*2;
    long red = 1024;
    long spec = 9*72;
    long raw = sA+sB+red+spec;
    long aligned = (raw + 15)/16*16;
    return aligned;
}
int main(){
    int TMs[]={128,64,256}, TNs[]={128,64,256}, ILs[]={2,1,3,4};
    printf("%-6s %-6s %-4s | %-9s %-9s %-9s | crossover(>48KB)\n","TILE_M","TILE_N","IL","S=2(KB)","S=3(KB)","S=4(KB)");
    const long CAP=48*1024;
    for(int tm: TMs) for(int tn: TNs) for(int il: ILs){
        long s2=smem(tm,tn,il,2), s3=smem(tm,tn,il,3), s4=smem(tm,tn,il,4);
        const char* note="";
        if(s2>CAP) note="S2 BUSTS STATIC!";
        else if(s3>CAP) note="S3 needs dyn (S2 static-ok)";
        else if(s4>CAP) note="S4 needs dyn (S2,S3 static-ok)";
        else note="all S<=48KB (no dyn needed)";
        printf("%-6d %-6d %-4d | %8.2f %8.2f %8.2f | %s\n", tm,tn,il, s2/1024.0, s3/1024.0, s4/1024.0, note);
    }
    return 0;
}
