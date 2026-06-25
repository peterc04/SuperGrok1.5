#!/usr/bin/env python3
"""Saturation sweep for the ViT flagship cell (best-achievable ceiling), AdamW only.
Loads the CACHED mega_vit_multiopt .so. Sweeps (B, ncta_cap)."""
import os, sys, importlib.util
REPO = "/workspace/SuperGrok1.5"; sys.path.insert(0, REPO); os.chdir(REPO)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch
SO = "/workspace/flagship_build/mega_vit_multiopt/mega_vit_multiopt.so"
spec = importlib.util.spec_from_file_location("mega_vit_multiopt", SO)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
total=int(mod.TOTAL); d=int(mod.D); layers=int(mod.LAYERS); seq=int(mod.SEQ)
vocab=int(mod.VOCAB); npatch=int(mod.NPATCH); patch=int(mod.PATCH)
dev="cuda"
def gemm_flops(d,B,layers,seq,vocab):
    T=B*seq; g=lambda M,N,K:2.0*M*N*K; dff=4*d; fwd=0.0
    for _ in range(layers): fwd+=g(T,3*d,d)+g(T,d,d)+g(T,dff,d)+g(T,d,dff)
    fwd+=g(B,vocab,d); return 3.0*fwd
g=torch.Generator(device=dev).manual_seed(42)
params=(torch.randn(total,generator=g,device=dev)*0.02).contiguous()
state=torch.zeros(3*total,dtype=torch.float32,device=dev)
lr,b1,b2,eps,wd=1e-3,0.9,0.98,1e-8,0.0; bc1,bc2=1.0-b1,1.0-b2
best=None
for cap in [8,16,32,64,132,0]:
    for B in [16,64,128,256,512]:
        patches=torch.randn(B,npatch,patch,generator=g,device=dev).contiguous()
        targets=torch.randint(0,vocab,(B,),generator=g,device=dev).to(torch.int32).contiguous()
        flops=gemm_flops(d,B,layers,seq,vocab)
        try:
            state.zero_()
            def call(s): return mod.tc_train_step_opt(0,params,patches,targets,state,lr,b1,b2,eps,wd,bc1,bc2,s,0.98,2.0,0.0,cap)
            for s in range(1,4): loss,_=call(s)
            torch.cuda.synchronize()
            NIT=8; e0=[torch.cuda.Event(enable_timing=True) for _ in range(NIT)]; e1=[torch.cuda.Event(enable_timing=True) for _ in range(NIT)]
            for i in range(NIT): e0[i].record(); call(4+i); e1[i].record()
            torch.cuda.synchronize()
            ms=sorted(e0[i].elapsed_time(e1[i]) for i in range(NIT)); step_ms=ms[len(ms)//2]
            tf=flops/(step_ms/1e3)/1e12; pct=100.0*tf/989.0; memgb=torch.cuda.max_memory_allocated()/1e9
            print(f"B={B:5d} cap={cap:4d}  step={step_ms:9.2f}ms  {tf:7.2f} TF/s ({pct:5.2f}%)  mem~{memgb:.1f}GB",flush=True)
            print(f"SAT,vit,adamw,B{B},cap{cap},{step_ms:.4f},{tf:.4f},{pct:.4f}",flush=True)
            if best is None or tf>best[0]: best=(tf,B,cap,step_ms,pct)
        except Exception as e:
            print(f"B={B:5d} cap={cap:4d}  FAILED: {repr(e)[:80]}",flush=True)
        finally:
            try: del patches,targets
            except: pass
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
if best: print(f"\nBEST: {best[0]:.2f} TF/s ({best[4]:.2f}%) at B={best[1]} cap={best[2]} step={best[3]:.2f}ms",flush=True)
