#!/usr/bin/env python3
"""Parse ptxas -v output (nvcc --resource-usage logs) into a per-kernel table."""
import re, sys, subprocess

def demangle(name):
    try:
        out = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip()
        return out or name
    except Exception:
        return name

def shorten(dem):
    # sg::fused::sm90::fused_decoder_megakernel_tc<(sg::fused::sm90::OptId)0>(...) ->
    # fused_decoder_megakernel_tc<Opt0>
    m = re.search(r"(?:sg::fused::sm90::)?(\w+)<\(sg::fused::sm90::OptId\)(\d+)>", dem)
    if m:
        return f"{m.group(1)}<OptId={m.group(2)}>"
    m = re.search(r"(?:sg::fused::sm90::)?(\w+)(?:<|\()", dem)
    return m.group(1) if m else dem[:80]

OPT = {0:"AdamW",1:"Lion",2:"Grokfast",3:"GrokAdamW",4:"LookSAM",5:"Prodigy",
       6:"NeuralGrok",7:"Muon",8:"SuperGrok11",9:"SuperGrok15",10:"SuperGrok2"}

def parse(path):
    txt = open(path).read()
    # blocks: "ptxas info    : Compiling entry function '<mangled>' for 'sm_90a'"
    # then Function properties line(s): N bytes stack frame, N bytes spill stores, N bytes spill loads
    # then Used N registers, [used M barriers,] N bytes smem...
    rows = []
    blocks = re.split(r"ptxas info\s+: Compiling entry function '", txt)
    for b in blocks[1:]:
        m = re.match(r"([^']+)' for 'sm_90a'", b)
        if not m: continue
        mangled = m.group(1)
        stack = spst = spld = 0
        fp = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
        if fp:
            stack, spst, spld = map(int, fp.groups())
        ur = re.search(r"Used (\d+) registers(?:, used (\d+) barriers)?", b)
        regs = int(ur.group(1)) if ur else -1
        sm = re.search(r"(\d+) bytes smem", b)
        smem = int(sm.group(1)) if sm else 0
        cm = re.search(r"(\d+) bytes cmem\[0\]", b)
        cmem = int(cm.group(1)) if cm else 0
        rows.append(dict(mangled=mangled, stack=stack, spill_st=spst, spill_ld=spld,
                         regs=regs, smem=smem, cmem=cmem))
    return rows

def optname(short):
    m = re.search(r"OptId=(\d+)", short)
    if m: return OPT.get(int(m.group(1)), short)
    return ""

if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = parse(path)
        print(f"\n### {path}")
        print(f"{'kernel':58s} {'opt':12s} {'regs':>4s} {'stack':>6s} {'sp_st':>6s} {'sp_ld':>6s} {'smem':>7s}")
        for r in rows:
            dem = demangle(r["mangled"])
            s = shorten(dem)
            print(f"{s:58s} {optname(s):12s} {r['regs']:4d} {r['stack']:6d} {r['spill_st']:6d} {r['spill_ld']:6d} {r['smem']:7d}")
