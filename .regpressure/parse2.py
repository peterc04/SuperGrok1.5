#!/usr/bin/env python3
"""ptxas -v parser v2: attributes callee (ABI device fn) spills to their entry.
ptxas prints, per entry section: the callee Function properties blocks, then
'Compiling entry function X', then X's properties + register count. Callee
blocks BEFORE an entry belong to that entry's compilation section."""
import re, sys, subprocess

OPT = {0:"AdamW",1:"Lion",2:"Grokfast",3:"GrokAdamW",4:"LookSAM",5:"Prodigy",
       6:"NeuralGrok",7:"Muon",8:"SuperGrok11",9:"SuperGrok15",10:"SuperGrok2"}

def parse(path):
    lines = open(path).read().splitlines()
    entries = []   # (optname, regs, smem, entry_st, entry_ld, callee_st, callee_ld, ncallee_spilling, entry_stack)
    pend_fns = []  # (name, stack, st, ld) callees seen since last entry
    cur = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.search(r"Function properties for (\S+)", ln)
        if m:
            name = m.group(1)
            fp = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", lines[i+1] if i+1 < len(lines) else "")
            st = ld = stk = 0
            if fp: stk, st, ld = map(int, fp.groups())
            if cur is not None and name == cur["mangled"]:
                cur["stack"], cur["st"], cur["ld"] = stk, st, ld
            elif cur is not None:
                cur["callees"].append((name, stk, st, ld))
            else:
                pend_fns.append((name, stk, st, ld))
            i += 1
        m = re.search(r"Compiling entry function '([^']+)' for 'sm_90a'", ln)
        if m:
            if cur is not None:
                entries.append(cur)
            cur = dict(mangled=m.group(1), regs=-1, smem=0, st=0, ld=0, stack=0,
                       callees=[])
            pend_fns = []
        m = re.search(r"Used (\d+) registers", ln)
        if m and cur is not None and cur["regs"] < 0:
            cur["regs"] = int(m.group(1))
            sm = re.search(r"(\d+) bytes smem", ln)
            if sm: cur["smem"] = int(sm.group(1))
        i += 1
    if cur is not None:
        entries.append(cur)
    return entries

def optname(mangled):
    m = re.search(r"OptIdE?(\d+)", mangled)
    if m: return OPT.get(int(m.group(1)), "?")
    m = re.search(r"\)(\d+)EEE", mangled)
    return "?"

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\n### {path}")
        print(f"{'opt':12s} {'regs':>4s} {'entry_sp_st':>11s} {'callee_sp_st':>12s} {'TOTAL_st':>9s} {'TOTAL_ld':>9s} {'#spill_fns':>10s}")
        for e in parse(path):
            if "EmptyKernel" in e["mangled"]: continue
            cst = sum(f[2] for f in e["callees"]); cld = sum(f[3] for f in e["callees"])
            nsf = sum(1 for f in e["callees"] if f[2] or f[3])
            print(f"{optname(e['mangled']):12s} {e['regs']:4d} {e['st']:11d} {cst:12d} {e['st']+cst:9d} {e['ld']+cld:9d} {nsf:10d}")
