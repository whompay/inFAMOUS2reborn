#!/usr/bin/env python3
"""lift_raw_jobs.py -- lift the headerless SPURS job binaries in both of their address views.

The loop at the end of lift.cmd, as one installer step (installer/infamous2.toml, "raw-jobs-lift"):
each spu-images/raw-jobs/job_<id>.bin is lifted at LS 0 (the PPU job-chain shortcut loads it
there) and at LS 0x4000 (the WWS policy module streams it into its job-code buffer).
spu/raw_jobs_register.c (hand-written) gives each pair one image id.

    python lift_raw_jobs.py <ps3recomp dir> <infamous2-recomp dir>
"""
import glob
import os
import subprocess
import sys

VIEWS = (("direct", "0"), ("overlay", "0x4000"))


def run(args):
    r = subprocess.run([sys.executable] + args)
    if r.returncode:
        sys.exit(r.returncode)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tools = os.path.join(sys.argv[1], "tools")
    port = sys.argv[2]
    jobs_dir = os.path.join(port, "spu-images", "raw-jobs")
    jobs = sorted(glob.glob(os.path.join(jobs_dir, "job_*.bin")))
    if not jobs:
        print(f"no job_*.bin in {jobs_dir}", file=sys.stderr)
        return 1
    for path in jobs:
        jid = os.path.basename(path)[len("job_"):-len(".bin")]
        for view, base in VIEWS:
            funcs = os.path.join(jobs_dir, f"job_{jid}.{view}.functions.json")
            run([os.path.join(tools, "find_spu_functions.py"), path, "--raw", "--base", base, "--out", funcs])
            run([os.path.join(tools, "spu_lifter.py"), path, "--base", base, "--functions", funcs,
                 "--symbol-prefix", f"job_{view}_{jid}_",
                 "--output", os.path.join(port, "spu", f"job_{view}_{jid}")])
    print(f"lifted {len(jobs)} raw SPURS jobs (direct + overlay views)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
