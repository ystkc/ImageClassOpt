"""Experiment launcher. Edit SCRIPT and CONFIG to select the program to run."""
import os
import subprocess
import sys
from datetime import datetime
from omegaconf import OmegaConf

SCRIPT = "train.py"
CONFIG = "train.yaml"

# load
cfg = OmegaConf.load(CONFIG)
ROOT = os.path.dirname(__file__)
EXP_NAME = "exptest"
EXP_ROOT = os.path.join(ROOT, "exp", EXP_NAME)
os.makedirs(EXP_ROOT, exist_ok=True)

# git
commit = subprocess.check_output("git log -1 --pretty=%s").strip().decode("utf-8")
os.system("git status")
print(f"Previous commit: {commit!r} {'amend commit' if commit == EXP_NAME else 'new commit'}\n")
input("continue?")
subprocess.check_call("git add .")
if commit == EXP_NAME:
    subprocess.check_call("git commit --amend --no-edit")
else:
    subprocess.check_call("git commit -m " + EXP_NAME)

proc = subprocess.Popen([sys.executable, SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# chdir, save
os.chdir(EXP_ROOT)
OmegaConf.save(cfg, "config.yaml", resolve=True)
with open("stdout" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log", "wb") as f:
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + line)
os.system("sync")