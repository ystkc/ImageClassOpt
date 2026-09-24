"""Experiment launcher. Edit SCRIPT and CONFIG to select the program to run."""
import os
import subprocess
import sys
from datetime import datetime
from omegaconf import OmegaConf

SCRIPT = "train_baseline.py"
CONFIG = "train.yaml" 

proc = subprocess.Popen([sys.executable, SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# load
cfg = OmegaConf.load(CONFIG)
ROOT = os.path.dirname(__file__)
EXP_NAME = cfg.exp_name
EXP_ROOT = os.path.join(ROOT, "exp", EXP_NAME)
os.makedirs(EXP_ROOT, exist_ok=True)

# save
OmegaConf.save(cfg, os.path.join(EXP_ROOT, "config.yaml"), resolve=True)

# git
commit = subprocess.check_output(["git", "log", "-1", "--pretty=%s"]).strip().decode("utf-8")
os.system("git status")
print(f"Previous commit: {commit!r} {'amend commit' if commit == EXP_NAME else 'new commit'}\n")
input("continue?")
subprocess.check_call(["git", "add", "."])
if commit == EXP_NAME:
    subprocess.check_call(["git", "commit", "--amend", "--no-edit"])
else:
    subprocess.check_call(["git", "commit", "-m", EXP_NAME])

# start
os.chdir(EXP_ROOT)
with open("stdout" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log", "wb") as f:
    for line in proc.stdout:
        sys.stdout.buffer.write(line)
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S ").encode() + line)
os.system("sync")