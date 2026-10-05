# Getting Week 2 onto GitHub — createwithajay/VisionEdge

This assumes you already have the project folder locally (unzip this
package into your existing `vision_edge_member3` folder, overwriting/adding
files — it's built on top of your Week 1 work).

## 1. Unzip into your existing project folder
```bash
cd ~/vision_edge_member3   # or wherever your Week 1 folder lives
unzip -o ~/Downloads/Vision_Edge_Member3_Week2.zip -d .
```
`-o` overwrites existing files cleanly (your README.md gets updated,
nothing from Week 1 is lost — the new Week 2 scripts and docs just get
added alongside it).

## 2. If this is the first time connecting to GitHub (skip if already done)
```bash
git init
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git remote add origin https://github.com/createwithajay/VisionEdge.git
git fetch origin
```

## 3. Switch to (or create) your branch
```bash
git checkout member3-ai-inference 2>/dev/null || git checkout -b member3-ai-inference
```

## 4. Stage and commit — following the 15-commit plan
See `docs/WEEK2_COMMIT_PLAN.md` for the suggested commit breakdown. Example
for the first one:
```bash
git add scripts/04_benchmark_pytorch_vs_tensorrt.py scripts/05_streaming_inference_loop.py scripts/06_vram_leak_profiler.py scripts/run_week2.sh
git commit -m "chore: scaffold week2 benchmark and profiling scripts"
```
Then continue staging/committing in logical chunks as you actually build
and test each piece — see the commit plan doc for the full suggested
sequence and reasoning.

## 5. Push to GitHub
```bash
git push -u origin member3-ai-inference
```
If prompted for a password, use a **Personal Access Token** instead (GitHub
no longer accepts account passwords for git operations):
1. https://github.com/settings/tokens → Generate new token (classic)
2. Check the `repo` scope → Generate
3. Use the token as your password when prompted

## 6. Open a Pull Request (when ready for Team Lead review/merge)
On GitHub, go to the repo → you'll see a prompt to open a PR from
`member3-ai-inference` into `main`. This is how your Team Lead handles
merges per the SOP — don't push directly to `main` yourself.

---

## Quick reference — running everything fresh after cloning
If you're setting this up on a new machine, or a teammate wants to run
your Week 2 scripts:
```bash
git clone https://github.com/createwithajay/VisionEdge.git
cd VisionEdge
git checkout member3-ai-inference
python3.11 -m venv venv_linux
source venv_linux/bin/activate
pip install -r requirements.txt
# Week 1 (build the engine first)
python -c "from ultralytics import YOLO; YOLO('yolov10n.pt')"
mv yolov10n.pt models/
bash scripts/run_week1.sh models/yolov10n.pt
# Week 2 (the performance audit)
bash scripts/run_week2.sh models/yolov10n.pt engines/yolov10n_fp16.engine
```
