# Franka VLA — cube lifting from vision and language in Isaac Sim

Teaching a Franka Panda arm to pick up a cube using **only camera images and a
language instruction** — no privileged access to the cube's coordinates at
inference time. The policy is [SmolVLA](https://huggingface.co/blog/smolvla)
(450M parameters, flow matching, action chunking), trained by imitation from a
scripted expert that *does* know where the cube is.

> **Status: the closed loop does not work yet (0/20).** Perception is at 36 mm
> median error in the live loop; grasping a 4 cm cube needs better than 20 mm.
> This repository is as much a record of *how the failure was narrowed down* as
> it is a pipeline. See [What the measurements revealed](#what-the-measurements-revealed).

---

## The pipeline

```
 1. collect        Isaac Sim + scripted expert  ──►  HDF5
                   (images, robot state, expert actions, true cube pose)

 2. convert        HDF5  ──►  LeRobot dataset
                   object-centric actions, zeroed state, auxiliary cube target

 3. train          SmolVLA fine-tune (local RTX 3060, 6 GB)

 4. serve          policy_server.py  ── socket ──►  Isaac Sim
                   (three isolated Python envs; raw bytes, not pickle)

 5. evaluate       closed loop in sim + offline localization probe
```

Steps 1 and 4 run in the `isaaclab` conda env, steps 2–3 in `lerobot`. They
cannot share a process: Isaac Sim's Kit runtime pins its own NumPy, so
cross-environment data moves as **raw bytes** over a socket
([`bridge_protocol.py`](scripts/bridge_protocol.py)), never pickle.

---

## What makes this hard

The expert state machine reads the cube's exact pose from the simulator. The
policy never sees it — two RGB cameras and a text instruction, that is all. So
the policy has to *learn to localize the cube from pixels* as a side effect of
imitating the expert's motion.

It turns out that is where almost all the difficulty lives, and that standard
imitation-learning metrics hide it completely.

---

## What the measurements revealed

Most of the effort here went into measurement, not modeling. Five findings that
generalize beyond this project:

**1. Open-loop metrics lie.** Action MAE 5 mm, quaternion MAE 0.0005, gripper
accuracy 100%, loss 0.029 — and the policy could not find the cube at all. Every
one of those compares the policy to *the expert's action sequence*. While the arm
is already moving, "keep doing what you were doing" saturates them. The metric
that mattered was a task-goal probe
([`localize_test.py`](scripts/diagnostics/localize_test.py)): take the policy's
plan, find its lowest point, and compare that XY to where the cube actually is.

**2. A probe that "proved" the data was fine was itself leaking.** A small CNN
trained on the same frames appeared to localize the cube to 12 mm, which for
weeks anchored the belief that the information was in the images and SmolVLA was
simply failing to use it. The probe split train/validation **by frame**, but the
cube is stationary within an episode — so frames of the same episode landed on
both sides and the model could memorize "this scene looks like that, the cube is
there". Re-run with an **episode-level** split, the same architecture scores
82 mm. The claim may still be true; it is no longer demonstrated.

**3. Shortcut learning.** With the robot's joint state in the observation, the
policy learned to read the arm's position instead of looking at the cube — the
expert's trajectory is a deterministic function of the cube pose, so the arm's
own configuration leaks the answer. Fixed by zeroing the state vector and making
actions **object-centric** (x, y expressed relative to the cube, which the model
must predict itself via an auxiliary output head).

**4. The two ends of the pipeline kept disagreeing.** This single failure mode
appeared six separate times, and cost more time than everything else combined:

| # | mismatch | how it showed up |
|---|---|---|
| 1 | training data collected with a *different camera pose* than eval | depth (x) collapsed live, lateral (y) was fine |
| 2 | 8 parallel envs when collecting, 1 when evaluating | x slope 0.202 → 0.494 once matched |
| 3 | frame 0 of every episode was the *previous* episode's render | all localization numbers were artifacts |
| 4 | success filter counted a cube *flung into the air* as a success | one episode out of 550 broke normalization (action std 0.113 → 2.466) |
| 5 | expert's 0.2 s REST phase wrote `des_ee_pose = ee_pose` | unlearnable labels; also inflated the x action std 7× |
| 6 | LeRobot ignores dataset cameras if the checkpoint config lists its own | a third camera was about to be *silently dropped* from training |
| 7 | eval's warmup stepped the sim with a **zero action** | in an IK-Abs action space that is an invalid end-effector target: the arm was flung into a pose never seen in training, and the arm dominates the camera view |

Number 6 is worth dwelling on: [`factory.py:305`](https://github.com/huggingface/lerobot)
only fills `input_features` from the dataset **if the policy config's copy is
empty**. Fine-tuning from an existing checkpoint would have trained on two
cameras while we believed it was three — and produced a confident, wrong
conclusion that the third camera does not help.

**5. Never evaluate a model on data it trained on.** A three-camera model scored
an x-slope of 0.903 on its own training set and 0.747 on freshly collected
episodes. The first number looked like a breakthrough. It was memorization.

**6. A wrong fix can close a branch for weeks.** Number 7 in the table above —
the warmup — was attempted once, nine days before it was found. The attempted
fix held the arm in place, but read the end-effector pose from a stale buffer
right after reset, so the arm was commanded to the *previous* episode's pose. It
measured worse than the zero action, and a comment went into the code saying to
keep the zero action. Both behaviors were broken; the measurement compared two
bugs and the losing one became a documented rule. The eventual fix avoids both
by issuing no command at all — the warmup only ever needed to refresh the camera
(`sim.render()`), not to step physics.

---

## Current numbers

Measured on `s3_val404` — 32 episodes none of the models have ever seen.

| model | cameras | offline median | closed loop, step 0 | live x correlation |
|---|---|---|---|---|
| `train_geo3` | 2 (front + wrist) | 41.4 mm | **36.4 mm** | **0.840** |
| `train_rest` | 2 (front + wrist) | 41.4 mm | not re-measured | — |
| `train_3kam` | 3 (+ side) | **33.6 mm** | not re-measured | — |

Grasp threshold is 20 mm. The closed loop still scores 0/20.

The live-versus-offline gap that dominated this project for weeks is **closed**:
live perception (36.4 mm) is now slightly better than offline (41.4 mm). It was
the warmup bug — see finding 7. Every closed-loop number measured before that fix
was taken with the arm flung out of distribution and is invalid; the two models
marked "not re-measured" are the first thing to redo.

What remains is a single, clearly stated problem: **perception is at 36 mm and
grasping needs 20 mm.** Because the live gap is closed, offline improvements
should now transfer to the live loop — an assumption that was not true before.

---

## Approaches tried

| approach | outcome |
|---|---|
| Object-centric action space | **kept** — closed-loop targeting improved ~10× |
| Auxiliary cube-position head | **kept** — also gives a direct readout of what the model sees |
| Zeroed state vector | **kept** — removes the shortcut |
| Domain randomization in eval | **kept** — eval had none while all training data was randomized |
| Weighting vision-critical frames | **kept** — vision only matters in ~15 of 250 frames per episode |
| DART (noise injection) | degradation curve flattened +80% → +21%, but the error floor doubled |
| Clean + noisy data mixture | no gain (clean 22 > mixture 38 > noisy 49 mm) |
| Dropping the expert's REST frames | fixed two real bugs, but did not close the live gap |
| Third (side) camera | offline 41 → 34 mm; live result invalid (measured before the warmup fix) |
| Unfreezing the vision encoder | **not worth it** — frozen SigLIP features beat a from-scratch CNN on the same data (61 vs 82 mm) |
| Render-only warmup in eval | **kept** — closed the live/offline gap (60 → 36 mm) |

---

## Repository layout

```
├── scripts/
│   ├── collect_demos.py          1. expert data collection in Isaac Sim
│   ├── pick_lift_sm.py              the expert state machine
│   ├── domain_randomizer.py         lighting / camera / table randomization
│   ├── convert_to_lerobot.py     2. HDF5 → LeRobot, with the data-shaping flags
│   ├── policy_server.py          3. serve the policy over a socket
│   ├── eval_policy_isaacsim.py   4. closed-loop evaluation
│   ├── bridge_protocol.py           raw-byte cross-environment protocol
│   ├── isaac_shutdown.py            watchdog for Isaac Sim's hanging close()
│   └── diagnostics/
│       ├── localize_test.py         ← THE metric
│       ├── vision_probe.py          is the information in the image? (small CNN)
│       └── ...
├── results/
│   ├── chain_*.sh                   32 experiment recipes, one per hypothesis
│   ├── sonuc_*.log                  their measured outputs
│   └── *.png                        camera-geometry evidence
└── docs/                            project memory (in Turkish)
    ├── DURUM.md                     where things stand, what is next
    ├── SONUCLAR.md                  every model × every test, and every
    │                                hypothesis that was eliminated
    ├── KOMUTLAR.md                  commands and pitfalls
    └── YOL_HARITASI.md              phases and design decisions
```

The `docs/` notes are written in Turkish — they are the working record kept
during development, including the reasoning behind every dead end.

Data, model weights, GIFs and progress-bar logs are gitignored. Every dataset
can be regenerated from the recipe in `results/`.

---

## Reproducing

```bash
# 1. collect (isaaclab env) — 100-episode batches; 200 at once needs ~15 GB RAM
python -u scripts/collect_demos.py --num_envs 8 --num_episodes 100 \
  --env_spacing 25.0 --seed 101 --headless --side_cam \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  --out data/franka_lift.hdf5

# 2. convert (lerobot env)
python -u scripts/convert_to_lerobot.py --hdf5 data/franka_lift.hdf5 \
  --repo_id franka_lift --zero_state --object_centric --aux_cube \
  --max_frames 150 --repeat_early 2 --early_len 60 --skip_first 12 \
  --only_success --max_abs 1.5 --root <lerobot-root>

# 3. train
lerobot-train --dataset.repo_id=franka_lift --dataset.root=<...> \
  --policy.path=<base-or-checkpoint> --policy.device=cuda \
  --batch_size=32 --steps=4000 --save_freq=4000

# 4. measure — this is the number that matters
HDF5=<held-out.hdf5> K=8 FRAME=2 OBJ_CENTRIC=1 REST_SKIP=12 \
  python scripts/diagnostics/localize_test.py <checkpoint> 45
```

The `chain_*.sh` files in `results/` run these end to end, each with a
post-conversion statistics gate that aborts before training if normalization
looks broken — added after three separate runs were wasted on corrupted data.

---

## Environment

| | |
|---|---|
| GPU | RTX 3060 Laptop, **6 GB** (three 224px camera streams fit at batch 32, barely) |
| Sim | Isaac Sim 4.5.0 + Isaac Lab v2.1.0 — pinned; driver 535 blocks 5.x |
| Policy | SmolVLA 450M, chunk size 50, 50 Hz control |
| Envs | `isaaclab` (sim), `lerobot` (training/inference) — separate, bridged by socket |

Isaac Sim needs `--kit_args="--/rtx/verifyDriverVersion/enabled=false"` on this
driver, and `simulation_app.close()` hangs, hence
[`isaac_shutdown.py`](scripts/isaac_shutdown.py).

---

## Roadmap

**Phase 1 — single cube in simulation** *(current)*. Done when the closed loop
lifts the cube consistently (≥5/10).

**Phase 2 — multiple objects and real language.** Right now every episode uses
the identical instruction, so the model can ignore language entirely and lose
nothing: the V and the A of "VLA" work, the L does not. Adding a second object
and instruction variety ("pick up the red cube") forces a genuine language
grounding, generalizes the auxiliary task from "where is the cube" to "where is
the object named in the instruction", and makes the action distribution
multimodal — which breaks shortcuts like sample averaging.

**Phase 3 — Dobot Nova 5, real hardware.** The action space transfers directly:
the policy emits absolute end-effector pose plus gripper, which an IK-Abs
controller executes in Isaac Lab and a simple IK solver or MoveIt Servo executes
on the real arm. (A full motion planner would break the 50 Hz closed loop.) The
real problem in that phase is data, not the model — the scripted expert needs
the object's pose, which in reality means AprilTag or a calibrated camera plus a
detector. Crucially, the detector drives the *expert* and labels the auxiliary
target; it never enters the policy's input, or the model would learn to look for
the marker instead of the object.
