# Franka VLA — cube lifting from vision and language in Isaac Sim

Teaching a Franka Panda arm to pick up a cube using **only camera images and a
language instruction** — no privileged access to the cube's coordinates at
inference time. The policy is [SmolVLA](https://huggingface.co/blog/smolvla)
(450M parameters, flow matching, action chunking), trained by imitation from a
scripted expert that *does* know where the cube is.

> **Status: the closed loop works — 76/200 (38%) on the best model.**
> For most of this project it read 0/20. That number was a measurement bug, not
> the policy — rows 8 and 9 of [the mismatch table](#what-the-measurements-revealed).

![Eight parallel episodes, three of them successful](media/rollout.gif)

*Eight episodes running in parallel, front camera, ~5 s each. Tiles 2, 5 and 7
grasp the cube and lift it; the rest come down beside it and never close the
gripper. That few-centimetre miss is what "perception is at 30–40 mm and
grasping needs 20 mm" looks like.*

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
weeks anchored the belief that the information was in the images and SmolVLA
was simply failing to use it. The probe split train/validation **by frame**, but the
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
appeared **nine** separate times and cost more than everything else combined.
The last two are the ones that had been hiding the working policy:

| # | mismatch | how it showed up |
|---|---|---|
| 1 | training data collected with a *different camera pose* than eval | depth (x) collapsed live, lateral (y) was fine |
| 2 | 8 parallel envs when collecting, 1 when evaluating | x slope 0.202 → 0.494 once matched |
| 3 | frame 0 of every episode was the *previous* episode's render | all localization numbers were artifacts |
| 4 | success filter counted a cube *flung into the air* as a success | one episode out of 550 broke normalization (action std 0.113 → 2.466) |
| 5 | expert's 0.2 s REST phase wrote `des_ee_pose = ee_pose` | unlearnable labels; also inflated the x action std 7× |
| 6 | LeRobot ignores dataset cameras if the checkpoint config lists its own | a third camera was about to be *silently dropped* from training |
| 7 | eval's warmup stepped the sim with a **zero action** | in an IK-Abs action space that is an invalid end-effector target: the arm was flung into a pose never seen in training, and the arm dominates the camera view |
| 8 | eval read the cube's height **after** `env.step()` | Isaac Lab auto-resets inside that call, so `final_z` was always the freshly spawned cube (0.055 m) and the success test `final_z > 0.10` **could never pass** |
| 9 | eval broadcast env 0's action to all 8 envs and scored only env 0 | collection gave each env its own action; the same policy scored 2% one way and 38% the other |

Number 6 is worth dwelling on: [`factory.py:305`](https://github.com/huggingface/lerobot)
only fills `input_features` from the dataset **if the policy config's copy is
empty**. Fine-tuning from an existing checkpoint would have trained on two
cameras while we believed it was three — and produced a confident, wrong
conclusion that the third camera does not help.

**5. Never evaluate a model on data it trained on.** A three-camera model scored
an x-slope of 0.903 on its own training set and 0.747 on freshly collected
episodes. The first number looked like a breakthrough. It was memorization.

**6. The replacement metric was also a proxy, and also lied.** After open-loop
metrics were caught lying, a localization probe became "THE metric" and every
decision rested on it for weeks. Once success could actually be measured, the
correlation between that probe and task success turned out to be **+0.31** —
the wrong sign. The model with the best probe score (26.3 mm) has the worst
success rate (1%); the model with the worst probe score (49.4 mm) is second
best (31%). The probe measures one plan at the start of an episode; the task is
a 250-step closed loop, and everything in between — recovering from drift,
gripper timing, not knocking the cube away — is invisible to it. Two branches
were closed on that evidence and both were wrong. Success rate is now the
decision metric; 200 episodes take 8 minutes.

**7. A wrong fix can close a branch for weeks.** Number 7 in the table above —
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

200 episodes per model, each evaluated in the camera regime it was trained in.

| model | regime | success | localization probe |
|---|---|---|---|
| **`train_fixcam`** | fixed camera | **76/200 (38%)** | 31.6 mm |
| `train_dart2` | randomized + DART noise | 62/200 (31%) | 49.4 mm |
| `train_3kam` | randomized, 3 cameras | 35/200 (18%) | 33.6 mm |
| `train_geo3` | randomized, 2 cameras | 24/200 (12%) | 41.4 mm |
| `train_rest` | randomized, REST frames dropped | 22/200 (11%) | 41.4 mm |
| `train_fixdag` | fixed camera + DAgger | 2/200 (1%) | **26.3 mm** |

Two branches had been closed as failures on the broken metric and are in fact
the second and third best: **DART** and the **third camera**.

The first eight episodes of every run are a known artifact — the table's
collision body is not ready on the first reset after `env.reset()`, so the cube
falls through it to the floor (0.021 m instead of 0.079 m) and the policy, which
has never seen the cube that low, fails all eight.

**Why `train_fixdag` collapses** is worth stating, because it is a structural
flaw in DAgger with a scripted expert rather than a tuning problem. The expert
is a state machine whose phase is internal. When the learner never reaches the
grasp pose, the machine never leaves `APPROACH`, so it never emits a close-gripper
command: **96 of 104 DAgger episodes contain no gripper-close label at all**
(41% of frames in expert data, 3.4% here). Half the training set then teaches
"in situations like this, keep the gripper open" — and those are exactly the
situations the policy meets at evaluation. This also explains why DART works
where DAgger fails: DART perturbs the expert's *own* trajectory, so the phase
advances normally and the close-gripper examples survive. The fix is a
phase-agnostic expert that recomputes its phase from geometry each step.

---

## Approaches tried

| approach | outcome |
|---|---|
| Object-centric action space | **kept** — closed-loop targeting improved ~10× |
| Auxiliary cube-position head | **kept** — also gives a direct readout of what the model sees |
| Zeroed state vector | **kept** — removes the shortcut |
| Weighting vision-critical frames | **kept** — vision only matters in ~15 of 250 frames per episode |
| Render-only warmup in eval | **kept** — stepping with a zero action flung the arm out of distribution |
| **Fixed camera** | **best single change** — 12% → 38%. Randomising the camera ±25 cm / ±20° changes the pixel→world mapping every episode, so the model must infer the camera pose before it can locate the cube, from 208 episodes |
| **DART (noise injection)** | **31%** — was wrongly eliminated on the broken metric |
| **Third (side) camera** | **18% vs 12%** — also wrongly eliminated |
| Domain randomization in eval | necessary once training used it, but adding it to eval was the wrong half of the fix: removing it from *training* is what helped |
| Clean + noisy data mixture | no gain |
| Dropping the expert's REST frames | 11% vs 12% — no real effect |
| DAgger | **1%** — see above; the scripted expert stops emitting gripper-close labels |
| Unfreezing the vision encoder | not worth it — frozen SigLIP features beat a from-scratch CNN on the same data |
| Calibrating out the regression-to-mean | 3 mm, not a lever |
| Higher camera resolution | not a lever — 112 px scores *better* than 224 px on the probe |

---

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
