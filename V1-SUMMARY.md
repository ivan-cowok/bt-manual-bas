# Manual Ball Action Spotting — Project Summary

## What This Project Does

Post-match soccer event detection using only the output of a player/ball tracking model (Model 1).
No deep learning. Pure rule-based geometry on per-frame tracking data.

**Target events (emitted):** `pass`, `pass_received`  
**Detected internally but not emitted:** `interception` (used to close flight windows correctly)  
**Input:** `output.json` per clip → **Output:** `result.json` with frame-level predictions

---

## Pipeline Architecture

```
output.json
    │
    ▼
[Parser]          Raw JSON → Frame objects. Infers GK team via neighbor vote.
    │
    ▼
[BallVelocityCalculator]   Transform-compensated velocity. Preserves state across short absences.
    │
    ▼
[PossessionTracker]        Per-frame: which team/player has the ball.
                           Role-aware distance (feet for players, bbox-boundary for GKs).
                           Requires N=3 consecutive frames to commit. Stale detection.
    │
    ▼
[PassInterceptionDetector] State machine: IDLE → POSSESSED → IN_FLIGHT → POSSESSED
                           Fires pass/pass_received/interception events.
    │
    ▼
[NMS]             Suppress duplicate events within nms_window_frames=15.
    │
    ▼
result.json       { "predictions": [{ "frame": N, "action": "pass|pass_received" }] }
```

---

## Key Design Decisions

### Possession Tracking
- Distance metric: `ball_center → player_feet` for outfield players; `ball_center → bbox_boundary` for goalkeepers
- Threshold: `bbox_height × 0.5` (scales with player size in frame)
- Requires 3 consecutive frames to commit → filters noise
- **Stale detection:** if committed player absent from ball for >15 frames, signal becomes stale
- Stale signals blocked in `IDLE` (won't enter POSSESSED) and blocked for interception events in `IN_FLIGHT`

### State Machine (PassInterceptionDetector)
- **IDLE → POSSESSED:** ball detected + known team + not stale
- **POSSESSED → IN_FLIGHT:** ball goes loose at speed ≥ threshold (5 px/frame)
- **IN_FLIGHT → POSSESSED (same team):** fires `pass` + `pass_received`
- **IN_FLIGHT → POSSESSED (different team):** fires `interception` (not emitted)
- **Flight timeout:** 75 frames for outfield, 200 frames for GK kicks
- **Stale interception guard:** stale possession signals in IN_FLIGHT never fire interception events — they extend the flight waiting for a committed receiver
- **Dribble guard:** same `track_id` = dribble, not pass (when both IDs are known)
- **Long loose ball guard:** ball loose in POSSESSED for >5 frames → reset to IDLE

### Goalkeeper Team Inference
- Model outputs `team_name=""` for goalkeepers (no team label from model)
- **Neighbor vote:** for each GK track_id, tally which team's field players appear most often among the 3 nearest players across all frames
- Majority team = GK's team
- Handles second-half side swaps that break position-based inference

### Ball Velocity
- Compensated for camera micro-movement using per-frame affine transform matrices
- Transform chain capped at 25 frames (limits accumulated error)
- State preserved across short ball absences (≤8 frames)

### Competition Scoring Awareness
- False positives are penalized — only emit events with high confidence
- `interception` removed from output (internally useful, too noisy to submit)
- `pass_min_peak_flight_speed = 0.0` (don't filter slow passes)
- `possession_proximity_factor = 0.5` (conservative threshold, reduces FPs)

---

## Scores on Ground-Truth Videos

| Video | Matched | FP | Score |
|-------|---------|-----|-------|
| V1    | 16/17   | 0   | **0.707** |
| V2    | 7/15    | 1   | **0.252** |
| V3    | 7/17    | 1   | **0.133** |

V3 is lower because GT contains shot/save/goal/recovery events we don't detect.  
V2's remaining FP (`pass_received` at f314) is a persistent edge case.

---

## Known Limitations of Current Approach

1. **Look-ahead only.** The state machine is causal — it only sees the future. It can't use context from before or after an event to resolve ambiguity.

2. **Single best possessor.** Only the closest player within threshold is considered at each frame. No ranking of candidates.

3. **Scalar velocity only.** Ball direction (velocity vector) is computed but not used for event classification. Direction would help distinguish shot from pass, identify attack direction, classify interceptions more confidently.

4. **Interception unreliable.** The current interception logic (different team receives after a flight) produces too many FPs to emit safely. True interceptions require distinguishing "ball deflection" from "clean interception" — very hard without directional context.

5. **Recovery not implemented.** Recovery (loose ball pickup after it was nobody's) requires distinguishing a dead-ball pickup from a pass reception. Not attempted.

6. **GK team inference can fail.** Neighbor vote has ~55% confidence in some clips. If field players are sparse near the GK, the vote is noise.

7. **Tracker ID instability.** IDs drift after ~100 frames. Dribble detection degrades. Long clips with many ID reassignments cause false passes.

8. **No temporal smoothing of possession.** A single bad frame (mis-detected ball position) can disrupt the state machine.

9. **No back-pass awareness.** All events are forward-causal. A "back-pass" to a player who just had the ball looks like a dribble.

---

---

## Ball State — Current, Problems, and V2 Proposals

### Layer 1: PossessionTracker (per-frame signal)

This layer answers: *"Which team/player is closest to the ball right now?"*

It outputs a `FramePossession` with one of three team values:

| Value | Meaning |
|-------|---------|
| `"left"` / `"right"` | A player from this team is within `bbox_height × 0.5` of the ball for ≥3 consecutive frames |
| `"loose"` | No player is within threshold, OR ball has been absent too long, OR possession is stale |

Extra flags on `FramePossession`:
- `ball_detected: bool` — False when the ball is absent from the frame
- `is_stale: bool` — True when the committed team's player has been absent from the ball for >15 consecutive frames (possession is carried over but no longer fresh)

**What works well:**
- Role-aware distance (feet for players, bbox boundary for GKs) handles headers and GK catches
- The 3-frame commit requirement filters one-frame noise and mis-detections
- Stale detection prevents the detector from re-attributing a kick to an old possessor

**What is missing / problematic:**
- **Only one candidate at a time.** If two players are within threshold simultaneously (e.g. a contested ball), the closer one wins outright. The loser is invisible to the detector — there is no "contested" signal.
- **Threshold is a hard binary.** A player at `0.49 × bbox_height` = POSSESSED. At `0.51` = loose. One noisy frame flips the signal.
- **No direction awareness.** The tracker doesn't know if the ball is *moving toward* or *away from* a player. A ball flying past a player at speed looks identical to a ball arriving at a player.
- **Stale logic is symmetric.** `is_stale=True` means "old committed team, but a new team is near." We currently use this to block interception events but allow pass_received. In edge cases this is still noisy.

---

### Layer 2: PassInterceptionDetector (event state machine)

This layer answers: *"What event happened based on the possession signal stream?"*

Three states:

| State | Meaning | Entry condition |
|-------|---------|----------------|
| `IDLE` | No established possession | Startup, or after reset |
| `POSSESSED` | A team has confirmed possession | 3-frame commit from PossessionTracker, ball present, not stale |
| `IN_FLIGHT` | Ball was kicked — tracking who receives | Ball goes loose with speed ≥ 5 px/frame while in POSSESSED |

State transitions and events fired:

```
IDLE        → POSSESSED   : possession committed (non-stale, ball detected)
POSSESSED   → IN_FLIGHT   : ball loose + speed ≥ 5 px/frame  → pass timestamp recorded
IN_FLIGHT   → POSSESSED   : same team receives (≥3 frames)   → fires pass + pass_received
IN_FLIGHT   → POSSESSED   : diff team receives (≥3 frames)   → fires interception (not emitted)
IN_FLIGHT   → IDLE        : flight exceeds max frames (75 outfield / 200 GK)  → reset
POSSESSED   → IDLE        : ball loose for >5 frames          → reset (stale possession guard)
POSSESSED   → POSSESSED   : direct low-speed team change      → silent adoption (no event)
```

**What works well:**
- GK-specific flight window (200 frames) handles long aerial distributions
- Dribble guard (same track_id = dribble) prevents false passes on self-possession
- Stale interception guard: stale different-team signal in IN_FLIGHT is blocked, keeping flight open

**What is missing / problematic:**
- **`IN_FLIGHT` is a single undifferentiated state.** It doesn't distinguish:
  - A fast direct pass (short flight, high speed)
  - A lobbed cross (long flight, arc trajectory)
  - A mis-kicked ball rolling loose (low speed throughout)
  - An aerial duel during flight (ball briefly near a player mid-air)
  This means all those cases share the same timeout logic and event attribution.
- **No `CONTESTED` state.** When two players challenge for the ball simultaneously, the detector oscillates between POSSESSED and IN_FLIGHT, generating false events.
- **No `DEAD` / `SET_PIECE` state.** Ball stationary for many frames (throw-in, corner, free kick) should suspend the detector. Currently it accumulates loose-frames and eventually resets, but the reset boundary is imprecise.
- **Causal only.** The state machine processes frames strictly forward. It can't revise a decision once more context arrives. Example: it fires a pass at frame N; at N+5 it discovers the "receiver" immediately lost the ball — the pass attribution is wrong but can't be corrected.
- **`POSSESSED → POSSESSED` silent adoption is dangerous.** A slow-moving stale different-team signal silently switches the attributed kicker team. This is the root cause of several false interceptions (V6 Issue 2 in this project).

---

### Proposed State Additions for V2

#### Add: `CONTESTED`
**When:** ≥2 players from different teams are within possession threshold simultaneously.  
**Why:** Currently this causes the tracker to oscillate between teams every frame, which creates spurious IN_FLIGHT transitions.  
**Behaviour in V2:** In CONTESTED, suppress all team-change transitions. Wait until one team clearly wins (sole player in threshold for ≥3 frames). This naturally handles tackles, challenges, and 50-50 balls without generating fake passes.

#### Add: `DEAD` (or `SET_PIECE`)
**When:** Ball speed < 1 px/frame AND no player in possession threshold for >20 frames.  
**Why:** Throw-ins, corners, goal kicks, and free kicks all look like "ball is loose and stationary." The detector currently drifts into IDLE and misses the subsequent play's first pass.  
**Behaviour in V2:** In DEAD, reset all flight context. When a player commits possession from DEAD, the next kick is the true "pass start" — no orphaned flight windows.

#### Add: `AERIAL`
**When:** Ball is detected but very small relative to player bboxes (indicating it is high in the air), AND speed is high.  
**Why:** Aerial balls during headers or long crosses look like "fast loose ball." If a player is near an aerial ball, current logic commits possession to them — but they may not touch it at all.  
**Behaviour in V2:** In AERIAL, only commit possession if ball decelerates rapidly (indicating it was caught/headed, not just passed by).

#### Remove or merge: `IDLE`
**Why IDLE exists now:** The detector needs a "cold start" state. But in practice, IDLE and the post-reset state of POSSESSED are identical — both wait for the first committed possession. IDLE adds an extra check (`not stale, ball detected`) that duplicates what PossessionTracker already guards.  
**V2 proposal:** Remove IDLE as a separate state. Replace with a `_cold_start: bool` flag on POSSESSED. Simpler, same semantics.

---

### Summary Table

| State | Keep? | Change in V2 | Reason |
|-------|-------|-------------|--------|
| `IDLE` | Merge into POSSESSED | Replace with `cold_start` flag | Redundant with tracker guards |
| `POSSESSED` | Yes, keep | Add CONTESTED sub-case | Core state, works well |
| `IN_FLIGHT` | Yes, keep | Track flight type (direct/aerial/rolling) | Single state handles too many scenarios |
| `CONTESTED` | **Add** | New state | Suppresses false events during challenges |
| `DEAD` | **Add** | New state | Handles set-pieces / stationary ball correctly |
| `AERIAL` | **Add (optional)** | New state | Reduces false possessions during long crosses |

---

## Ideas for V2 (More Scalable Architecture)

### 1. Velocity Vector (Direction-Aware)
Use `(vx, vy)` not just `|v|`. Direction tells you:
- Which goal the ball is heading toward → shot vs pass
- Whether a kick is forward or backward relative to attacking direction
- Angle of ball arrival at a player → headers vs feet receptions

### 2. Rethink Ball State
Current: IDLE / POSSESSED / IN_FLIGHT  
Consider: add `CONTESTED` (two players within threshold), `DEAD` (ball stationary, no nearby players for N frames), `AERIAL` (ball above typical head height based on bbox scale).

### 3. Bidirectional / Self-Attention Event Attribution
Instead of causal state machine, process a sliding window of ~60 frames and ask:
- "Who possessed the ball just before this kick?" (backward look)
- "Who received it?" (forward look)
- Attribution improves dramatically with even a 10-frame look-back

This naturally handles: back-passes, delayed receptions, and mid-air interceptions.

### 4. Possession Candidate Ranking (Top-K)
Instead of committing to the single closest player:
- Keep top-2 or top-3 candidates with distances
- Resolve ambiguity using look-ahead (who eventually controls the ball?)
- Reduces hard commitment errors from momentary proximity

### 5. Event Classifier on Possession Transitions
Rather than hard rules, train a lightweight classifier on the transition:
- Features: `(ball_speed, flight_frames, kicker_team, receiver_team, receiver_role, angle_of_arrival, distance_traveled)`
- Labels: `pass`, `interception`, `recovery`, `dribble`
- Small training set needed; very interpretable

### 6. Recovery Detection
Recovery = ball was loose (no possessor) for >T frames, then a player commits.
Distinguishable from pass_received because: no preceding pass event, flight duration > pass range, ball speed low at reception.

---

## Files

| File | Purpose |
|------|---------|
| `src/config.py` | All tunable thresholds |
| `src/parser.py` | JSON → Frame objects, GK team inference |
| `src/models.py` | Dataclasses (Frame, PlayerDetection, FramePossession, Event) |
| `src/features/possession.py` | PossessionTracker |
| `src/features/velocity.py` | BallVelocityCalculator |
| `src/detectors/pass_detector.py` | PassInterceptionDetector state machine |
| `src/pipeline.py` | Wires everything together, runs NMS |
| `main.py` | CLI entry point, writes result.json |
| `score.py` | Evaluates result.json against ground_truth.json |
| `data/{N}/output.json` | Model 1 tracking output per clip |
| `data/{N}/result.json` | Our predictions |
| `data/{N}/ground_truth.json` | Competition GT (videos 1, 2, 3 only) |
