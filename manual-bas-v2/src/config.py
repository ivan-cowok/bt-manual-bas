from dataclasses import dataclass


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Possession                                                           #
    # ------------------------------------------------------------------ #
    # Ball within this factor × player bbox_height of player = in possession
    possession_proximity_factor: float = 0.5
    # Consecutive frames required to commit a possession change
    possession_min_frames: int = 3
    # Minimum team-classification confidence (None = skip check)
    min_team_confidence: float = 0.80
    # Frames of committed team's absence before possession becomes stale
    possession_stale_frames: int = 15

    # ------------------------------------------------------------------ #
    # Ball absence                                                         #
    # ------------------------------------------------------------------ #
    # Frames to hold last committed state when ball is missing
    ball_absent_hold_frames: int = 8
    # Frames before resetting possession to loose
    ball_absent_reset_frames: int = 25

    # ------------------------------------------------------------------ #
    # DEAD state                                                           #
    # ------------------------------------------------------------------ #
    # Ball speed (px/frame) below which it is considered stationary
    dead_speed_threshold: float = 1.5
    # Consecutive stationary + no-possessor frames required to enter DEAD
    dead_min_frames: int = 20

    # ------------------------------------------------------------------ #
    # CONTESTED state                                                      #
    # ------------------------------------------------------------------ #
    # Max frames to stay in CONTESTED before treating as DEAD (stalemate)
    contested_max_frames: int = 30

    # ------------------------------------------------------------------ #
    # IN_FLIGHT / Pass detection                                           #
    # ------------------------------------------------------------------ #
    # Ball speed threshold (px/frame) to enter IN_FLIGHT
    ball_velocity_threshold: float = 5.0
    # Minimum consecutive in-flight frames to qualify as a real pass
    pass_min_flight_frames: int = 4
    # Maximum in-flight frames for outfield kicks before timeout
    pass_max_flight_frames: int = 75
    # Maximum in-flight frames for GK kicks (long aerial distributions)
    gk_pass_max_flight_frames: int = 200
    # Flight ≤ this = could be a dribble when track_id is unknown
    dribble_max_flight_frames: int = 10
    # Max in-flight frames for an interception (longer = recovery, not interception)
    interception_max_flight_frames: int = 20
    # Frames to wait after a potential interception before confirming/rejecting it.
    # During the wait we check post-contact ball velocity for physical evidence.
    interception_confirm_frames: int = 3
    # Case 1 — ball absorbed: speed_after / speed_before < this ratio → genuine
    interception_speed_drop_ratio: float = 0.5
    # Case 2 — ball deflected: direction change (degrees) > this → genuine
    interception_direction_change_deg: float = 60.0
    # Spike-filter for vel_before: if the most recent pre-contact velocity
    # exceeds this multiple of the one before it, treat it as a tracker
    # glitch and use the calmer, earlier reading instead.
    interception_vel_spike_ratio: float = 1.9
    # Max consecutive loose-ball frames while in POSSESSED before resetting to DEAD
    pass_held_loose_max_frames: int = 7
    # After a "recovery" event, the new possessor may have obtained the ball via a
    # tracker glitch or brief loose-ball pickup.  Require at least this many
    # same-team possession frames before attributing a subsequent kick to that team.
    # Applied ONLY after recovery events (not after pass/interception/dead).
    min_post_recovery_possession_frames: int = 3
    # Minimum consecutive absence frames during IN_FLIGHT before we require
    # 2 confirmation frames on the next reception (prevents premature
    # pass_received at the first ball detection after a long ball-absent gap).
    reception_long_absence_threshold: int = 5
    # Ball position jump (px, raw pixel distance, no camera compensation) that
    # combined with the ball disappearing in the NEXT frame, indicates a
    # tracker glitch (e.g. a white boot briefly detected as the ball).
    # The Pipeline marks FramePossession.ball_glitch=True when both conditions
    # are met; the PassDetector then treats that frame as ball-not-detected.
    ball_glitch_speed_threshold: float = 100.0

    # ------------------------------------------------------------------ #
    # Shot suppressor (internal only — never emitted)                     #
    # ------------------------------------------------------------------ #
    # Minimum peak speed for a flight ending at the opposing GK to be
    # treated internally as a shot (suppresses false pass/pass_received)
    shot_min_peak_flight_speed: float = 20.0

    # ------------------------------------------------------------------ #
    # Look-back buffer                                                     #
    # ------------------------------------------------------------------ #
    # Number of recent FramePossession frames passed to the detector
    lookback_window_frames: int = 10

    # ------------------------------------------------------------------ #
    # Velocity / transform                                                 #
    # ------------------------------------------------------------------ #
    # Maximum affine-transform chain length before reset (limits drift)
    transform_max_chain_frames: int = 25
    # Maximum consecutive-frame gap in the ball track before the velocity is
    # considered unreliable (multi-frame NO-BALL gaps inflate apparent speed,
    # causing false IN-FLIGHT entries from stale positions).  When the gap
    # exceeds this value the velocity calculator returns None → speed = 0.
    max_velocity_gap: int = 2

    # ------------------------------------------------------------------ #
    # NMS — per event type                                                 #
    # ------------------------------------------------------------------ #
    nms_window_pass: int = 15
    nms_window_pass_received: int = 15
    nms_window_interception: int = 10
    nms_window_recovery: int = 10
