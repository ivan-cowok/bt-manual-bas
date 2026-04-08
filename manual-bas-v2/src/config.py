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
    # Minimum player detection confidence to participate in possession logic
    min_detection_confidence: float = 0.70
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
    # Max consecutive loose-ball frames while in POSSESSED before resetting to DEAD
    pass_held_loose_max_frames: int = 7

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

    # ------------------------------------------------------------------ #
    # NMS — per event type                                                 #
    # ------------------------------------------------------------------ #
    nms_window_pass: int = 15
    nms_window_pass_received: int = 15
    nms_window_interception: int = 10
    nms_window_recovery: int = 10
