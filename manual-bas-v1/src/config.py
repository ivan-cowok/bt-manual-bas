from dataclasses import dataclass


@dataclass
class Config:
    # --- Possession ---
    # Ball within this factor * player_bbox_height of player center = in possession
    possession_proximity_factor: float = 0.5
    # Minimum consecutive frames to confirm a possession change (A→B or loose→A)
    possession_min_frames: int = 3
    # Minimum detection confidence to include a player in possession logic
    min_detection_confidence: float = 0.70
    # Minimum team confidence to include a player in possession logic
    min_team_confidence: float = 0.80

    # --- Ball absence ---
    # Frames to hold last possession state when ball is missing
    ball_absent_hold_frames: int = 8
    # Frames before resetting possession state entirely
    ball_absent_reset_frames: int = 25

    # --- Pass / Interception ---
    # Minimum ball speed (pixels/frame) to trigger IN_FLIGHT transition
    ball_velocity_threshold: float = 5.0
    # Minimum consecutive in-flight frames to qualify as a pass (filters dribble touches)
    pass_min_flight_frames: int = 4
    # Maximum in-flight frames before abandoning the pass window
    pass_max_flight_frames: int = 75
    # When the kicker is a goalkeeper, allow a much longer flight window.
    # GK distribution kicks arc high and can disappear from the camera for
    # 3+ seconds before landing — well beyond the normal 75-frame limit.
    # Outfield passes are unaffected.
    gk_pass_max_flight_frames: int = 200
    # If in-flight longer than this, always treat as a different player (never a dribble)
    dribble_max_flight_frames: int = 10
    # The peak ball speed during a flight window must reach this threshold for any
    # pass/interception event to fire.  Flights where the ball never exceeded this
    # speed are rolling/loose-ball situations (recoveries), not genuine passes.
    pass_min_peak_flight_speed: float = 0.0
    # Minimum peak flight speed to classify a flight ending at a goalkeeper as a
    # shot+save.  Shots are faster than passes; this prevents a slow back-pass to
    # one's own goalkeeper from being misclassified as a shot.
    shot_min_peak_flight_speed: float = 20.0
    # Flights longer than this frame count are recoveries, not interceptions.
    # A directed pass that gets intercepted is cut short; a long loose-ball pickup
    # belongs to the "recovery" category and should not penalise us as a false
    # interception.
    interception_max_flight_frames: int = 20

    # --- Transform ---
    # Maximum frames to chain affine transforms before resetting (limits accumulated error)
    transform_max_chain_frames: int = 25

    # Maximum consecutive frames the ball can be present-but-loose while in
    # POSSESSED state before possession is considered stale and the state is
    # reset to IDLE.  Prevents a fast ball movement from being misclassified
    # as a "pass from team X" when team X last touched the ball many frames ago.
    # Set small (5) because legitimate kicks depart within 1-2 frames of the
    # player last touching the ball; longer loose gaps mean the ball drifted
    # away before someone else kicked it.
    pass_held_loose_max_frames: int = 5

    # Tracker-side stale-possession threshold.  If the committed team's player
    # has been absent from the ball for more than this many consecutive frames
    # (ball present but not near any eligible player from the committed team),
    # any candidate accumulation from a *different* team returns "loose" instead
    # of the stale committed state.  Prevents the detector from re-entering
    # POSSESSED immediately after resetting to IDLE.
    possession_stale_frames: int = 15

    # Suppress duplicate events of the same type within this frame window
    nms_window_frames: int = 15
