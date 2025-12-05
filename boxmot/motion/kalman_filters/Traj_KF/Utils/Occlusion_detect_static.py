# side_occlusion.py
from typing import List, Dict, Tuple, Any, Optional


class OcclusionDetect:
    """
    Stateless sideways (lateral) occlusion detector using only fixed thresholds.
    - You provide and own `history` per track (list[dict]).
    - Call `step(pred_xywh, history, det_xywh)` each frame.
    - Returns: (occ_flag, updated_history, info_dict)
    """

    def __init__(
        self,
        *,
        # history / hysteresis
        maxlen: int = 10,
        enter_needed: int = 2,
        recover_needed: int = 4,
        # normalisation & gating
        min_norm_w: float = 30.0,
        min_norm_h: float = 30.0,
        min_gate_h: float = 20.0,
        # fixed thresholds (single-frame)
        edge_asym_diff: float = 0.10,
        edge_any_min: float   = 0.10,
        asym_ratio_min: float = 0.25,
        min_width_drop: float = -0.15,
        cx_shift_min: float   = 0.03,
        height_ok: float      = 0.06,
        bottom_stable: float  = 0.12,
        votes_needed: int     = 3,
    ):
        self.cfg = dict(
            maxlen=maxlen,
            enter_needed=enter_needed,
            recover_needed=recover_needed,

            min_norm_w=min_norm_w,
            min_norm_h=min_norm_h,
            min_gate_h=min_gate_h,

            edge_asym_diff=edge_asym_diff,
            edge_any_min=edge_any_min,
            asym_ratio_min=asym_ratio_min,

            min_width_drop=min_width_drop,
            cx_shift_min=cx_shift_min,
            height_ok=height_ok,
            bottom_stable=bottom_stable,
            votes_needed=votes_needed,
        )

    # ---------- main API (stateless) ----------
    def step(
        self,
        pred_xywh: Tuple[float, float, float, float],
        history: List[Dict[str, Any]],
        det_xywh: Tuple[float, float, float, float],
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Decide occlusion for this frame and update the provided history
        using fixed thresholds only (no adaptive / MAD-based scaling).
    
        Returns:
            occ_flag (bool), updated_history (list), info (dict)
        """
        cfg = self.cfg
        hist = list(history or [])
        p_x, p_y, p_w, p_h = pred_xywh
        d_x, d_y, d_w, d_h = det_xywh
        eps = 1e-6

        # early gate for tiny predictions
        if p_h < cfg["min_gate_h"]:
            
            entry = self._make_entry(occ=False)
            hist.append(entry)
            if len(hist) > cfg["maxlen"]:
                hist = hist[-cfg["maxlen"]:]
            info = self._make_info(
                flagged_now=False, phase="p_h error", side=None, votes=0, adaptive=False,
                edge_asymmetry=False, width_undershoot=False,
                centroid_towards_side=False, height_consistent=False,
                bottom_stable=False,
                rx=0.0, ry=0.0, rw=0.0, rh=0.0,
                eL=0.0, eR=0.0, eT=0.0, eB=0.0,
                AEC=0.0, asym_ratio=0.0,
            )
            return False, hist, info

        # normalisation
        norm_w = p_w
        norm_h = p_h

        # --- residuals (normalised differences between pred and det) ---
        dx, dy = d_x - p_x, d_y - p_y
        dw, dh = d_w - p_w, d_h - p_h
        rx = dx / norm_w
        ry = dy / norm_h
        rw = dw / norm_w
        rh = dh / norm_h

        # --- edges in image space ---
        p_left,  p_right  = p_x - 0.5 * p_w, p_x + 0.5 * p_w
        p_top,   p_bottom = p_y - 0.5 * p_h, p_y + 0.5 * p_h
        d_left,  d_right  = d_x - 0.5 * d_w, d_x + 0.5 * d_w
        d_top,   d_bottom = d_y - 0.5 * d_h, d_y + 0.5 * d_h

        # --- normalised edge offsets ---
        edge_thres = 0.05
        eL = (d_left   - p_left)   / norm_w
        eL = 0 if abs(eL) <= edge_thres else abs(eL)
        eR = (d_right  - p_right)  / norm_w
        eR = 0 if abs(eR) <= edge_thres else abs(eR)
        eT = (d_top    - p_top)    / norm_h
        eT = 0 if abs(eT) <= edge_thres else abs(eT) 
        eB = (d_bottom - p_bottom) / norm_h
        eB = 0 if abs(eB) <= edge_thres else abs(eB)
        
        

        # --- compulsory edge asymmetry (fixed thresholds only) ---
        AEC = abs(eL - eR)                       # magnitude difference between edge shifts
        asym_ratio = AEC / (eL + eR + eps)       # how one-sided the change is
        asym_side = "right" if abs(eR) > abs(eL) else "left"

        edge_any = max(eL, eR)
        asymmetry_ok = (
            (AEC >= cfg["edge_asym_diff"]) or
            (edge_any >= cfg["edge_any_min"]) and
            (asym_ratio >= cfg["asym_ratio_min"])
        )

        # --- secondary cues, only meaningful if asymmetry_ok is True ---
        if asymmetry_ok:
            # Width undershoot: detection is narrower than prediction by a margin
            width_undershoot = (rw <= cfg["min_width_drop"])

            if asym_side == "right":
                cx_dir_ok = (rx <= -cfg["cx_shift_min"])
            else:  # occlusion on left
                cx_dir_ok = (rx >= cfg["cx_shift_min"])
            cx_ok = cx_dir_ok

            # Height approximately consistent → lateral occlusion, not vertical crop
            height_ok_flag = (abs(rh) <= cfg["height_ok"])

            # Bottom edge stays roughly stable → base of object not chopped
            bottom_ok_flag = (abs(eB) <= cfg["bottom_stable"])
        else:
            width_undershoot = False
            cx_ok = False
            height_ok_flag = False
            bottom_ok_flag = False

        adaptive_used = False  # explicit marker: fixed-threshold mode

        # --- voting over cues ---
        votes_now = (
            int(width_undershoot) +
            int(cx_ok) +
            int(height_ok_flag) +
            int(bottom_ok_flag)
        ) if asymmetry_ok else 0

        flagged_now = asymmetry_ok and (votes_now >= cfg["votes_needed"])

        # --- hysteresis using history you gave us ---
        prev_occ = hist[-1]['occ'] if hist else False
        enter_streak_prev = hist[-1].get('enter_streak', 0) if hist else 0
        recover_streak_prev = hist[-1].get('recover_streak', 0) if hist else 0

        if flagged_now:
            enter_streak = enter_streak_prev + 1
            if prev_occ or (enter_streak >= cfg["enter_needed"]):
                occ = True
                recover_streak = 0
                phase = "occluded"
            else:
                occ = False
                recover_streak = 0
                phase = "arming"
        else:
            enter_streak = 0
            if prev_occ:
                recover_streak = recover_streak_prev + 1
                if recover_streak >= cfg["recover_needed"]:
                    occ = False
                    recover_streak = 0
                    phase = "clean"
                else:
                    occ = True
                    phase = "recovering"
            else:
                recover_streak = 0
                occ = False
                phase = "clean"

        # --- info for your metadata dump ---
        info = self._make_info(
            flagged_now=flagged_now,
            phase=phase,
            side=(asym_side if occ else None),
            votes=votes_now,
            adaptive=adaptive_used,
            edge_asymmetry=asymmetry_ok,
            width_undershoot=width_undershoot,
            centroid_towards_side=cx_ok,
            height_consistent=height_ok_flag,
            bottom_stable=bottom_ok_flag,
            rx=rx, ry=ry, rw=rw, rh=rh,
            eL=eL, eR=eR, eT=eT, eB=eB,
            AEC=AEC, asym_ratio=asym_ratio,
        )

        # --- append compact record to history (used only for hysteresis) ---
        hist.append({
            'occ': bool(occ),
            'enter_streak': int(enter_streak),
            'recover_streak': int(recover_streak),
            'xywh': det_xywh,
            'phase': phase,
        })
        if len(hist) > cfg["maxlen"]:
            hist = hist[-cfg["maxlen"]:]

        return bool(occ), hist, info

    # ---------- info builders ----------
    def _make_entry(self, occ: bool) -> Dict[str, Any]:
        return {
            'rx': 0.0, 'ry': 0.0, 'rw': 0.0, 'rh': 0.0,
            'eL': 0.0, 'eR': 0.0, 'eT': 0.0, 'eB': 0.0,
            'aec': 0.0, 'asym_ratio': 0.0,
            'occ': bool(occ),
            'enter_streak': 0,
            'recover_streak': 0,
        }

    def _make_info(
        self,
        *,
        flagged_now: bool,
        phase: str,
        side: Optional[str],
        votes: int,
        adaptive: bool,
        edge_asymmetry: bool,
        width_undershoot: bool,
        centroid_towards_side: bool,
        height_consistent: bool,
        bottom_stable: bool,
        rx: float, ry: float, rw: float, rh: float,
        eL: float, eR: float, eT: float, eB: float,
        AEC: float, asym_ratio: float,
    ) -> Dict[str, Any]:
        t = self.cfg
        return {
            "flagged_now": bool(flagged_now),
            "occ": phase in {"occluded", "recovering"},
            "phase": phase,
            "side_if_occluded": (side if phase in {"occluded", "recovering"} else None),
            "votes_now": votes,
            "adaptive": adaptive,
            "indicators": {
                "edge_asymmetry": bool(edge_asymmetry),
                "width_undershoot": bool(width_undershoot),
                "centroid_towards_side": bool(centroid_towards_side),
                "height_consistent": bool(height_consistent),
                "bottom_stable": bool(bottom_stable),
            },
            "residuals": {
                "rx": rx, "ry": ry, "rw": rw, "rh": rh,
                "eL": eL, "eR": eR, "eT": eT, "eB": eB,
                "AEC": AEC, "asym_ratio": asym_ratio,
            },
            "thresholds": {
                "edge_asym_diff": t["edge_asym_diff"],
                "edge_any_min": t["edge_any_min"],
                "asym_ratio_min": t["asym_ratio_min"],
                "min_width_drop": t["min_width_drop"],
                "cx_shift_min": t["cx_shift_min"],
                "height_ok": t["height_ok"],
                "bottom_stable": t["bottom_stable"],
                "votes_needed": t["votes_needed"],
                "enter_needed": t["enter_needed"],
                "recover_needed": t["recover_needed"],
                "adaptive_used": adaptive,
                "min_norm_w": t["min_norm_w"],
                "min_norm_h": t["min_norm_h"],
                "min_gate_h": t["min_gate_h"],
            },
        }


# ============================================================
# Occlusion detector variable glossary (non-adaptive version)
# ============================================================
#
# ---------- Inputs ----------
# pred_xywh : tuple[float, float, float, float]
#     Predicted bounding box in (centre_x, centre_y, width, height).
#     Usually from a motion model / Kalman filter.
#
# det_xywh : tuple[float, float, float, float]
#     Detected bounding box in (centre_x, centre_y, width, height)
#     for the same object in the current frame.
#
# history : list[dict]
#     Per-track temporal history owned by the caller.
#     Each entry contains residuals, edge metrics, and hysteresis state:
#         'rx', 'ry', 'rw', 'rh'
#         'eL', 'eR', 'eT', 'eB'
#         'aec', 'asym_ratio'
#         'occ', 'enter_streak', 'recover_streak'
#
# ---------- Main configuration (self.cfg) ----------
# maxlen : int
#     Maximum length of per-track history to keep.
#
# enter_needed : int
#     Number of consecutive "flagged" frames required to ENTER occlusion.
#
# recover_needed : int
#     Number of consecutive "clean" frames required to EXIT occlusion.
#
# min_norm_w : float
#     Minimum width used for normalisation. Prevents tiny widths from
#     exploding residuals.
#
# min_norm_h : float
#     Minimum height used for normalisation. Same idea as min_norm_w.
#
# min_gate_h : float
#     If predicted height p_h is below this, we skip occlusion logic and
#     treat the frame as clean (too small / unreliable prediction).
#
# edge_asym_diff : float
#     Threshold on AEC (edge asymmetry magnitude) for asymmetry_ok.
#
# edge_any_min : float
#     Minimum required movement of *at least one* horizontal edge
#     (max(|eL|, |eR|)) for asymmetry_ok.
#
# asym_ratio_min : float
#     Minimum asymmetry ratio required to treat change as strongly one-sided.
#
# min_width_drop : float
#     Threshold on width residual rw for considering width undershoot.
#     rw <= min_width_drop means detection is narrower than prediction.
#
# cx_shift_min : float
#     Minimum signed shift in centroid x required in the DIRECTION of the
#     occluding side. Applied after abs(rx) >= cx_shift_abs_min.
#
# height_ok : float
#     Maximum allowed |rh| (normalised height change) for height to be
#     considered "consistent" with prediction.
#
# bottom_stable : float
#     Maximum allowed |eB| (bottom-edge shift) for the bottom to be treated
#     as stable (no vertical chopping at the base).
#
# votes_needed : int
#     Number of cues that must be true (width_undershoot, cx_ok,
#     height_ok_flag, bottom_ok_flag) for flagged_now to be True.
#
# edge_asym_diff_abs_min : float
#     Absolute minimum on AEC to avoid triggering on tiny numerical noise.
#
# edge_any_abs_min : float
#     Absolute minimum on edge_any = max(|eL|, |eR|) to ignore very small shifts.
#
# width_drop_abs_min : float
#     Minimum |rw| required before a negative width change counts as
#     "width undershoot".
#
# cx_shift_abs_min : float
#     Minimum |rx| required before centroid shift can be considered,
#     regardless of direction.
#
# ---------- Per-frame derived variables (geometry) ----------
# p_x, p_y, p_w, p_h : float
#     Components of pred_xywh (prediction).
#
# d_x, d_y, d_w, d_h : float
#     Components of det_xywh (detection).
#
# eps : float
#     Small epsilon to avoid division by zero in asym_ratio.
#
# norm_w : float
#     Normalisation width = max(p_w, min_norm_w).
#
# norm_h : float
#     Normalisation height = max(p_h, min_norm_h).
#
# dx : float
#     Raw horizontal shift of detection centre: d_x - p_x.
#
# dy : float
#     Raw vertical shift of detection centre: d_y - p_y.
#
# dw : float
#     Raw change in width: d_w - p_w.
#
# dh : float
#     Raw change in height: d_h - p_h.
#
# ---------- Normalised residuals ----------
# rx : float
#     Normalised horizontal centroid shift: dx / norm_w.
#     > 0 → detection centre is to the right of prediction.
#     < 0 → detection centre is to the left.
#
# ry : float
#     Normalised vertical centroid shift: dy / norm_h.
#
# rw : float
#     Normalised width change: dw / norm_w.
#     < 0 → detection narrower than prediction (candidate width drop).
#
# rh : float
#     Normalised height change: dh / norm_h.
#
# ---------- Box edges (image space) ----------
# p_left, p_right : float
#     Predicted left/right edges of box in image coordinates.
#
# p_top, p_bottom : float
#     Predicted top/bottom edges of box in image coordinates.
#
# d_left, d_right : float
#     Detected left/right edges.
#
# d_top, d_bottom : float
#     Detected top/bottom edges.
#
# ---------- Normalised edge offsets ----------
# eL : float
#     Normalised left-edge shift: (d_left - p_left) / norm_w.
#
# eR : float
#     Normalised right-edge shift: (d_right - p_right) / norm_w.
#
# eT : float
#     Normalised top-edge shift: (d_top - p_top) / norm_h.
#
# eB : float
#     Normalised bottom-edge shift: (d_bottom - p_bottom) / norm_h.
#
# ---------- Asymmetry metrics ----------
# edge_any : float
#     max(|eL|, |eR|). How much at least one horizontal edge moved.
#
# AEC : float
#     Absolute Edge Contrast (asymmetry magnitude):
#     AEC = | |eL| - |eR| |. How different the horizontal edge shifts are.
#
# asym_ratio : float
#     Asymmetry ratio:
#       asym_ratio = AEC / (|eL| + |eR| + eps)
#     Approaches 1 when one edge moves much more than the other.
#
# asym_side : str
#     "right" if |eR| > |eL| else "left".
#     Interpreted as the likely occluding side.
#
# asymmetry_ok : bool
#     True if AEC, edge_any, and asym_ratio all exceed their thresholds
#     (plus absolute minima). This is the compulsory gate for occlusion.
#
# ---------- Cue flags (single-frame, given asymmetry_ok) ----------
# width_undershoot : bool
#     True if detection width is significantly smaller than prediction:
#       rw <= min_width_drop  and  |rw| >= width_drop_abs_min.
#
# cx_mag_ok : bool
#     True if |rx| >= cx_shift_abs_min. Ensures centroid shift is not tiny.
#
# cx_dir_ok : bool
#     True if centroid shift is towards the occluding side:
#         asym_side == "right" → rx must be negative and <= -cx_shift_min
#         asym_side == "left"  → rx must be positive and >=  cx_shift_min
#
# cx_ok : bool
#     Centroid-towards-side cue:
#       cx_ok = cx_mag_ok and cx_dir_ok
#
# height_ok_flag : bool
#     True if |rh| <= height_ok, meaning height is consistent enough that
#     we treat the occlusion as lateral rather than vertical.
#
# bottom_ok_flag : bool
#     True if |eB| <= bottom_stable, meaning bottom edge is stable and
#     we don’t think the base has been chopped.
#
# ---------- Voting / per-frame decision ----------
# votes_now : int
#     Number of cues satisfied in this frame:
#       votes_now = sum(
#           width_undershoot,
#           cx_ok,
#           height_ok_flag,
#           bottom_ok_flag
#       )  if asymmetry_ok else 0
#
# flagged_now : bool
#     Per-frame occlusion flag (before temporal hysteresis):
#       flagged_now = asymmetry_ok and (votes_now >= votes_needed)
#
# ---------- Hysteresis (temporal smoothing) ----------
# prev_occ : bool
#     Occlusion state in the previous frame (from last history entry).
#
# enter_streak_prev : int
#     Previous count of consecutive flagged_now == True frames
#     while not yet fully occluded.
#
# recover_streak_prev : int
#     Previous count of consecutive flagged_now == False frames
#     while still in occlusion.
#
# enter_streak : int
#     Updated enter streak:
#         if flagged_now: enter_streak_prev + 1
#         else: 0
#
# recover_streak : int
#     Updated recover streak:
#         if not flagged_now and prev_occ: recover_streak_prev + 1
#         else: 0
#
# occ : bool
#     Final occlusion state for this frame after hysteresis.
#     True if we are in "occluded" or "recovering" phases.
#
# phase : str
#     One of:
#       "clean"      → no current or recent occlusion
#       "arming"     → flagged_now True but enter_streak < enter_needed
#       "occluded"   → fully in occlusion
#       "recovering" → still occluded but accumulating evidence to exit
#
# ---------- Info / metadata dictionaries ----------
# info : dict
#     Human-readable / serialisable metadata returned by step():
#       "flagged_now"       : bool
#       "occ"               : bool (same as phase in {"occluded","recovering"})
#       "phase"             : str
#       "side_if_occluded"  : "left"/"right"/None
#       "votes_now"         : int
#       "adaptive"          : always False in this non-adaptive version
#       "indicators"        : dict of the cue flags
#       "residuals"         : dict of rx, ry, rw, rh, eL, eR, eT, eB, AEC, asym_ratio
#       "thresholds"        : a copy of the main cfg thresholds
#
# _make_entry(occ) : dict
#     Produces a minimal history entry with residuals set to 0 and
#     hystersis state initialised.
#
# _make_info(...) : dict
#     Helper to package all metadata for logging / visualisation.
#
# ============================================================
