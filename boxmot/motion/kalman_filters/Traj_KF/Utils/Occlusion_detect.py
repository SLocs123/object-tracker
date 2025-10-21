# side_occlusion.py
from typing import List, Dict, Tuple, Any, Optional

class OcclusionDetect:
    """
    Stateless sideways (lateral) occlusion detector.
    - You provide and own `history` per track (list[dict]).
    - Call `step(pred_xywh, history, det_xywh)` each frame.
    - Returns: (occ_flag, updated_history, info_dict)
    """

    # ---------- small helpers ----------
    @staticmethod
    def _median(xs):
        if not xs: return 0.0
        s = sorted(xs); n = len(s); m = n // 2
        return s[m] if n % 2 else 0.5 * (s[m-1] + s[m])

    @staticmethod
    def _mad(xs, eps=1e-9):
        if not xs: return 1.0
        med = OcclusionDetect._median(xs)
        return max(OcclusionDetect._median([abs(x - med) for x in xs]), eps)

    def __init__(
        self,
        *,
        # history / hysteresis
        maxlen: int = 10,
        enter_needed: int = 2,
        recover_needed: int = 2,
        # normalisation & gating
        min_norm_w: float = 30.0,
        min_norm_h: float = 30.0,
        min_gate_h: float = 0.0,
        # fixed thresholds (when little clean history)
        edge_asym_diff: float = 0.30,
        edge_any_min: float   = 0.35,
        asym_ratio_min: float = 0.45,
        min_width_drop: float = -0.15,
        cx_shift_min: float   = 0.22,
        height_ok: float      = 0.06,
        bottom_stable: float  = 0.12,
        votes_needed: int     = 3,
        # absolute minima (apply in BOTH fixed & adaptive paths)
        edge_asym_diff_abs_min: float = 0.06,
        edge_any_abs_min: float       = 0.08,
        width_drop_abs_min: float     = 0.10,
        cx_shift_abs_min: float       = 0.12,
        # adaptive thresholds
        use_adaptive: bool = True,
        min_clean_for_adapt: int = 7,
        z_edge_asym: float = 2.7,
        z_edge_any: float  = 3.0,
        z_width_drop: float = -2.2,
        z_cx_shift: float  = 2.3,
        z_height_ok: float = 1.0,
        # MAD floors (in normalised units)
        mad_floor_edges: float = 0.02,
        mad_floor_rw: float    = 0.03,
        mad_floor_rx: float    = 0.03,
        mad_floor_rh: float    = 0.03,
    ):
        self.cfg = dict(
            maxlen=maxlen, enter_needed=enter_needed, recover_needed=recover_needed,
            min_norm_w=min_norm_w, min_norm_h=min_norm_h, min_gate_h=min_gate_h,
            edge_asym_diff=edge_asym_diff, edge_any_min=edge_any_min, asym_ratio_min=asym_ratio_min,
            min_width_drop=min_width_drop, cx_shift_min=cx_shift_min, height_ok=height_ok,
            bottom_stable=bottom_stable, votes_needed=votes_needed,
            edge_asym_diff_abs_min=edge_asym_diff_abs_min, edge_any_abs_min=edge_any_abs_min,
            width_drop_abs_min=width_drop_abs_min, cx_shift_abs_min=cx_shift_abs_min,
            use_adaptive=use_adaptive, min_clean_for_adapt=min_clean_for_adapt,
            z_edge_asym=z_edge_asym, z_edge_any=z_edge_any, z_width_drop=z_width_drop,
            z_cx_shift=z_cx_shift, z_height_ok=z_height_ok,
            mad_floor_edges=mad_floor_edges, mad_floor_rw=mad_floor_rw,
            mad_floor_rx=mad_floor_rx, mad_floor_rh=mad_floor_rh,
        )

    # ---------- main API (stateless) ----------
    def step(
        self,
        pred_xywh: Tuple[float, float, float, float],
        history: List[Dict[str, float]],
        det_xywh: Tuple[float, float, float, float],
    ) -> Tuple[bool, List[Dict[str, float]], Dict[str, Any]]:
        """
        Decide occlusion for this frame and update the provided history.

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
            if len(hist) > cfg["maxlen"]: hist = hist[-cfg["maxlen"]:]
            info = self._make_info(
                flagged_now=False, phase="clean", side=None, votes=0, adaptive=False,
                edge_asymmetry=False, width_undershoot=False, centroid_towards_side=False,
                height_consistent=False, bottom_stable=False,
                rx=0, ry=0, rw=0, rh=0, eL=0, eR=0, eT=0, eB=0, AEC=0, asym_ratio=0
            )
            return False, hist, info

        # normalisation floors
        norm_w = max(p_w, cfg["min_norm_w"])
        norm_h = max(p_h, cfg["min_norm_h"])

        # residuals
        dx, dy = d_x - p_x, d_y - p_y
        dw, dh = d_w - p_w, d_h - p_h
        rx = dx / norm_w
        ry = dy / norm_h
        rw = dw / norm_w
        rh = dh / norm_h

        # edges
        p_left,  p_right  = p_x - 0.5 * p_w, p_x + 0.5 * p_w
        p_top,   p_bottom = p_y - 0.5 * p_h, p_y + 0.5 * p_h
        d_left,  d_right  = d_x - 0.5 * d_w, d_x + 0.5 * d_w
        d_top,   d_bottom = d_y - 0.5 * d_h, d_y + 0.5 * d_h

        eL  = (d_left   - p_left)   / norm_w
        eR  = (d_right  - p_right)  / norm_w
        eT  = (d_top    - p_top)    / norm_h
        eB  = (d_bottom - p_bottom) / norm_h

        # compulsory asymmetry
        AEC = abs(abs(eL) - abs(eR))
        asym_ratio = AEC / (abs(eL) + abs(eR) + eps)
        asym_side = "right" if abs(eR) > abs(eL) else "left"

        clean_hist = [h for h in hist if not h.get('occ', False)]
        have_adapt = cfg["use_adaptive"] and (len(clean_hist) >= cfg["min_clean_for_adapt"])

        def z_with_floor(val: float, key: str, floor: float) -> float:
            arr = [h[key] for h in clean_hist]
            med = self._median(arr); scale = max(self._mad(arr), floor)
            return (val - med) / scale

        if have_adapt:
            eL_z = z_with_floor(eL, 'eL', cfg["mad_floor_edges"])
            eR_z = z_with_floor(eR, 'eR', cfg["mad_floor_edges"])
            AEC_z = (z_with_floor(AEC, 'aec', cfg["mad_floor_edges"])
                     if all('aec' in h for h in clean_hist) else max(abs(eL_z), abs(eR_z)))
            edge_any_z = max(abs(eL_z), abs(eR_z))

            asymmetry_ok = (
                (AEC_z >= cfg["z_edge_asym"]) and
                (edge_any_z >= cfg["z_edge_any"]) and
                (asym_ratio >= cfg["asym_ratio_min"]) and
                (AEC >= cfg["edge_asym_diff_abs_min"]) and
                (max(abs(eL), abs(eR)) >= cfg["edge_any_abs_min"])
            )

            if asymmetry_ok:
                rw_z = z_with_floor(rw, 'rw', cfg["mad_floor_rw"])
                rh_z = z_with_floor(rh, 'rh', cfg["mad_floor_rh"])
                rx_z = z_with_floor(rx, 'rx', cfg["mad_floor_rx"])

                width_undershoot = (rw_z <= cfg["z_width_drop"]) and (abs(rw) >= cfg["width_drop_abs_min"])
                cx_mag_ok = (abs(rx) >= cfg["cx_shift_abs_min"])
                cx_dir_ok = (rx < 0) if asym_side == "right" else (rx > 0)
                cx_ok = (abs(rx_z) >= cfg["z_cx_shift"]) and cx_mag_ok and cx_dir_ok
                height_ok_flag = (abs(rh_z) <= cfg["z_height_ok"])
                bottom_ok_flag = (abs(eB) <= cfg["bottom_stable"])
            else:
                width_undershoot = cx_ok = height_ok_flag = bottom_ok_flag = False

            adaptive_used = True

        else:
            asymmetry_ok = (
                (AEC >= cfg["edge_asym_diff"]) and
                (max(abs(eL), abs(eR)) >= cfg["edge_any_min"]) and
                (asym_ratio >= cfg["asym_ratio_min"]) and
                (AEC >= cfg["edge_asym_diff_abs_min"]) and
                (max(abs(eL), abs(eR)) >= cfg["edge_any_abs_min"])
            )
            if asymmetry_ok:
                width_undershoot = (rw <= cfg["min_width_drop"]) and (abs(rw) >= cfg["width_drop_abs_min"])
                cx_mag_ok = (abs(rx) >= cfg["cx_shift_abs_min"])
                cx_dir_ok = (rx <= -cfg["cx_shift_min"]) if asym_side == "right" else (rx >= cfg["cx_shift_min"])
                cx_ok = cx_mag_ok and cx_dir_ok
                height_ok_flag = (abs(rh) <= cfg["height_ok"])
                bottom_ok_flag = (abs(eB) <= cfg["bottom_stable"])
            else:
                width_undershoot = cx_ok = height_ok_flag = bottom_ok_flag = False

            adaptive_used = False

        votes_now = (int(width_undershoot) + int(cx_ok) + int(height_ok_flag) + int(bottom_ok_flag)) if asymmetry_ok else 0
        flagged_now = asymmetry_ok and (votes_now >= cfg["votes_needed"])

        # hysteresis using history you gave us
        prev_occ = hist[-1]['occ'] if hist else False
        enter_streak_prev = hist[-1].get('enter_streak', 0) if hist else 0
        recover_streak_prev = hist[-1].get('recover_streak', 0) if hist else 0

        if flagged_now:
            enter_streak = enter_streak_prev + 1
            if prev_occ or (enter_streak >= cfg["enter_needed"]):
                occ = True; recover_streak = 0; phase = "occluded"
            else:
                occ = False; recover_streak = 0; phase = "arming"
        else:
            enter_streak = 0
            if prev_occ:
                recover_streak = recover_streak_prev + 1
                if recover_streak >= cfg["recover_needed"]:
                    occ = False; recover_streak = 0; phase = "clean"
                else:
                    occ = True; phase = "recovering"
            else:
                recover_streak = 0; occ = False; phase = "clean"

        # info for your metadata dump
        info = self._make_info(
            flagged_now=flagged_now, phase=phase, side=(asym_side if occ else None),
            votes=votes_now, adaptive=adaptive_used,
            edge_asymmetry=asymmetry_ok, width_undershoot=width_undershoot,
            centroid_towards_side=cx_ok, height_consistent=height_ok_flag,
            bottom_stable=bottom_ok_flag,
            rx=rx, ry=ry, rw=rw, rh=rh, eL=eL, eR=eR, eT=eT, eB=eB, AEC=AEC, asym_ratio=asym_ratio
        )

        # append to *external* history you manage
        hist.append({
            'rx': rx, 'ry': ry, 'rw': rw, 'rh': rh,
            'eL': eL, 'eR': eR, 'eT': eT, 'eB': eB,
            'aec': AEC, 'asym_ratio': asym_ratio,
            'occ': bool(occ),
            'enter_streak': int(enter_streak),
            'recover_streak': int(recover_streak),
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
            'occ': bool(occ), 'enter_streak': 0, 'recover_streak': 0
        }

    def _make_info(
        self, *, flagged_now: bool, phase: str, side: Optional[str], votes: int, adaptive: bool,
        edge_asymmetry: bool, width_undershoot: bool, centroid_towards_side: bool,
        height_consistent: bool, bottom_stable: bool,
        rx: float, ry: float, rw: float, rh: float, eL: float, eR: float, eT: float, eB: float,
        AEC: float, asym_ratio: float
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
                "AEC": AEC, "asym_ratio": asym_ratio
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
                "edge_asym_diff_abs_min": t["edge_asym_diff_abs_min"],
                "edge_any_abs_min": t["edge_any_abs_min"],
                "width_drop_abs_min": t["width_drop_abs_min"],
                "cx_shift_abs_min": t["cx_shift_abs_min"],
                "mad_floor_edges": t["mad_floor_edges"],
                "mad_floor_rw": t["mad_floor_rw"],
                "mad_floor_rx": t["mad_floor_rx"],
                "mad_floor_rh": t["mad_floor_rh"],
                "min_norm_w": t["min_norm_w"],
                "min_norm_h": t["min_norm_h"],
                "min_gate_h": t["min_gate_h"],
                "min_clean_for_adapt": t["min_clean_for_adapt"],
                "z_edge_asym": t["z_edge_asym"],
                "z_edge_any": t["z_edge_any"],
                "z_width_drop": t["z_width_drop"],
                "z_cx_shift": t["z_cx_shift"],
                "z_height_ok": t["z_height_ok"],
            }
        }
