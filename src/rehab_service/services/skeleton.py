"""Display-only coordinates; action state always uses the original inference metrics."""

import math


def pose_message(metrics, session_id, sequence, decoded_ms, completed_ms):
    width, height = metrics["frame_size"]

    def points(values, count, confidence=False):
        if values is None or len(values) != count:
            return None
        output = []
        for point in values:
            x, y = float(point[0]), float(point[1])
            score = float(point[2]) if confidence else None
            valid = (math.isfinite(x) and math.isfinite(y) and 0 <= x <= width and 0 <= y <= height
                     and (not confidence or math.isfinite(score) and 0 <= score <= 1))
            output.append(([round(x / width, 6), round(y / height, 6)]
                           + ([round(score, 6)] if confidence else [])) if valid else None)
        return output

    body = points(metrics.get("pose"), 17, confidence=True)
    return {
        "type": "pose", "version": 1, "session_id": session_id, "seq": sequence,
        "width": width, "height": height, "rotation": 0,
        "decoded_at_ms": decoded_ms, "completed_at_ms": completed_ms,
        "body_state": ("missing" if body is None else
                       "fresh" if metrics.get("pose_t") == metrics["t"] else "cached"),
        "body": body,
        "hands": [{
            "side": hand["side"], "state": hand.get("state", "fresh"),
            "predicted": bool(hand.get("predicted")),
            # MediaPipe's current output has no per-landmark confidence score.
            "confidence": None,
            "points": points(hand["pts"], 21) if hand.get("state", "fresh") == "fresh"
                      and not hand.get("predicted") else None,
        } for hand in metrics.get("display_hands", metrics.get("hands", []))],
        "tracking_confidence": metrics.get("tracking_confidence", 0.0),
    }
