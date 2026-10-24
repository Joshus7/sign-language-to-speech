"""Draw hand skeletons and small labels onto OpenCV frames."""
from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as L
from .hands import HandsResult

RIGHT_COLOR = (90, 220, 90)    # BGR green
LEFT_COLOR = (235, 170, 60)    # BGR blue-ish
WHITE = (245, 245, 245)


def draw_hands(frame: np.ndarray, result: HandsResult, thickness: int | None = None) -> np.ndarray:
    h, w = frame.shape[:2]
    t = thickness or max(1, int(round(min(h, w) / 300)))
    for hand in result.hands:
        pts = (hand.landmarks[:, :2] * np.array([w, h])).astype(int)
        color = LEFT_COLOR if hand.is_left else RIGHT_COLOR
        for a, b in L.CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, t + 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), t + 2, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p), t + 2, WHITE, 1, cv2.LINE_AA)
    return frame


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int] = (12, 36),
               color: tuple[int, int, int] = WHITE, scale: float | None = None) -> np.ndarray:
    h = frame.shape[0]
    scale = scale or max(0.6, h / 600)
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(frame, (x - 6, y - tht - 8), (x + tw + 6, y + base + 4), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)
    return frame
