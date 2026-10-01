from .common import ActionContext, ELBOW, WRIST
from .rub_back_of_hand import RubBackOfHand
from .fist_open_close import FistOpenClose
from .finger_stretch import FingerStretch
from .cup_transfer import CupTransfer
from .finger_abduction import FingerAbduction
from .finger_press import FingerPress

REGISTRY = [RubBackOfHand, FistOpenClose, FingerStretch, CupTransfer,
            FingerAbduction, FingerPress]
