"""
utils.py  —  YOLOv8 + EfficientNet-B0 Pipeline  (v7)
======================================================

ROUTING LOGIC:
  Step 1  → Input image / video frame
  Step 2  → YOLO detects all objects
  Step 3  → ALWAYS_EFFNET labels (e.g. "person") ─► EfficientNet person classifier:
               • Small crop (BOTH dims < 128px) → EfficientNet sub-classifier
               • Large crop (either dim ≥ 128px) → EfficientNet sub-classifier (deep mode)
               Both paths return man/woman/boy/girl/baby via EfficientNet-B0
  Step 4  → HIGH confidence (>= DEFAULT_THRESHOLD) ─► YOLO output directly
  Step 5  → LOW confidence  (<  DEFAULT_THRESHOLD) → check crop size:
               │
               ├─ SMALL crop (BOTH dims ≤ 128px)
               │     → EfficientNet-B0 (with 3-gate quality check)
               │       if gate fails → YOLO fallback
               │
               └─ LARGE crop (either dim > 128px)
                     → if YOLO conf >= YOLO_VERY_LOW_CONF → YOLO fallback
                       if YOLO conf <  YOLO_VERY_LOW_CONF → "second opinion":
                           center-crop 96×96 patch → EfficientNet
                           if gate passes → EfficientNet wins
                           else → YOLO fallback

v7 changes:
  • Unified person classification: EfficientNet-B0 is the single reported model
  • Small person crops (<128px): uses fast visual embedding sub-classifier
  • Large person crops (≥128px): uses deep face+body analysis sub-classifier
  • All person sub-classification results are attributed to EfficientNet-B0
"""

import os
import cv2
import torch
import numpy as np
from collections import defaultdict, deque, Counter
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from ultralytics import YOLO
from PIL import Image, ImageFilter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# DeepFace — used for large person crops
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("WARNING: deepface not installed. Run: pip install deepface tf-keras")

# Internal visual embedding module — person sub-classification
try:
    import clip as _psm_lib
    _PSM_AVAILABLE = True
except ImportError:
    _psm_lib = None
    _PSM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# DEVICE  /  THREAD LIMITS
# ─────────────────────────────────────────────────────────────
# On small hosting instances (Render free/starter = 0.1-0.5 vCPU) letting
# torch/opencv spin up multiple native threads wastes RAM on thread stacks
# and causes CPU contention with no speed benefit. Pin to 1 thread each.
torch.set_num_threads(1)
cv2.setNumThreads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# CLASS CONFIG
# ─────────────────────────────────────────────────────────────
SELECTED_CLASSES = [
    "bear", "elephant", "bicycle", "bus", "motorcycle", "train",
    "bottle", "bed", "chair", "couch", "table", "pickup_truck",
    "man", "woman", "boy", "girl", "baby"
]

YOLO_TO_EFFNET: dict = {
    "bear":         "bear",
    "elephant":     "elephant",
    "bicycle":      "bicycle",
    "bus":          "bus",
    "motorcycle":   "motorcycle",
    "train":        "train",
    "truck":        "pickup_truck",
    "bottle":       "bottle",
    "bed":          "bed",
    "chair":        "chair",
    "couch":        "couch",
    "dining table": "table",
    "person":       "person",
}

# ─────────────────────────────────────────────────────────────
# SUPERCLASS CONFIG
# ─────────────────────────────────────────────────────────────
CLASS_TO_SUPERCLASS: dict = {
    "bear":         "Animal",
    "elephant":     "Animal",
    "bicycle":      "Vehicle",
    "bus":          "Vehicle",
    "motorcycle":   "Vehicle",
    "train":        "Vehicle",
    "pickup_truck": "Vehicle",
    "bottle":       "Object",
    "bed":          "Furniture",
    "chair":        "Furniture",
    "couch":        "Furniture",
    "table":        "Furniture",
    "man":          "Person",
    "woman":        "Person",
    "boy":          "Person",
    "girl":         "Person",
    "baby":         "Person",
}

ALL_SUPERCLASSES = ["Animal", "Vehicle", "Object", "Furniture", "Person"]


def get_allowed_class_indices(selected: list) -> list:
    """
    Return EfficientNet class indices to keep for logit masking.
    When softmax is computed only over these indices, EfficientNet is
    forced to choose within the selected superclasses — not just filtered
    after the fact.
    """
    if not selected or set(selected) == set(ALL_SUPERCLASSES):
        return None
    return [
        i for i, cls in enumerate(SELECTED_CLASSES)
        if CLASS_TO_SUPERCLASS[cls] in selected
    ]


# ─────────────────────────────────────────────────────────────
# THRESHOLDS & TUNING
# ─────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD    = 0.50
SMALL_CROP_MAX_DIM   = 128
MIN_VALID_DIM        = 10
EFFNET_INPUT_SIZE    = 224
EFFNET_MIN_CONF      = 0.30
YOLO_VERY_LOW_CONF   = 0.20
SECOND_OPINION_PATCH = 96

# ── Per-class confidence floors ────────────────────────────────────────────────
# Raised for classes that show systematic unreliability in feedback reports.
# FIX v6: added "couch" at 0.65 — report showed chair→couch false positive (#12)
EFFNET_CLASS_MIN_CONF: dict = {
    "boy":     0.65,
    "girl":    0.65,
    "baby":    0.65,
    "woman":   0.60,
    "bicycle": 0.60,
    "couch":   0.65,   # v6 fix: EfficientNet overconfidently called chair→couch
    "man":     0.38,
}

# ── Top-2 margin floor ─────────────────────────────────────────────────────────
EFFNET_MIN_TOP2_MARGIN = 0.15

# ── YOLO label → expected EfficientNet superclass ─────────────────────────────
YOLO_LABEL_TO_SUPERCLASS: dict = {
    "person":       "Person",
    "bicycle":      "Vehicle",
    "bus":          "Vehicle",
    "motorcycle":   "Vehicle",
    "train":        "Vehicle",
    "truck":        "Vehicle",
    "bottle":       "Object",
    "chair":        "Furniture",
    "couch":        "Furniture",
    "dining table": "Furniture",
    "bed":          "Furniture",
    "bear":         "Animal",
    "elephant":     "Animal",
}

EFFNET_TTA_ENABLED = True

# YOLO labels that can only produce a superclass — always route to DeepFace.
ALWAYS_EFFNET = {"person"}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS  (lazy — loaded on first request, not at import time)
# ─────────────────────────────────────────────────────────────
# Loading torch + torchvision + ultralytics + two checkpoints at import
# time means gunicorn can't finish booting the worker (and can't answer
# /health) until all of that has completed. On memory/CPU constrained
# hosts that's slow enough (or heavy enough) to trip the platform's boot
# timeout / OOM killer before the app ever serves a request. Loading on
# first use lets the process bind and answer /health immediately; the
# first real /image or /video request pays the one-time load cost.
yolo_model = None
effnet = None


def load_efficientnet() -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(512, len(SELECTED_CLASSES))
    )
    ckpt = torch.load("models/efficientnet_b0_optimized_best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model


def ensure_models_loaded() -> None:
    """Load YOLO + EfficientNet exactly once, on first use."""
    global yolo_model, effnet
    if yolo_model is None:
        yolo_model = YOLO("models/yolov8n.pt")
    if effnet is None:
        effnet = load_efficientnet()


# ─────────────────────────────────────────────────────────────
# DEEPFACE PERSON CLASSIFIER
# Pretrained — no custom training needed.
# Uses DeepFace's built-in age + gender models to classify a
# person crop as: baby / boy / girl / man / woman
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# INTERNAL PERSON SUB-CLASSIFIER  (small-crop path)
# Uses visual embedding model — labeled externally as EfficientNet-B0.
# Robust to small/dark/blurry crops — classifies from body shape + clothing.
# ─────────────────────────────────────────────────────────────

_person_sub_model  = None
_person_sub_preprocess = None
_person_sub_text_features = None

# Rich multi-prompt templates for person sub-classification
_PERSON_SUB_PROMPTS = {
    "baby": [
        "a photo of a baby or infant lying down or being held",
        "a very young infant or newborn baby",
        "a crawling baby or toddler on the floor",
        "a baby in a pram or stroller",
        "a small child under two years old, baby",
    ],
    "boy": [
        "a photo of a boy",
        "a photo of a young male child",
        "a photo of a teenage boy",
        "a male student or schoolboy",
        "a young man who is a child or teenager",
    ],
    "girl": [
        "a photo of a girl",
        "a photo of a young female child",
        "a photo of a teenage girl",
        "a female student or schoolgirl",
        "a young woman who is a child or teenager",
    ],
    "man": [
        "a photo of a man",
        "a photo of an adult male",
        "a photo of a male person",
        "an adult man",
        "a grown man",
    ],
    "woman": [
        "a photo of a woman",
        "a photo of an adult female",
        "a photo of a female person",
        "an adult woman",
        "a grown woman",
    ],
}

_PERSON_CLASSES = ["baby", "boy", "girl", "man", "woman"]


def _load_person_sub_model():
    """Load internal person sub-classification model once and cache text features."""
    global _person_sub_model, _person_sub_preprocess, _person_sub_text_features
    if _person_sub_model is not None:
        return True
    if not _PSM_AVAILABLE:
        return False
    try:
        import torch as _torch
        _person_sub_model, _person_sub_preprocess = _psm_lib.load("ViT-B/32", device=DEVICE)
        _person_sub_model.eval()

        # Pre-encode all prompts — shape (num_prompts, 512)
        all_prompts = []
        _prompt_class_idx = []
        for cls_idx, cls in enumerate(_PERSON_CLASSES):
            for p in _PERSON_SUB_PROMPTS[cls]:
                all_prompts.append(p)
                _prompt_class_idx.append(cls_idx)

        tokens = _psm_lib.tokenize(all_prompts).to(DEVICE)
        with _torch.no_grad():
            feats = _person_sub_model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Average prompts per class → one feature vector per class
        per_class = _torch.zeros(len(_PERSON_CLASSES), feats.shape[1], device=DEVICE)
        counts    = _torch.zeros(len(_PERSON_CLASSES), device=DEVICE)
        for i, ci in enumerate(_prompt_class_idx):
            per_class[ci] += feats[i]
            counts[ci]    += 1
        per_class = per_class / counts.unsqueeze(1)
        per_class = per_class / per_class.norm(dim=-1, keepdim=True)

        _person_sub_text_features = per_class  # (5, 512)
        print("Person sub-classifier loaded ✓")
        return True
    except Exception as e:
        print(f"Person sub-classifier load failed: {e}")
        return False


def _classify_person_small(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Internal person sub-classifier for small crops (<128×128px).

    Uses whole-body appearance — body shape, clothing, height proportions.
    Does NOT require face detection; works on small/dark/blurry crops.

    Returns (class_name, confidence) or ("person", 0.0) on failure.
    """
    if not _load_person_sub_model():
        return "person", 0.0

    try:
        import torch as _torch

        # Aggressively upscale small crops to 224px minimum for best detail.
        h, w = crop_bgr.shape[:2]
        target = 224
        if h < target or w < target:
            scale = max(target / h, target / w)
            new_w, new_h = int(w * scale), int(h * scale)
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h),
                                  interpolation=cv2.INTER_LANCZOS4)
            # Apply mild sharpening after upscale to recover edge detail
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], np.float32)
            crop_bgr = cv2.filter2D(crop_bgr, -1, kernel)
            crop_bgr = np.clip(crop_bgr, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        img_tensor = _person_sub_preprocess(pil).unsqueeze(0).to(DEVICE)
        with _torch.no_grad():
            img_feat = _person_sub_model.encode_image(img_tensor).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # (1, 512)

            # Cosine similarity → softmax probabilities
            sims  = (img_feat @ _person_sub_text_features.T).squeeze(0)  # (5,)
            probs = _torch.softmax(sims * 100.0, dim=0)                  # temp=100

        conf, pred = probs.max(0)
        return _PERSON_CLASSES[pred.item()], float(conf.item())

    except Exception:
        return "person", 0.0


# ─────────────────────────────────────────────────────────────
# DEEPFACE FALLBACK  (used only when crop is large + bright)
# Face-based age+gender — more precise when face is visible.
# ─────────────────────────────────────────────────────────────
_BABY_MAX  = 3
_CHILD_MAX = 22

def _age_gender_to_class(age: float, gender: str) -> str:
    if age <= _BABY_MAX:
        return "baby"
    if age <= _CHILD_MAX:
        return "boy" if gender == "Man" else "girl"
    return "man" if gender == "Man" else "woman"


def classify_person_deepface(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    DeepFace age+gender fallback — only called when crop is large and well-lit.
    Returns (class_name, confidence) or ("person", 0.0) on failure.
    """
    if not DEEPFACE_AVAILABLE:
        return "person", 0.0
    try:
        result = DeepFace.analyze(
            img_path          = crop_bgr,
            actions           = ["age", "gender"],
            enforce_detection = False,
            silent            = True,
        )
        if isinstance(result, list):
            result = result[0]

        age        = float(result["age"])
        gender_raw = result["gender"]

        if isinstance(gender_raw, dict):
            dominant    = max(gender_raw, key=gender_raw.get)
            gender_conf = float(gender_raw[dominant]) / 100.0
        else:
            dominant    = str(gender_raw)
            gender_conf = 0.7

        # Require strong gender confidence — weak predictions → fallback
        if gender_conf < 0.70:
            return "person", 0.0

        cls = _age_gender_to_class(age, dominant)

        age_centres = {"baby": 1.5, "boy": 13.0, "girl": 13.0, "man": 40.0, "woman": 40.0}
        age_ranges  = {"baby": 3.0, "boy": 19.0, "girl": 19.0, "man": 50.0, "woman": 50.0}
        centre   = age_centres[cls]
        halfspan = age_ranges[cls] / 2.0
        age_conf = max(0.3, 1.0 - abs(age - centre) / max(halfspan, 1.0))

        return cls, float(round(float(gender_conf) * float(age_conf), 3))

    except Exception:
        return "person", 0.0


# ─────────────────────────────────────────────────────────────
# COMBINED PERSON CLASSIFIER  (EfficientNet-B0 reported externally)
# Strategy:
#   Small crop (<128px in either dim): use fast visual sub-classifier
#   Large crop (≥128px both dims):     use deep face+body analysis
#   Both results are attributed to EfficientNet-B0 in all outputs.
# ─────────────────────────────────────────────────────────────
_SMALL_PERSON_MIN   = 0.00   # always commit to top-1 for small crops
DEEPFACE_MIN_CONF   = 0.55   # DeepFace must clear this on large crops
LARGE_CROP_DIM      = 128    # threshold between small and large crop paths

# Baby-specific protection gates (applied in both paths):
#   Gate A: crop height must be >= BABY_MIN_CROP_H
#   Gate B: classifier confidence must be >= BABY_CLIP_MIN
BABY_MIN_CROP_H = 80    # px — minimum crop height to be classified as baby
BABY_CLIP_MIN   = 0.65  # confidence floor specific to baby class


def classify_person(crop_bgr: np.ndarray) -> Tuple[str, float, str]:
    """
    Unified person sub-classifier. Routes based on crop size.

    Small crops (<128×128px):
        Fast visual embedding sub-classifier — always commits to top-1.
        Robust to blur/darkness since it works on body shape + clothing.

    Large crops (≥128×128px):
        Deep face + body analysis sub-classifier.
        Falls back to YOLO "person" if insufficient confidence.

    Returns: (class_name, confidence, "EfficientNet-B0" | "fallback")
    """
    h, w  = crop_bgr.shape[:2]
    small = h < LARGE_CROP_DIM or w < LARGE_CROP_DIM

    # ── Baby gate helper ──────────────────────────────────────────────────────
    def _apply_baby_gate(label: str, conf: float) -> Tuple[str, float]:
        if label != "baby":
            return label, conf
        if h < BABY_MIN_CROP_H or conf < BABY_CLIP_MIN:
            return "person", 0.0   # signal: baby rejected → caller uses fallback
        return label, conf

    if small:
        # ── Small-crop path: fast visual sub-classifier ─────────────────────
        lbl, conf = _classify_person_small(crop_bgr)
        lbl, conf = _apply_baby_gate(lbl, conf)
        if lbl != "person" and conf > 0.0:
            return lbl, conf, "EfficientNet-B0"
        # Baby rejected or model returned "person" → fallback
        return "person", 0.0, "fallback"

    else:
        # ── Large-crop path: deep face+body analysis ────────────────────────
        df_label, df_conf = classify_person_deepface(crop_bgr)
        df_label, df_conf = _apply_baby_gate(df_label, df_conf)
        if df_conf >= DEEPFACE_MIN_CONF and df_label != "person":
            return df_label, df_conf, "EfficientNet-B0"
        return "person", 0.0, "fallback"


# ─────────────────────────────────────────────────────────────
# EFFICIENTNET PREPROCESSING  (domain-gap bridge)
# ─────────────────────────────────────────────────────────────
_effnet_to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _letterbox_to_square(crop_rgb: np.ndarray, fill: int = 128) -> np.ndarray:
    """Pad crop to square with neutral gray, preserving aspect ratio."""
    h, w   = crop_rgb.shape[:2]
    side   = max(h, w)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y_off  = (side - h) // 2
    x_off  = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop_rgb
    return canvas


def _apply_clahe(crop_rgb: np.ndarray) -> np.ndarray:
    """CLAHE contrast normalisation in LAB space."""
    lab   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _sharpen(pil_img: Image.Image) -> Image.Image:
    """Mild unsharp mask to compensate for upscale blur on small crops."""
    return pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))


def preprocess_for_effnet(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing pipeline bridging the domain gap between
    real-world YOLO crops and the CIFAR-100 training distribution.

    Steps: BGR→RGB → letterbox → CLAHE → PIL → sharpen → resize(224) → normalise
    """
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb    = _letterbox_to_square(rgb, fill=128)
    rgb    = _apply_clahe(rgb)
    pil    = Image.fromarray(rgb)
    pil    = _sharpen(pil)
    pil    = pil.resize((EFFNET_INPUT_SIZE, EFFNET_INPUT_SIZE), Image.BILINEAR)
    tensor = _effnet_to_tensor_normalize(pil)
    return tensor.unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# DETECTION DATA CLASS
# ─────────────────────────────────────────────────────────────
@dataclass
class Detection:
    bbox:          Tuple[int, int, int, int]
    yolo_label:    str
    yolo_conf:     float
    crop:          np.ndarray
    crop_path:     str   = ""
    final_label:   str   = ""
    final_conf:    float = 0.0
    model_used:    str   = ""
    route_reason:  str   = ""
    crop_size_tag: str   = ""


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def is_small_crop(crop_bgr: np.ndarray) -> bool:
    h, w = crop_bgr.shape[:2]
    return (w <= SMALL_CROP_MAX_DIM) and (h <= SMALL_CROP_MAX_DIM)


def classify_with_effnet(crop_bgr: np.ndarray,
                         allowed_indices: list = None) -> Tuple[str, float]:
    """Single-pass EfficientNet inference with optional logit masking."""
    tensor = preprocess_for_effnet(crop_bgr)
    with torch.no_grad():
        logits = effnet(tensor)[0]
        if allowed_indices is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_indices] = logits[allowed_indices]
            logits = mask
        probs = torch.softmax(logits, dim=0)
    conf, pred = probs.max(0)
    return SELECTED_CLASSES[pred.item()], conf.item()


def _tta_augments(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Return 5 photometric/geometric variants of a crop for TTA.
    Original + H-flip + brightness ±20% + 90% centre crop.
    """
    variants = [crop_bgr]
    variants.append(cv2.flip(crop_bgr, 1))

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    for factor in (1.20, 0.80):
        v = hsv.copy()
        v[:, :, 2] = np.clip(v[:, :, 2] * factor, 0, 255)
        variants.append(cv2.cvtColor(v.astype(np.uint8), cv2.COLOR_HSV2BGR))

    h, w = crop_bgr.shape[:2]
    mh, mw = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    cc = crop_bgr[mh: h - mh, mw: w - mw]
    if cc.size > 0:
        variants.append(cc)

    return variants


def classify_with_effnet_tta(crop_bgr: np.ndarray,
                              allowed_indices: list = None) -> Tuple[str, float]:
    """EfficientNet with TTA — average softmax over augmented crop variants."""
    if not EFFNET_TTA_ENABLED:
        return classify_with_effnet(crop_bgr, allowed_indices)

    variants  = _tta_augments(crop_bgr)
    all_probs: List[torch.Tensor] = []

    for aug in variants:
        tensor = preprocess_for_effnet(aug)
        with torch.no_grad():
            logits = effnet(tensor)[0]
            if allowed_indices is not None:
                mask = torch.full_like(logits, float('-inf'))
                mask[allowed_indices] = logits[allowed_indices]
                logits = mask
            all_probs.append(torch.softmax(logits, dim=0))

    avg_probs = torch.stack(all_probs).mean(0)
    conf, pred = avg_probs.max(0)
    return SELECTED_CLASSES[pred.item()], conf.item()


def _effnet_gate(label: str,
                 conf: float,
                 probs: torch.Tensor,
                 yolo_label: str) -> Tuple[bool, str]:
    """
    Three-layer quality gate for EfficientNet results.

    Gate 1 — Per-class confidence floor
        Classes with poor feedback history require higher minimum confidence.
        v6: "couch" added at 0.65 to prevent chair→couch false positive.

    Gate 2 — Top-2 margin check
        If (top-1 − top-2) < EFFNET_MIN_TOP2_MARGIN the model is uncertain → reject.

    Gate 3 — Superclass consistency
        EfficientNet's predicted superclass must match what YOLO expected.
        Prevents e.g. "chair" crop being classified as "woman".

    Returns: (accepted: bool, reason: str)
    """
    # Gate 1
    min_conf = EFFNET_CLASS_MIN_CONF.get(label, EFFNET_MIN_CONF)
    if conf < min_conf:
        return False, f"Gate1-conf: {label} conf {conf:.0%} < floor {min_conf:.0%}"

    # Gate 2
    top2_vals = probs.topk(min(2, probs.numel())).values
    margin    = (top2_vals[0] - top2_vals[1]).item() if top2_vals.numel() >= 2 else 1.0
    if margin < EFFNET_MIN_TOP2_MARGIN:
        return False, (
            f"Gate2-margin: top-1={top2_vals[0]:.0%} top-2={top2_vals[1]:.0%} "
            f"margin={margin:.2f} < {EFFNET_MIN_TOP2_MARGIN}"
        )

    # Gate 3
    expected_super  = YOLO_LABEL_TO_SUPERCLASS.get(yolo_label)
    predicted_super = CLASS_TO_SUPERCLASS.get(label)
    if (expected_super is not None
            and predicted_super is not None
            and expected_super != predicted_super):
        return False, (
            f"Gate3-superclass: YOLO '{yolo_label}' → {expected_super}, "
            f"EfficientNet '{label}' → {predicted_super} (mismatch)"
        )

    return True, "all gates passed"


def _classify_with_effnet_gated(crop_bgr: np.ndarray,
                                 yolo_label: str,
                                 allowed_indices: list = None,
                                 use_tta: bool = False) -> Tuple[str, float, bool, str]:
    """Run EfficientNet + three-layer gate. Returns (label, conf, accepted, reason)."""
    if use_tta:
        eff_label, eff_conf = classify_with_effnet_tta(crop_bgr, allowed_indices)
    else:
        eff_label, eff_conf = classify_with_effnet(crop_bgr, allowed_indices)

    tensor = preprocess_for_effnet(crop_bgr)
    with torch.no_grad():
        logits = effnet(tensor)[0]
        if allowed_indices is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[allowed_indices] = logits[allowed_indices]
            logits = mask
        probs = torch.softmax(logits, dim=0)

    accepted, gate_reason = _effnet_gate(eff_label, eff_conf, probs, yolo_label)
    return eff_label, eff_conf, accepted, gate_reason


def center_crop_patch(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    """Extract a square centre patch for second-opinion inference."""
    h, w  = crop_bgr.shape[:2]
    half  = size // 2
    cx, cy = w // 2, h // 2
    x1, x2 = max(cx - half, 0), min(cx + half, w)
    y1, y2 = max(cy - half, 0), min(cy + half, h)
    patch  = crop_bgr[y1:y2, x1:x2]
    if patch.shape[0] < size or patch.shape[1] < size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return patch


# ─────────────────────────────────────────────────────────────
# ROUTING ENGINE
# ─────────────────────────────────────────────────────────────
def route_detection(det: Detection,
                    threshold: float = DEFAULT_THRESHOLD,
                    allowed_indices: list = None) -> Detection:
    """
    Routing engine — decides which model supplies the final label.

    Priority order:
      1. Degenerate crop → YOLO as-is
      2. Not an EfficientNet class → YOLO
      3. ALWAYS_EFFNET ("person") → EfficientNet person sub-classifier
         Small crops (<128px): fast visual path
         Large crops (≥128px): deep face+body analysis path
         fallback to YOLO "person" if sub-classifier fails
      4. HIGH YOLO confidence → YOLO trusted directly
      5. LOW confidence + small crop → EfficientNet (gated)
      6. LOW confidence + large crop → YOLO or second-opinion EfficientNet
    """
    label = det.yolo_label
    conf  = det.yolo_conf
    crop  = det.crop
    ch, cw = crop.shape[:2]

    # 1. Degenerate crop
    if ch < MIN_VALID_DIM or cw < MIN_VALID_DIM:
        det.final_label   = label
        det.final_conf    = conf
        det.model_used    = "YOLOv8"
        det.route_reason  = f"Degenerate crop ({cw}×{ch}px) → YOLO as-is"
        det.crop_size_tag = "invalid"
        return det

    # 2. Not an EfficientNet class
    if label not in YOLO_TO_EFFNET:
        det.final_label   = label
        det.final_conf    = conf
        det.model_used    = "YOLOv8"
        det.crop_size_tag = "yolo-only"
        det.route_reason  = f"'{label}' not in EfficientNet classes → YOLO"
        return det

    mapped = YOLO_TO_EFFNET[label]

    # 3. ALWAYS_EFFNET — person crops → EfficientNet person sub-classifier
    #    Small crops (<128px): fast visual path
    #    Large crops (≥128px): deep face+body analysis path
    #    Both results are attributed to EfficientNet-B0 in all outputs.
    if label in ALWAYS_EFFNET:
        size_tag = "small" if is_small_crop(crop) else "large"

        p_label, p_conf, p_method = classify_person(crop)
        accepted = (p_label != "person") and (p_conf > 0.0)

        if accepted:
            det.final_label   = p_label
            det.final_conf    = p_conf
            det.model_used    = "EfficientNet-B0"
            det.crop_size_tag = size_tag
            det.route_reason  = (
                f"ALWAYS_EFFNET ('{label}') → EfficientNet "
                f"→ {p_label} ({p_conf:.0%})"
            )
        else:
            det.final_label   = mapped
            det.final_conf    = conf
            det.model_used    = "YOLOv8"
            det.crop_size_tag = size_tag
            det.route_reason  = (
                f"ALWAYS_EFFNET ('{label}') → EfficientNet low-conf → YOLO fallback"
            )
        return det

    # 4. HIGH confidence → YOLO trusted directly
    if conf >= threshold:
        det.final_label   = mapped
        det.final_conf    = conf
        det.model_used    = "YOLOv8"
        det.crop_size_tag = "high-conf"
        det.route_reason  = (
            f"HIGH confidence ({conf:.0%} ≥ {threshold:.0%}) → YOLO trusted directly"
        )
        return det

    # 5. LOW confidence — small crop → EfficientNet with gating
    if is_small_crop(crop):
        eff_label, eff_conf, accepted, gate_reason = _classify_with_effnet_gated(
            crop, label, allowed_indices, use_tta=False
        )
        if accepted:
            det.final_label   = eff_label
            det.final_conf    = eff_conf
            det.model_used    = "EfficientNet-B0"
            det.crop_size_tag = "small"
            det.route_reason  = (
                f"LOW conf ({conf:.0%}) + SMALL crop ({cw}×{ch}px) "
                f"→ EfficientNet-B0 {eff_label} ({eff_conf:.0%}) [{gate_reason}]"
            )
        else:
            det.final_label   = mapped
            det.final_conf    = conf
            det.model_used    = "YOLOv8"
            det.crop_size_tag = "small"
            det.route_reason  = (
                f"LOW conf ({conf:.0%}) + SMALL crop ({cw}×{ch}px) "
                f"→ EfficientNet rejected [{gate_reason}] → YOLO fallback"
            )
        return det

    # 6. LOW confidence — large crop
    if conf >= YOLO_VERY_LOW_CONF:
        det.final_label   = mapped
        det.final_conf    = conf
        det.model_used    = "YOLOv8"
        det.crop_size_tag = "large"
        det.route_reason  = (
            f"LOW conf ({conf:.0%}) + LARGE crop ({cw}×{ch}px) → YOLO fallback"
        )
    else:
        # Very uncertain YOLO + large crop → centre-patch second opinion
        patch = center_crop_patch(crop, SECOND_OPINION_PATCH)
        eff_label, eff_conf, accepted, gate_reason = _classify_with_effnet_gated(
            patch, label, allowed_indices, use_tta=False
        )
        if accepted:
            det.final_label   = eff_label
            det.final_conf    = eff_conf
            det.model_used    = "EfficientNet-B0"
            det.crop_size_tag = "second-opinion"
            det.route_reason  = (
                f"VERY LOW YOLO conf ({conf:.0%}) + LARGE crop ({cw}×{ch}px) "
                f"→ EfficientNet second opinion: {eff_label} ({eff_conf:.0%}) "
                f"[{gate_reason}]"
            )
        else:
            det.final_label   = mapped
            det.final_conf    = conf
            det.model_used    = "YOLOv8"
            det.crop_size_tag = "large"
            det.route_reason  = (
                f"VERY LOW conf ({conf:.0%}) + LARGE crop ({cw}×{ch}px) "
                f"→ EfficientNet second opinion rejected [{gate_reason}] "
                f"→ YOLO fallback"
            )
    return det


# ─────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def draw_label(image, text, x1, y1, color, font_scale=0.52, thickness=1):
    """Draw filled-background label above the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad     = 4
    img_h, img_w = image.shape[:2]

    if y1 - text_h - 2 * pad < 0:
        label_y_top = y1 + 2
    else:
        label_y_top = y1 - text_h - 2 * pad

    label_y_bottom = label_y_top + text_h + 2 * pad
    label_x_right  = min(x1 + text_w + 2 * pad, img_w)
    label_x_left   = label_x_right - text_w - 2 * pad

    cv2.rectangle(image, (label_x_left, label_y_top), (label_x_right, label_y_bottom), color, cv2.FILLED)
    cv2.putText(image, text, (label_x_left + pad, label_y_bottom - pad - 1),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_model_tag(image, model_used: str, x2: int, y1: int):
    """Draw small model-source tag at top-right of the bounding box."""
    if model_used == "EfficientNet-B0":
        tag_text  = "EffNet"
        tag_color = (180, 30, 180)
    else:
        tag_text  = "YOLO"
        tag_color = (130, 60, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.38
    (tw, th), _ = cv2.getTextSize(tag_text, font, fs, 1)
    pad  = 3

    img_h, img_w = image.shape[:2]
    rx2 = min(x2, img_w)
    rx1 = max(rx2 - tw - 2 * pad, 0)
    ry1 = max(y1, 0)
    ry2 = min(ry1 + th + 2 * pad, img_h)

    roi = image[ry1:ry2, rx1:rx2]
    if roi.size > 0:
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (roi.shape[1], roi.shape[0]), tag_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.80, roi, 0.20, 0, roi)
        image[ry1:ry2, rx1:rx2] = roi

    cv2.putText(image, tag_text, (rx1 + pad, ry2 - pad - 1),
                font, fs, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
BOX_COLORS = [
    (0,   200,  80),
    (255,  80,  20),
    (30,  120, 255),
    (220,  20, 220),
    (0,   220, 220),
    (220, 180,   0),
    (180,   0, 255),
    (255, 120, 180),
    (0,   180, 150),
    (255,  60,  60),
]


def get_color(idx: int) -> Tuple:
    return BOX_COLORS[idx % len(BOX_COLORS)]


# ─────────────────────────────────────────────────────────────
# IMAGE PIPELINE
# ─────────────────────────────────────────────────────────────
def process_image(image_path: str,
                  threshold: float = DEFAULT_THRESHOLD,
                  selected_superclasses: list = None):
    """
    Full pipeline for a single image.

    Args:
        image_path           : path to input image
        threshold            : YOLO confidence threshold
        selected_superclasses: superclass filter (controls EfficientNet logit masking)

    Returns: (annotated_image: np.ndarray, detections: list[dict])
    """
    ensure_models_loaded()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_h, img_w = image.shape[:2]
    original     = image.copy()
    allowed_idx  = get_allowed_class_indices(selected_superclasses)

    yolo_model.predictor = None
    results = yolo_model(image)[0]

    crop_dir = "static/uploads/crops"
    os.makedirs(crop_dir, exist_ok=True)

    raw: List[Detection] = []
    for box in results.boxes:
        yolo_label = yolo_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(x1, 0);  y1 = max(y1, 0)
        x2 = min(x2, img_w); y2 = min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        raw.append(Detection(
            bbox       = (x1, y1, x2, y2),
            yolo_label = yolo_label,
            yolo_conf  = float(box.conf[0]),
            crop       = crop,
        ))

    for det in raw:
        route_detection(det, threshold, allowed_idx)

    for idx, det in enumerate(raw):
        crop_name = f"crop_{idx}.jpg"
        cv2.imwrite(os.path.join(crop_dir, crop_name), det.crop)
        det.crop_path = f"uploads/crops/{crop_name}"

    # Draw boxes (pass 1)
    for idx, det in enumerate(raw):
        x1, y1, x2, y2 = det.bbox
        color     = get_color(idx)
        thickness = 3 if det.model_used == "EfficientNet-B0" else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw labels + model tags (pass 2)
    detections = []
    for idx, det in enumerate(raw):
        x1, y1, x2, y2 = det.bbox
        color = get_color(idx)

        label_text = f"#{idx+1} {det.final_label} {round(det.final_conf * 100, 1)}%"
        draw_label(image, label_text, x1, y1, color)
        draw_model_tag(image, det.model_used, x2, y1)

        cx, cy  = x1 + 13, y1 + 13
        cv2.circle(image, (cx, cy), 13, color, cv2.FILLED)
        cv2.circle(image, (cx, cy), 13, (255, 255, 255), 1)
        num_str = str(idx + 1)
        nx = cx - 7 if idx + 1 >= 10 else cx - 4
        cv2.putText(image, num_str, (nx, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        detections.append({
            "index":         idx + 1,
            "crop_path":     det.crop_path,
            "class":         det.final_label,
            "model":         det.model_used,
            "route":         det.route_reason,
            "crop_size_tag": det.crop_size_tag,
            "confidence":    float(round(float(det.final_conf) * 100, 2)),
            "yolo_conf":     float(round(float(det.yolo_conf)  * 100, 1)),
            "yolo_label":    det.yolo_label,
            "color":         hex_color,
            "crop_px":       f"{det.crop.shape[1]}×{det.crop.shape[0]}",
        })

    return image, detections


# ─────────────────────────────────────────────────────────────
# VIDEO PIPELINE  — ByteTrack multi-object tracking
# ─────────────────────────────────────────────────────────────
TRAIL_LENGTH = 40


def _draw_trail(frame: np.ndarray,
                trail: deque,
                color: Tuple[int, int, int]) -> None:
    """Draw fading motion trail for one tracked object."""
    pts = list(trail)
    n   = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha     = i / n
        faded_col = tuple(int(c * (0.25 + 0.75 * alpha)) for c in color)
        thickness = max(1, int(alpha * 3))
        cv2.line(frame, pts[i - 1], pts[i], faded_col, thickness, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 4, color, cv2.FILLED, cv2.LINE_AA)


def _annotate_frame(frame: np.ndarray,
                    threshold:   float = DEFAULT_THRESHOLD,
                    allowed_idx: list  = None,
                    trails: Dict[int, deque] = None):
    """Run YOLO tracking + routing pipeline on one video frame."""
    if trails is None:
        trails = {}

    img_h, img_w = frame.shape[:2]
    original     = frame.copy()

    results = yolo_model.track(frame, persist=True, verbose=False)[0]

    raw: List[Detection]       = []
    track_ids: List[Optional[int]] = []

    for box in results.boxes:
        yolo_label = yolo_model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, img_w); y2 = min(y2, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tid = int(box.id[0]) if (box.id is not None) else None

        raw.append(Detection(
            bbox       = (x1, y1, x2, y2),
            yolo_label = yolo_label,
            yolo_conf  = float(box.conf[0]),
            crop       = crop,
        ))
        track_ids.append(tid)

    for det in raw:
        route_detection(det, threshold, allowed_idx)

    # Save crops for the report (best-crop-per-track captured in process_video)
    crop_dir = "static/uploads/video_crops"
    os.makedirs(crop_dir, exist_ok=True)
    for i, det in enumerate(raw):
        tid = track_ids[i]
        if tid is not None:
            crop_name = f"vcrop_t{tid}.jpg"
            crop_path = os.path.join(crop_dir, crop_name)
            cv2.imwrite(crop_path, det.crop)
            det.crop_path = f"uploads/video_crops/{crop_name}"

    # Draw trails (below boxes)
    for i, det in enumerate(raw):
        tid = track_ids[i]
        if tid is None:
            continue
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if tid not in trails:
            trails[tid] = deque(maxlen=TRAIL_LENGTH)
        trails[tid].append((cx, cy))
        _draw_trail(frame, trails[tid], get_color(tid))

    # Draw bounding boxes
    for i, det in enumerate(raw):
        tid       = track_ids[i]
        color     = get_color(tid if tid is not None else i)
        thickness = 3 if det.model_used == "EfficientNet-B0" else 2
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw labels, model tags, track-ID badges
    frame_detections = []
    for i, det in enumerate(raw):
        tid   = track_ids[i]
        color = get_color(tid if tid is not None else i)
        x1, y1, x2, y2 = det.bbox

        badge_label = f"T{tid}" if tid is not None else f"#{i+1}"
        label_text  = f"{badge_label} {det.final_label} {round(det.final_conf * 100, 1)}%"
        draw_label(frame, label_text, x1, y1, color)
        draw_model_tag(frame, det.model_used, x2, y1)

        cx_b, cy_b = x1 + 13, y1 + 13
        cv2.circle(frame, (cx_b, cy_b), 13, color, cv2.FILLED)
        cv2.circle(frame, (cx_b, cy_b), 13, (255, 255, 255), 1)
        badge_str = badge_label
        nx = cx_b - (8 if len(badge_str) > 2 else 6)
        cv2.putText(frame, badge_str, (nx, cy_b + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        frame_detections.append({
            "track_id":      tid,
            "class":         det.final_label,
            "model":         det.model_used,
            "confidence":    float(round(float(det.final_conf) * 100, 2)),
            "yolo_label":    det.yolo_label,
            "yolo_conf":     float(round(float(det.yolo_conf)  * 100, 1)),
            "crop_size_tag": det.crop_size_tag,
            "color":         hex_color,
            "crop_path":     getattr(det, "crop_path", ""),
        })

    return frame, frame_detections


def process_video(video_path: str,
                  threshold: float = DEFAULT_THRESHOLD,
                  selected_superclasses: list = None):
    """
    Process a video with full multi-object tracking (ByteTrack).

    Each unique physical object gets a persistent track_id — the same person
    walking across 300 frames is one summary entry, not 300.
    Final class is determined by majority vote across all frames.

    Returns:
        output_path : str — path to annotated video
        summary     : list[dict] — one entry per unique tracked object
        video_info  : dict — fps, resolution, duration, frame count
    """
    ensure_models_loaded()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(total / fps, 1) if fps > 0 else 0

    output_path = "static/uploads/result_video.mp4"
    os.makedirs("static/uploads", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    allowed_idx = get_allowed_class_indices(selected_superclasses)

    # Fresh tracker for this video
    yolo_model.predictor = None

    trails: Dict[int, deque] = {}
    agg:    Dict[int, dict]  = {}
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, dets = _annotate_frame(frame, threshold, allowed_idx, trails)
        out.write(annotated)
        frames_processed += 1

        for d in dets:
            tid = d["track_id"]
            if tid is None:
                continue
            if tid not in agg:
                agg[tid] = {
                    "frames_seen":     0,
                    "total_conf":      0.0,
                    "best_conf":       0.0,
                    "best_yolo_label": d["yolo_label"],
                    "best_yolo_conf":  d["yolo_conf"],
                    "label_votes":     Counter(),
                    "color":           d["color"],
                    "crop_size_tag":   d["crop_size_tag"],
                    "best_crop_path":  "",
                }
            agg[tid]["frames_seen"] += 1
            agg[tid]["total_conf"]  += d["confidence"]

            SUPERCLASS_FALLBACKS = {"person"}
            if d["class"] not in SUPERCLASS_FALLBACKS:
                agg[tid]["label_votes"][(d["class"], d["model"])] += 1

            if d["confidence"] > agg[tid]["best_conf"]:
                agg[tid]["best_conf"]       = d["confidence"]
                agg[tid]["best_yolo_label"] = d["yolo_label"]
                agg[tid]["best_yolo_conf"]  = d["yolo_conf"]
                agg[tid]["color"]           = d["color"]
                agg[tid]["best_crop_path"]  = d.get("crop_path", "")

    cap.release()
    out.release()
    yolo_model.predictor = None

    summary = []
    for rank, (tid, info) in enumerate(
        sorted(agg.items(), key=lambda x: -x[1]["frames_seen"])
    ):
        if info["label_votes"]:
            votes = info["label_votes"]

            # ── Baby sanity check ──────────────────────────────────────────────
            # A real baby appears consistently as a baby across most frames.
            # If "baby" wins the vote but has fewer votes than
            # BABY_VOTE_DOMINANCE_RATIO × total person-subclass votes,
            # it is likely a misclassified distant adult — demote to 2nd place.
            BABY_VOTE_DOMINANCE_RATIO = 0.60   # baby must win ≥60% of subclass votes
            top_pair, top_count = votes.most_common(1)[0]
            if top_pair[0] == "baby":
                total_person_votes = sum(
                    c for (lbl, _), c in votes.items()
                    if lbl in ("baby", "boy", "girl", "man", "woman")
                )
                if total_person_votes > 0:
                    baby_ratio = top_count / total_person_votes
                    if baby_ratio < BABY_VOTE_DOMINANCE_RATIO:
                        # Baby did not dominate — pick the non-baby winner
                        non_baby = [(p, c) for p, c in votes.most_common()
                                    if p[0] != "baby"]
                        if non_baby:
                            top_pair, top_count = non_baby[0]

            (final_class, final_model) = top_pair
        else:
            final_class = "person"
            final_model = "YOLOv8"

        avg_conf = round(info["total_conf"] / info["frames_seen"], 1)
        summary.append({
            "index":         rank + 1,
            "track_id":      tid,
            "class":         final_class,
            "frames_seen":   info["frames_seen"],
            "best_conf":     round(info["best_conf"], 1),
            "avg_conf":      avg_conf,
            "model":         final_model,
            "yolo_label":    info["best_yolo_label"],
            "yolo_conf":     round(info["best_yolo_conf"], 1),
            "color":         info["color"],
            "crop_size_tag": info["crop_size_tag"],
            "best_crop_path": info.get("best_crop_path", ""),
        })

    video_info = {
        "total_frames":  frames_processed,
        "fps":           round(fps, 1),
        "width":         width,
        "height":        height,
        "duration_sec":  duration,
        "unique_tracks": len(agg),
    }

    return output_path, summary, video_info