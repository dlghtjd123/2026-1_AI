from __future__ import annotations

from pathlib import Path


RANDOM_STATE = 42

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw" / "cic-ids2017"
OUT_DIR = ROOT_DIR / "artifacts" / "ae_cgan_bot_multiclass"

CLASS_NAMES = [
    "Benign",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack",
    "FTP-Patator",
    "SSH-Patator",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS Slowloris",
    "Heartbleed",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
BOT_CLASS_ID = CLASS_TO_ID["Bot"]

ID_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "label",
    "Fwd Header Length.1",
}

LOG_TRANSFORM_HINTS = (
    "duration",
    "packet",
    "pkt",
    "bytes",
    "byts",
    "iat",
    "length",
    "active",
    "idle",
    "header",
    "subflow",
    "win",
)
