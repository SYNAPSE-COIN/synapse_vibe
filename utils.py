from typing import Tuple

import cv2
import numpy as np

def compress_image(img: np.ndarray, fmt: str, quality: int, width: int) -> Tuple[bytes, int, int]:
    quality = int(round(quality))
    width = int(round(width))

    if width != img.shape[1]:
        new_height = int(img.shape[0] * width / img.shape[1])
        resized = cv2.resize(img, (width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    if fmt == "jpeg":
        _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    elif fmt == "png":
        _, buffer = cv2.imencode(".png", resized, [int(cv2.IMWRITE_PNG_COMPRESSION), quality])
    elif fmt == "webp":
        _, buffer = cv2.imencode(".webp", resized, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
    else:
        raise ValueError(f"Format not supported: {fmt}")

    return buffer.tobytes(), quality, width
