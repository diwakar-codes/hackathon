import cv2


def motion_energy(prev_gray, gray):
    """Mean absolute pixel difference between consecutive sampled frames.

    Cheap stand-in for optical flow -- catches motion bursts (accidents,
    fighting) without needing a GPU or a trained model.
    """
    diff = cv2.absdiff(prev_gray, gray)
    return float(diff.mean())
