import PIL
import cv2
import numpy as np
import base64
import io
from typing import Union, List
from PIL import Image


def video_to_frames(video_path: str, n_frames: int) -> list[PIL.Image]:
    """
    Extract frames from a video file and return them as a list of PIL Images.

    Args:
        video_path: Path to the video file
        n_frames: Number of frames to extract (evenly spaced throughout the video)

    Returns:
        List of PIL Images containing the extracted frames
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        # Set video to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(frame_rgb)
        frames.append(pil_image)

    # Release video capture
    cap.release()

    return frames


def to_base64(
    images: Union[Image.Image, List[Image.Image]], format: str = "JPEG"
) -> Union[str, List[str]]:
    """
    Convert PIL Image(s) to base64 string(s).

    Args:
        images: Single PIL Image or list of PIL Images
        format: Image format for encoding ('JPEG', 'PNG', etc.)

    Returns:
        Single base64 string or list of base64 strings
    """

    def _single_image_to_base64(image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

    if isinstance(images, list):
        return [_single_image_to_base64(img) for img in images]
    else:
        return _single_image_to_base64(images)
