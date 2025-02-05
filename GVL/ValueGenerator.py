from typing import List
import PIL.Image
from .vlm import GeminiVLM, VLMOutput
from .helpers import video_to_frames


class ValueGenerator:
    """
    Primary interface for generating values from videos using VLM.
    Handles teacher examples and value generation for video frames.
    """

    def __init__(self, task_description: str, n_samples: int = 10, api_key: str = None):
        """
        Initialize the value generator.

        Args:
            task_description: Description of the evaluation task
            n_samples: Number of frames to sample from videos
            api_key: Google API key for Gemini access
        """
        if not api_key:
            raise ValueError("API key is required for ValueGenerator")

        self.task_description = task_description
        self.n_samples = n_samples
        self.teacher_examples = []
        self.vlm = GeminiVLM(api_key=api_key)

    def add_teacher_example(self, video_path: str) -> None:
        """
        Add a teacher example video for comparison.

        Args:
            video_path: Path to the teacher example video file
        """
        frames = video_to_frames(video_path, self.n_samples)
        self.teacher_examples.append(frames)

    def generate_value(self, video_path: str) -> List[VLMOutput]:
        """
        Generate values for frames in the target video.

        Args:
            video_path: Path to the video file to evaluate

        Returns:
            List of VLMOutput objects containing frame information and values
        """
        # Extract frames from video
        inference_frames = video_to_frames(video_path, self.n_samples)

        # Format prompt using VLM
        self.vlm.format_prompt(
            task_desc=self.task_description,
            inference_frames=inference_frames,
            teacher_frames=self.teacher_examples,
        )

        # Call VLM and get results
        results = self.vlm.call_VLM()

        return results
