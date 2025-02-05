from typing import List
import PIL.Image
from typing import List
import PIL.Image
import google.generativeai as genai
from .vlm import VLM
from .prompt import Prompt
from .helpers import to_base64


class VLM:
    """
    Base class for Vision Language Model interactions.
    Handles prompt formatting and model calls for value generation.
    """

    def __init__(self):
        """Initialize the VLM instance."""
        pass

    def format_prompt(
        self,
        task_desc: str,
        inference_frames: List[PIL.Image.Image],
        teacher_frames: List[List[PIL.Image.Image]],
    ) -> str:
        """
        Prepare the formatted prompt for the VLM.

        Args:
            task_desc: Description of the evaluation task
            inference_frames: List of frames to evaluate
            teacher_frames: List of lists containing teacher example frames

        Returns:
            Formatted prompt string for the VLM
        """
        raise NotImplementedError("Subclasses must implement format_prompt method")

    def call_VLM(self) -> List[float]:
        """
        Call the vision-language model and get computed values.

        Returns:
            List of computed values for each frame
        """
        raise NotImplementedError("Subclasses must implement call_VLM method")


class GeminiVLM(VLM):
    """
    Gemini implementation of the Vision Language Model.
    Uses Google's Gemini Pro model for vision-language tasks.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Gemini VLM instance.

        Args:
            api_key: Google API key for Gemini access
        """
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        self.prompt_formatter = Prompt()
        self.current_prompt = None

    def format_prompt(
        self,
        task_desc: str,
        inference_frames: List[PIL.Image.Image],
        teacher_frames: List[List[PIL.Image.Image]],
    ) -> str:
        """
        Prepare the formatted prompt for Gemini.

        Args:
            task_desc: Description of the evaluation task
            inference_frames: List of frames to evaluate
            teacher_frames: List of lists containing teacher example frames

        Returns:
            Formatted prompt string
        """
        # Convert images to base64 strings
        inference_b64 = to_base64(inference_frames)
        teacher_b64 = (
            [to_base64(frames) for frames in teacher_frames] if teacher_frames else None
        )

        # Use the Prompt class to format the prompt
        self.current_prompt = self.prompt_formatter.format_prompt(
            task_desc=task_desc,
            inference_video=inference_b64,
            teacher_examples=teacher_b64,
        )

        return self.current_prompt

    def call_VLM(self) -> List[float]:
        """
        Call the Gemini model and parse the response for frame values.

        Returns:
            List of computed values for each frame
        """
        if not self.current_prompt:
            raise ValueError("Prompt must be formatted before calling VLM")

        # Call Gemini model
        response = self.model.generate_content(self.current_prompt)

        # Parse the response text to extract values
        values = []
        for line in response.text.split("\n"):
            if "Task Completion Percentages:" in line:
                # Extract percentage value and convert to float
                try:
                    value_str = line.split("Task Completion Percentages:")[1].strip()
                    value = float(value_str.rstrip("%"))
                    values.append(
                        value / 100.0
                    )  # Convert percentage to float between 0 and 1
                except (ValueError, IndexError):
                    continue

        return values
