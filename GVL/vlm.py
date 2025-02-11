from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict
from PIL import Image
from .helpers import ImageData
from .clients import GeminiClient, ClaudeClient, OpenAIClient


@dataclass
class PromptComponents:
    """Components of the prompt that will be assembled in order."""

    introduction: str = """
You are an expert roboticist tasked to predict task completion percentages for frames of a robot for the task of '{task_description}'. 
The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. 
In the initial robot scene, the task completion percentage is 0.
    """

    teacher_header: str = """
Here are some references of an ideal success video for the task.
Reference Video {video_num}:
    """

    inference_header: str = """
Now, for the task of '{task_description}', output the task completion percentage for the following frames. 
Compare each frame to the corrponding frame in the success scenario and provide the task completion percentage for each frame. 
Base your completion percentage based on task completion in comparison to the success scenario.
For each frame, format your response as follow:
Frame: [i] Description: [Description of evaluation of the progress of the robot on the task]: Task Completion Percentages: []%
    """


class BaseVLM(ABC):
    """Base class for Vision Language Models."""

    def __init__(self, components: Optional[PromptComponents] = None):
        """Initialize the VLM with a prompt template.

        Args:
            components: Custom prompt components. If None, uses default.
        """
        self.components = components or PromptComponents()
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """Initialize the appropriate client for this VLM."""
        pass

    @abstractmethod
    def create_message_part(self, content: Union[str, ImageData]) -> Any:
        """Create a message part in the format required by the specific VLM API."""
        pass

    def prepare_request(
        self,
        task_desc: str,
        inference_frames: List[ImageData],
        teacher_examples: Optional[List[List[ImageData]]] = None,
    ) -> List[Any]:
        """Prepare the interleaved request for the VLM API."""
        if not task_desc.strip():
            raise ValueError("Task description cannot be empty")
        if not inference_frames:
            raise ValueError("Inference frames cannot be empty")

        # Start with introduction
        message_parts = [
            self.create_message_part(
                self.components.introduction.format(task_description=task_desc)
            )
        ]

        # Add teacher examples with their frames
        if teacher_examples:
            for idx, example in enumerate(teacher_examples, 1):
                # Add teacher header
                message_parts.append(
                    self.create_message_part(
                        self.components.teacher_header.format(video_num=idx)
                    )
                )
                # Add teacher frames
                for ix, frame in enumerate(example):
                    message_parts.append(f"Frame: {ix + 1}")
                    message_parts.append(self.create_message_part(frame))

        # Add inference header and frames
        message_parts.append(
            self.create_message_part(
                self.components.inference_header.format(task_description=task_desc)
            )
        )
        for ix, frame in enumerate(inference_frames):
            message_parts.append(f"Frame: {ix + 1}")
            message_parts.append(self.create_message_part(frame))

        return message_parts


class GeminiVLM(BaseVLM):
    """Gemini implementation of VLM."""

    def _initialize_client(self):
        """Automatically initialize Gemini client."""
        return GeminiClient()

    def create_message_part(
        self, content: Union[str, ImageData]
    ) -> Union[str, Image.Image]:
        """Convert content to Gemini-compatible format."""
        if isinstance(content, str):
            return content
        return content.image  # Gemini accepts PIL.Image directly

    def generate(
        self,
        task_desc: str,
        inference_frames: List[ImageData],
        teacher_examples: Optional[List[List[ImageData]]] = None,
    ):
        contents = self.prepare_request(task_desc, inference_frames, teacher_examples)
        response = self.client.client.models.generate_content(
            model="gemini-2.0-flash-exp", contents=contents
        )
        return response


class ClaudeVLM(BaseVLM):
    """Claude implementation of VLM."""

    def _initialize_client(self):
        """Automatically initialize Claude client."""
        return ClaudeClient()

    def create_message_part(self, content: Union[str, ImageData]) -> Dict[str, Any]:
        """Convert content to Claude-compatible format."""
        if isinstance(content, str):
            return {"type": "text", "text": content}
        return {"type": "image", "source": {"type": "base64", "data": content.encoding}}

    def generate(
        self,
        task_desc: str,
        inference_frames: List[ImageData],
        teacher_examples: Optional[List[List[ImageData]]] = None,
    ):
        message_parts = self.prepare_request(
            task_desc, inference_frames, teacher_examples
        )
        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": message_parts}],
        }
        response = self.client.client.messages.create(**request)
        return response
