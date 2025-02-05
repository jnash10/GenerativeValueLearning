# GenerativeValueLearning
# Project: Value Generator Library

## Overview
The Value Generator Library is designed to extract and evaluate frames from a video using a vision-language model (VLM). It provides a structured interface for adding teacher examples, generating values for video frames, and utilizing helper functions for image and video processing.

## Architecture
The library is structured into three main components:
1. **User Facing** (ValueGenerator)
2. **Implementation** (Prompt, VLM)
3. **Helpers** (Utility functions)

---

## 1. User Facing: ValueGenerator
The `ValueGenerator` class serves as the primary interface for users.

### Methods:
- `__init__(task_description: str, n_samples: int = 10)`: Initializes the value generator.
- `add_teacher_example(video_path: str)`: Adds teacher examples for comparison.
- `generate_value(video_path: str) -> List[DataClass]`: Generates a value for each frame in the video.

### DataClass Structure:
- `frame_no: int` - Frame number in the video.
- `frame: PIL.Image` - Extracted frame.
- `description: str` - Description of the frame.
- `value: float` - Computed value for the frame.

---

## 2. Implementation
### Prompt Class
Handles the formatting of prompts for the VLM.
- `__init__()`: Initializes the prompt with formatters.
- `format(task_desc: str, inference_frames: List[PIL.Image], teacher_frames: List[List[PIL.Image]]) -> str`: Formats the base prompt for processing.

### VLM Class
Responsible for interaction with the vision-language model.
- `__init__()`: Initializes the VLM instance.
- `format_prompt(task_desc: str, inference_frames: List[PIL.Image], teacher_frames: List[List[PIL.Image]]) -> str`: Prepares the formatted prompt.
- `call_VLM() -> List[float]`: Calls the VLM model and returns computed values.

---

## 3. Helpers
Utility functions for image and video processing.
- `img_to_base64(img: PIL.Image) -> str`: Converts an image to a base64-encoded string.
- `video_to_frames(video_path: str, n_frames: int) -> List[PIL.Image]`: Extracts frames from a video file.

---

## Usage Example
```python
vg = ValueGenerator("Evaluate motion clarity", n_samples=5)
vg.add_teacher_example("teacher_video.mp4")

