"""This file contains helper functions for the GVL project."""


class Prompt:
    """This class contains the prompt for the GVL project."""

    def __init__(self):
        """Initialise the base prompt with variable formatters for:
        task_desc, teacher examples, inference video
        All inputs are strings. videos must be base64 encoded as strings."""

        self.system_and_task_desc_prompt = """You are an expert robotics engineer and are training an RL model for which you are labelling data.
        The task you are training the model for is: {task_description}. You need to create ground truth labels for a video of a robot performing the task.
        You are given multiple frames from a video of a robot performing the task. For each frame, you need to predict the task completion percentage, which represents how close the robot is to completing the task.
        The task completion percentages are between 0 and 100, where 100 corresponds to full task completion. In the initial robot scene, the task completion percentage is 0.
        """

        self.teacher_examples_prompt = """
        Here are some references of of an ideal success video for the task. Remember, these are ideal cases and progress will be monotonic, but for the inference video, progress might not be monotonic as failure cases can be there.
        Reference videos:
        """

        self.inference_video_prompt = """
        Now, for the task of '{task_description}', output the task completion percentage for each frame. 
        {teacher_reminder_prompt}
        For each frame, format your response as follow: 
        Frame: [i] Description: [Desciption of evaluation of the progress of the robot on the task]: Task Completion Percentages:[]%
        for example:
        Frame: [1]  Description: [Desciption of evaluation of the progress of the robot on the task]: Task Completion Percentages:10%
        Frame: [2] Frame Description: [Desciption of evaluation of the progress of the robot on the task]: Task Completion Percentages:20%
        ...

        Inference video:
        """

        self.teacher_reminder_prompt = """Compare each frame to the corresponding frame(position wise) in the ideal success scenarios and make a judgement on the task completion percentage."""

        self.final_prompt = """"""

    def format_prompt(
        self,
        task_desc: str,
        inference_video: list[str],
        teacher_examples: list[list[str]] = None,
    ) -> str:  # type: ignore
        """Format the prompt for the given task description, teacher examples and inference video."""

        self.final_prompt = self.system_and_task_desc_prompt.format(
            task_description=task_desc
        )

        if teacher_examples:
            self.final_prompt += self.teacher_examples_prompt

            for ix, example in enumerate(teacher_examples):
                self.final_prompt += f"Reference Video {ix + 1}:\n"
                for frame in example:
                    self.final_prompt += f"    Frame {ix + 1}: {frame}\n"

        self.final_prompt += self.inference_video_prompt.format(
            task_description=task_desc,
            teacher_reminder_prompt=self.teacher_reminder_prompt
            if teacher_examples
            else "",
        )

        self.final_prompt += self.teacher_reminder_prompt

        return self.final_prompt
