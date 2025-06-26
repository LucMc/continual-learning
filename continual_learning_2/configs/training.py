from dataclasses import dataclass


@dataclass
class TrainingConfig:
    resume: bool = False
    resume_from_step: int = -1


@dataclass
class RLTrainingConfig:
    resume: bool = False
    resume_from_step: int = -1

    num_steps: int = 100_000
    steps_per_task: int = 1_000_000
