from dataclasses import dataclass


@dataclass
class TrainingConfig:
    resume: bool = False
    resume_from_step: int = -1


@dataclass
class RLTrainingConfig:
    resume: bool = False
    resume_from_step: int = -1

    steps_per_task: int = 2_000_000
