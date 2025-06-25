from dataclasses import dataclass


@dataclass
class TrainingConfig:
    resume: bool = False
    resume_from_step: int = -1
