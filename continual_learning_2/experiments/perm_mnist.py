# experiments/perm_mnist_modal.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional, List
import time

import modal

# --- Modal image -------------------------------------------------------------
# OS libs help mujoco / gymnasium[mujoco] & friends even if you don't use them in this run.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "build-essential",
        "libgl1",          # OpenGL runtime
        "libglfw3",        # GLFW for headless mujoco
        "libosmesa6",      # OSMesa
        "libglew-dev",
        "libglib2.0-0",
        "patchelf",
    )
    # Make your local code importable inside the container
    .add_local_python_source("continual_learning")
    .add_local_python_source("continual_learning_2")
    # Install deps declared in pyproject.toml (plus the extras you want)
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cuda12"])
    # A couple of quality-of-life env vars for JAX
    .env({
        # don't grab 100% of GPU memory on startup
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    })
)

# Optional: keep datasets / artifacts across runs
results_vol = modal.Volume.from_name("crl-results", create_if_missing=True)
# HuggingFace datasets cache inside the volume (if you use them)
image = image.env({"HF_HOME": "/data/hf"})

# Pass your W&B API key via a Modal Secret you create in the dashboard:
#  Name: "wandb", contains key "WANDB_API_KEY"
wandb_secret = modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])

app = modal.App("perm_mnist")

# If you import heavyweight libs at module scope locally, you'll get ImportError.
# This context tells Modal to defer those imports to the remote container.
with image.imports():  # :contentReference[oaicite:6]{index=6}
    import jax
    from chex import dataclass as chex_dataclass  # (unused here but available)
    from continual_learning_2.trainers.continual_supervised_learning import (
        HeadResetClassificationCSLTrainer,
    )
    from continual_learning_2.configs.models import MLPConfig
    from continual_learning_2.configs import (
        AdamConfig,
        AdamwConfig,
        MuonConfig,
        CbpConfig,
        RedoConfig,
        RegramaConfig,
        CcbpConfig,
        ShrinkAndPerterbConfig,
        DatasetConfig,
        LoggingConfig,
        TrainingConfig,
    )


@dataclass(frozen=True)
class Args:
    seed: int = 42
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: Optional[str] = "crl_experiments"
    wandb_entity: Optional[str] = "lucmc"
    resume: bool = False
    exclude: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    postfix: Optional[str] = None
    base_optim: Literal["adam", "adamw", "muon"] = "adam"


@app.function(
    image=image,
    gpu=modal.gpu.A10G(),          # pick A10G by default; swap to A100/H100/L4 as needed. :contentReference[oaicite:7]{index=7}
    timeout=60 * 60 * 12,          # 12h max run; adjust as you like
    secrets=[wandb_secret],        # makes WANDB_API_KEY available
    volumes={"/data": results_vol} # persist datasets & logs under /data
)
def run_all_perm_mnist_remote(
    seed: int = 42,
    wandb_mode: str = "online",
    wandb_project: Optional[str] = "crl_experiments",
    wandb_entity: Optional[str] = "lucmc",
    resume: bool = False,
    exclude: List[str] = (),
    include: List[str] = (),
    postfix: Optional[str] = None,
    base_optim: str = "adam",
):
    # Recreate your dataclass instance (so logic below is unchanged)
    args = Args(
        seed=seed,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        resume=resume,
        exclude=list(exclude),
        include=list(include),
        postfix=postfix,
        base_optim=base_optim,
    )

    if args.wandb_mode != "disabled":
        assert args.wandb_project is not None
        assert args.wandb_entity is not None

    base_optimizers = {
        "adam": AdamConfig(learning_rate=1e-3),
        "muon": MuonConfig(learning_rate=1e-3),
        "adamw": AdamwConfig(learning_rate=1e-3),
    }
    base_optim = base_optimizers[args.base_optim]

    optimizers = {
        "standard": base_optim,
        "regrama": RegramaConfig(
            tx=base_optim,
            update_frequency=1000,
            score_threshold=0.25,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "ccbp": CcbpConfig(
            tx=base_optim,
            seed=args.seed,
            decay_rate=0.99,
            replacement_rate=0.01,
            update_frequency=100,
        ),
        "redo": RedoConfig(
            tx=base_optim,
            update_frequency=1000,
            score_threshold=0.5,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "cbp": CbpConfig(
            tx=base_optim,
            decay_rate=0.99,
            replacement_rate=1e-5,
            maturity_threshold=100,
            seed=args.seed,
            weight_init_fn=jax.nn.initializers.he_uniform(),
        ),
        "shrink_and_perturb": ShrinkAndPerterbConfig(
            tx=base_optim,
            param_noise_fn=jax.nn.initializers.he_uniform(),
            seed=args.seed,
            shrink=1 - 1e-5,
            perturb=1e-5,
            every_n=1,
        ),
    }

    if args.include:
        optimizers = {n: c for n, c in optimizers.items() if n in args.include}
    for algorithm in args.exclude:
        optimizers.pop(algorithm, None)

    print(f"Running algorithms: {list(optimizers.keys())}")

    exp_start = time.time()
    for opt_name, opt_conf in optimizers.items():
        print(f"Config: {opt_conf}")
        run_name = f"{opt_name}_{args.seed}"
        if args.postfix:
            run_name += f"_{args.postfix}"

        start = time.time()
        trainer = HeadResetClassificationCSLTrainer(
            seed=args.seed,
            model_config=MLPConfig(output_size=10, hidden_size=128),
            optim_cfg=opt_conf,
            data_cfg=DatasetConfig(
                name="permuted_mnist",
                seed=args.seed,
                batch_size=8,
                num_tasks=150,
                num_epochs_per_task=1,
                num_workers=0,
            ),
            train_cfg=TrainingConfig(resume=args.resume),
            logs_cfg=LoggingConfig(
                run_name=run_name,
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                group="perm_mnist",
                wandb_mode=args.wandb_mode,
                interval=100,
                eval_during_training=True,
            ),
        )
        trainer.train()
        print(f"Training time: {time.time() - start:.2f} seconds")
        del trainer
    print(f"Total training time: {time.time() - exp_start:.2f} seconds")


# A local entrypoint lets `modal run` parse CLI and forward to the remote function.
@app.local_entrypoint()
def main():
    import tyro
    args = tyro.cli(Args)
    run_all_perm_mnist_remote.remote(**dataclasses.asdict(args))

