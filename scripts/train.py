import os
import hydra
from omegaconf import OmegaConf
from text2cypher.finetuning.train import train


def main():
    env = os.environ.get("ENV", "dev")
    config_name = f"config.{env}"
    config_path = "src/text2cypher/finetuning/config"  # absolute path inside container

    # Dynamically run hydra.compose instead of @hydra.main
    with hydra.initialize_config_dir(config_dir=config_path):
        cfg = hydra.compose(config_name=config_name)

    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
