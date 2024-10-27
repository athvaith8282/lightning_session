import hydra
import pytest

from src.train import main

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=dogbreed_ex"],
        )
        return cfg


# def test_dogbreed_ex_training(config, tmp_path):
#     # Update output and log directories to use temporary path
#     config.paths.output_dir = str(tmp_path)
#     config.paths.log_dir = str(tmp_path / "logs")

#     # Instantiate components
#     datamodule = hydra.utils.instantiate(config.data)
#     model = hydra.utils.instantiate(config.model)
#     trainer = hydra.utils.instantiate(config.trainer)

#     # Run training
#     train(trainer, model, datamodule)

def test_dogbreed_ex_main(config, tmp_path):

    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    # Run training
    main(config)
