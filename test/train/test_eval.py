import pytest 
import hydra

from src.eval import main

@pytest.mark.dependency(depends=["test_dogbreed_ex_main"])  # Add this line
def test_dogbreed_ex_testing(test_cfg, tmp_path):

    test_cfg.paths.output_dir = str(tmp_path)
    test_cfg.paths.log_dir = str(tmp_path / "logs")

    # Run training
    main(test_cfg)