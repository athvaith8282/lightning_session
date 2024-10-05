import pytest 
import hydra

from src.eval import main

def test_dogbreed_ex_testing(test_cfg, tmp_path):

    test_cfg.paths.output_dir = str(tmp_path)
    test_cfg.paths.log_dir = str(tmp_path / "logs")

    # Run training
    main(test_cfg)