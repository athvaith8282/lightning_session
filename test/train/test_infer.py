import pytest 
import hydra

from src.infer import main

def test_dogbreed_ex_infering(infer_cfg, tmp_path):

    infer_cfg.paths.output_dir = str(tmp_path)
    infer_cfg.paths.log_dir = str(tmp_path / "logs")

    # Run training
    main(infer_cfg)