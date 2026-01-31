import os
import hydra
from hydra.utils import instantiate

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    trainer = instantiate(cfg.trainer, cfg=cfg)
    if not os.path.exists(f"{trainer.evaldir}/metrics.json") or cfg.overwrite:
        trainer.evaluate()

if __name__ == "__main__":
    main()
