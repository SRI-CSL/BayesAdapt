import hydra
from train import train
from evaluate import evaluate
from bayesadapt.utils import infer_logdir_from_cfg

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    cfg.logdir = infer_logdir_from_cfg(cfg)
    train(cfg)
    evaluate(cfg)
    
if __name__ == "__main__":
    main()
