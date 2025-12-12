import hydra
from train import train
from evaluate import evaluate
from bayesadapt.utils import infer_logdir_from_cfg

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    cfg.logdir = infer_logdir_from_cfg(cfg)
    model = train(cfg)
    evaluate(cfg, model=model)
    
if __name__ == "__main__":
    main()
