import hydra
from omegaconf import DictConfig
from mitocheck_trainer import run

@hydra.main(version_base=None, config_path='../config', config_name='mitocheck')
def main(cfg: DictConfig):
    run(cfg)

if __name__=='__main__':main()
