from pathlib import Path

import wandb

from omegaconf import OmegaConf

class WandBLogger:

    def __init__(self, cfg):

        self.cfg = cfg
        self.run = None
    

    def start(self):

        self.run = wandb.init(
            project = self.cfg.wandb.project,
            entity = self.cfg.wandb.entity,
            name= self.cfg.wandb.run_name,
            tags= self.cfg.wandb.tags,
            group= self.cfg.wandb.group,
            notes= self.cfg.wandb.notes,
            mode=self.cfg.wandb.mode,
            config=OmegaConf.to_container(
                self.cfg,
                resolve=True,
            ),
        )

        return self.run
    

    def log_metrics(self, metrics: dict, step: int | None = None,):
        
        wandb.log(metrics, step= step)


    def log_dataframe(self, name:str, dataframe,):
        
        wandb.log(
            {
                name: wandb.Table(
                    dataframe= dataframe
                )
            }
        )
    

    def log_image(self, name:str, image_path,):

        wandb.log(
            {
                name: wandb.Image(
                    str(image_path)
                )
            }
        )


    def log_directory(self, directory):
        
        directory = Path(directory)

        if not directory.exists():
            return
        
        for file in directory.glob("*.png"):
            self.log_image(file.stem, file)


    def log_model(self, model_path, artifact_name= "model",):

        artifact = wandb.Artifact(artifact_name, type="model",)
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)


    def save_file(self, path,):

        wandb.save(str(path))


    def finish(self):

        wandb.finish()