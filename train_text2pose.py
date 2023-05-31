
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

# from models_phoneix.point2text_model_vqvae_tr_latent_dm_stage2 import Point2textModelStage2
# from stage2_models.mask_predict import Point2textModelStage2
# from stage2_models.cond_discrete_dm import Point2textModelStage2
# from stage2_models.efficient_cond_discrete_dm import Point2textModelStage2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_phoneix.stage2_phoneix_data import PhoenixPoseData
from util.util import CheckpointEveryNSteps
from util.train_utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy

def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2_model', type=str, help='stage2 mode config')
    # parser = Point2textModelStage2.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()

    # data = How2SignTextPoseData(opt)
    config = OmegaConf.load(opt.stage2_model)

    data = instantiate_from_config(config.data)
    data.train_dataloader()
    data.test_dataloader()

    model = instantiate_from_config(config.model)
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print({'Total': total_num, 'Trainable': trainable_num})
    get_parameter_number(model)
    
    callbacks = []
    model_save_ccallback = ModelCheckpoint(monitor="val/rec_wer", filename='{epoch}-{step}-{val/rec_wer:4f}', save_top_k=10, mode="min")
    # early_stop_callback = EarlyStopping(monitor="test_wer", min_delta=0.00, patience=100, verbose=False, mode="min")
    callbacks.append(model_save_ccallback)
    # callbacks.append(early_stop_callback)

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(
            check_val_every_n_epoch=2,
            accelerator='cuda', 
            gpus=opt.gpus, 
            strategy=DDPStrategy(find_unused_parameters=False))
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.validate(model, dataloaders=data.val_dataloader())
    # trainer.validate(model, dataloaders=data.test_dataloader())
    # trainer.fit(model, data)


if __name__ == "__main__":
    main()