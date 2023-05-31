
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

# from stage1_models.pose_vqvae import PoseVQVAE
from stage1_models.pose_vqvae_sep import PoseVQVAE
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_phoneix.stage1_phoneix_data import PhoenixPoseData, PoseDataset
from util.util import CheckpointEveryNSteps
import os
from data.vocabulary import Dictionary


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = PoseVQVAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()

    data = PhoenixPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)

    model = PoseVQVAE(opt, text_dict)

    if os.path.exists(opt.resume_ckpt):
        print("=== Load from {}!".format(opt.resume_ckpt))
        model = model.load_from_checkpoint(opt.resume_ckpt, strict=True)
    else:
        print("=== {} is not existed!".format(opt.resume_ckpt))

    callbacks = []
    model_save_ccallback = ModelCheckpoint(monitor="val_rec_loss", filename='{epoch}-{step}-{val_wer:.4f}-{val_rec_loss:.4f}-{val_ce_loss:.4f}', save_top_k=10, mode="min")
    # early_stop_callback = EarlyStopping(monitor="val_rec_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    callbacks.append(model_save_ccallback)
    # callbacks.append(early_stop_callback)

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(accelerator='cuda', gpus=opt.gpus, strategy="ddp")
    trainer = pl.Trainer.from_argparse_args(
        opt, callbacks=callbacks, 
        max_steps=200000000, **kwargs)
    # trainer.validate(model, dataloaders=data.test_dataloader())
    trainer.fit(model, data)


if __name__ == "__main__":
    main()