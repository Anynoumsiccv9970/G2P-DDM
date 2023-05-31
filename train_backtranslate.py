
from asyncio.log import logger
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse
import torch
from backmodels.point2text_model import BackTranslateModel
# from backmodels.point2text_model_2 import BackTranslateModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from data_phoneix.phoneix_text2pose_img_data_shift import PhoenixPoseData, PoseDataset
from data_phoneix.phonex_data import PhoenixPoseData
from util.util import CheckpointEveryNSteps
import os
from pytorch_lightning.loggers import NeptuneLogger
from data.vocabulary import Dictionary


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = BackTranslateModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    # print(opt)
    # print("opt.gpu_ids: ", opt.gpu_ids, type(opt.gpu_ids))
    # exit()

    # opt.world_size = int(os.environ['WORLD_SIZE'])
    # opt.local_rank = int(os.environ['LOCAL_RANK'])
    # opt.rank = int(os.environ['RANK'])
    # print(opt.local_rank, opt.rank)

    data = PhoenixPoseData(opt)
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)

    model = BackTranslateModel(opt, text_dict)
    model = model.load_from_checkpoint("experiments/backmodel/lightning_logs/small_v0/checkpoints/epoch=21-step=19514-val_wer=0.5195.ckpt")    
    
    callbacks = []
    model_save_ccallback = ModelCheckpoint(monitor="val_wer", filename='{epoch}-{step}-{val_wer:.4f}', save_top_k=-1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_wer", min_delta=0.00, patience=20, verbose=False, mode="min")
    callbacks.append(model_save_ccallback)
    callbacks.append(early_stop_callback)

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(accelerator='cuda', gpus=opt.gpus, strategy="ddp")
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=2000000, **kwargs)

    # print(torch.distributed.get_rank())
    # trainer.fit(model, data)
    trainer.validate(model, dataloaders=data.test_dataloader())
    trainer.validate(model, dataloaders=data.val_dataloader())


if __name__ == "__main__":
    main()