

from data_phoneix.phoneix_text2pose_img_data_shift import PhoenixPoseData
from data_phoneix.stage2_phoneix_data import PhoenixPoseData as PhoenixPoseData2


if __name__ == "__main__":
    pass
    csv_path = "Data"
    data_path = "Data/ProgressiveTransformersSLP/"
    vocab_file = data_path + "src_vocab.txt"
    class Option:
        vocab_file = vocab_file
        hand_generator = True
        resolution = 256
        csv_path=csv_path
        data_path=data_path
        batchSize=2
        num_workers=2
        sequence_length=8
        debug = 100
        max_frames_num = 300

    opts= Option()

    dataloader = PhoenixPoseData(opts).val_dataloader()
    dataloader2 = PhoenixPoseData2(opts).val_dataloader()

    for data1, data2 in zip(dataloader, dataloader2):
        print(data1.keys())
        print(data1["gloss"])
        print(data1["gloss_id"])
        print(data1["skel_len"])
        print(data1["skel_3d"].shape)
        print(data2.keys())
        print(data2["gloss"])
        print(data2["gloss_id"])
        print(data2["skel_len"])
        print(data2["skel_3d"].shape)
        # exit()
