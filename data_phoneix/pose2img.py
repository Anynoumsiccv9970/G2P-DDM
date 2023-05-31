

import numpy as np
from util.plot_videos import draw_frame_2D
import cv2
import os
from tqdm import tqdm


file_path = "Data/ProgressiveTransformersSLP/test.files"
skel_path = "Data/ProgressiveTransformersSLP/test.skels"

with open(file_path, "r") as f1, open(skel_path, "r") as f2:
    files = f1.readlines()
    skels = f2.readlines()

    assert len(files) == len(skels)

    for file, skel in tqdm(zip(files, skels)):
        skels_3d = np.array([float(s) for s in skel.strip().split()])
        assert len(skels_3d) % 151 == 0
        skel_len = len(skels_3d) // 151
        skels_3d = skels_3d.reshape(skel_len, 151)

        for i in range(skel_len):
            cur_frame = skels_3d[i][:-1]  # [150]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(cur_frame, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d) # [h, w, c]

            save_dir = os.path.join("/Dataset/everybody_sign_now_experiments/images/", file.strip())
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "{}.png".format(str(i).zfill(5)))

            cv2.imwrite("{}".format(save_path), im)
