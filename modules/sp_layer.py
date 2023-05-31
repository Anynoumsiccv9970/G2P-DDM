from multiprocessing.sharedctypes import Value
import torch
from torch import nn


H36M_SKELETON = [
    [(-1, 0, "Hips")],
    [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
    [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
    [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"), (10, 13, "LeftShoulder"), (10, 11, "Neck")],
    [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"), (11, 12, "Head")],
    [(18, 19, "RightForeArm"), (14, 15, "LeftForeArm")],
    [(19, 20, "RightHand"), (15, 16, "LeftHand")]
]

NTU_SKELETON = [
    [(-1, 0, "SpineBase")],
    [(0, 1, "SpineMid"), (0, 12, "HipLeft"), (0, 16, "HipRight")],
    [(1, 20, "SpineShoulder"), (12, 13, "KneeLeft"), (16, 17, "KneeRight")],
    [(1, 2, "Neck"), (1, 4, "ShoulderLeft"), (1, 8, "ShoulderRight"), (13, 14, "AnkleLeft"), (17, 18, "AnkleRight")],
    [(2, 3, "Head"), (4, 5, "ElbowLeft"), (8, 9, "ElbowRight"), (14, 15, "FootLeft"), (18, 19, "FootRight")],
    [(5, 6, "WristLeft"), (9, 10, "WristRight")],
    [(6, 7, "HandLeft"), (6, 22, "ThumbLeft"), (10, 11, "HandRight"), (10, 24, "THumbRight")],
    [(7, 21, "HandTipLeft"), (11, 23, "HandTipRight")]
]

"""
pose:  [0,1,2,3,4,5,6,7]
rhand: [1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
rhand: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

lhand: [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
lhand: [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
"""

SIGN_SKELETON = [
    [(-1, 0, "neck")],
    [(0, 1, "head"), (0, 5, "LeftUpArm"), (0, 2, "RightUpArm")],
    [(2, 3, "RightElbow"), (5, 6, "LeftElbow")],
    [(3, 4, "RightWrist"), (6, 7, "LeftWrist")],
    [(4, 8, "rhand1"), (4, 12, "rhand5"), (4, 16, "rhand9"), (4, 20, "rhand13"), (4, 24, "rhand17"), (7, 28, "lhand1"), (7, 32, "lhand5"), (7, 36, "lhand9"), (7, 40, "lhand13"), (7, 44, "lhand17"),],
    [(8, 9, "rhand2"), (12, 13, "rhand6"), (16, 17, "rhand10"), (20, 21, "rhand14"), (24, 25, "rhand18"), (28, 29, "lhand2"), (32, 33, "lhand6"), (36, 37, "lhand10"), (40, 41, "lhand14"), (44, 45, "lhand18")],
    [(9, 10, "rhand3"), (13, 14, "rhand7"), (17, 18, "rhand11"), (21, 22, "rhand15"), (25, 26, "rhand19"), (29, 30, "lhand3"), (33, 34, "lhand7"), (37, 38, "lhand11"), (41, 42, "lhand15"), (45, 46, "lhand19")],
    [(10, 11, "rhand4"), (14, 15, "rhand8"), (18, 19, "rhand12"), (22, 23, "rhand16"), (26, 27, "rhand20"), (30, 31, "lhand4"), (34, 35, "lhand8"), (38, 39, "lhand12"), (42, 43, "lhand16"), (46, 47, "lhand20")],
    
]

SIGN_POSE_SKELETON = [
    [(-1, 0, "neck")],
    [(0, 1, "head"), (0, 5, "LeftUpArm"), (0, 2, "RightUpArm")],
    [(2, 3, "RightElbow"), (5, 6, "LeftElbow")],
    [(3, 4, "RightWrist"), (6, 7, "LeftWrist")]]

SIGN_HAND_SKELETON = [
    [(-1, 0, "Wrist")],
    [(0, 1, "hand1"), (0, 5, "hand5"), (0, 9, "hand9"), (0, 13, "hand13"), (0, 17, "hand17")],
    [(1, 2, "hand2"), (5, 6, "hand6"), (9, 10, "hand10"), (13, 14, "hand14"), (17, 18, "hand18")],
    [(2, 3, "hand3"), (6, 7, "hand7"), (10, 11, "hand11"), (14, 15, "hand15"), (18, 19, "hand19")],
    [(3, 4, "hand4"), (7, 8, "hand8"), (11, 12, "hand12"), (15, 16, "hand16"), (19, 20, "hand20")],
]

class SPL(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, joint_size, reuse, sparse=False, SKELETON = None):
        super().__init__()
        self.input_size = input_size
        self.L_num = hidden_layers
        self.hid_size = hidden_units
        self.reuse = reuse
        self.sparse_spl = sparse


        if SKELETON == "sign_pose":
            self.skeleton = SIGN_POSE_SKELETON  # 选择数据集
            self.num_joints = 8  # 关节点数
        elif SKELETON == "sign_hand":
            self.skeleton = SIGN_HAND_SKELETON   # 选择数据集
            self.num_joints = 21          # 关节点数
        elif SKELETON == "sign_pose_hand":
            self.skeleton = SIGN_SKELETON
            self.num_joints = 48
        else:
            raise ValueError("{} is not existed!".format(SKELETON))
            

        self.joint_size = joint_size
        self.human_size = self.num_joints * self.joint_size

        kinematic_tree = dict()
        for joint_list in self.skeleton:  # 实现一个节点的层级结构
            for joint_entry in joint_list:
                parent_list_ = [joint_entry[0]] if joint_entry[0] > -1 else []
                kinematic_tree[joint_entry[1]] = [parent_list_, joint_entry[1], joint_entry[2]]
        
        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)

        self.prediction_order = list()
        self.indexed_skeleton = dict()

        # Reorder the structure so that we can access joint information by using its index.
        self.prediction_order = list(range(len(kinematic_tree)))  # 比起kinematic_tree，丰富了层级结构
        for joint_id in self.prediction_order:
            joint_entry = kinematic_tree[joint_id]
            if self.sparse_spl:
                new_entry = joint_entry
            else:
                parent_list_ = list()
                if len(joint_entry[0]) > 0:
                    get_all_parents(parent_list_, joint_entry[0][0], kinematic_tree)
                
                new_entry = [parent_list_, joint_entry[1], joint_entry[2]]
            self.indexed_skeleton[joint_id] = new_entry
        
        # print("indexed_skeleton: ", self.indexed_skeleton)
        
        self.joint_predictions = nn.ModuleList()

        for joint_key in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]

            input_size = self.input_size
            input_size += self.joint_size*len(parent_joint_ids)
            self.joint_predictions.append(SP_block(input_size, self.hid_size, self.joint_size, self.L_num))

    def forward(self, x):
        out = dict()
        for joint_key in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]
            if len(parent_joint_ids) > 0:
                xinput = [x]
                for i in parent_joint_ids:
                    xinput.append(out[i])
                xinput = torch.cat(xinput, dim=-1)
            else:
                xinput = x
            out[joint_key] = self.joint_predictions[joint_key](xinput)
        
        prediction = torch.cat(list(out.values()), dim= -1)
        return prediction


class SP_block(nn.Module):
    def __init__(self, input_size, hid_size, out_size, L_num):
        super().__init__()
        self.in_size = input_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.L_num = L_num

        self.Layer = nn.ModuleList()
        self.Relu = nn.ModuleList()
        self.inLayer = nn.Linear(input_size, hid_size)
        for i in range(L_num):
            self.Layer.append(nn.Linear(hid_size, hid_size))
            self.Relu.append(nn.ReLU())
        self.outLayer = nn.Linear(hid_size, out_size)

    def forward(self, x):
        h0 = self.inLayer(x)
        for i in range(self.L_num):
            h0 = self.Layer[i](h0)
            h0 = self.Relu[i](h0)
        out = self.outLayer(h0)
        return out


if __name__ == "__main__":
    model = SPL(256, 10, 128, 2, True, H36M = False)
    x = torch.randn(5, 256)
    out = model(x)
    print(out.shape)
    print(model.indexed_skeleton)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)