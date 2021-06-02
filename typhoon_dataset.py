'''
实现对台风图像和轨迹数据的读取
包括训练集和测试集
'''
import os
import numpy as np
from torch.utils.data import Dataset, dataset
import torch

class TyphoonDataset(Dataset):
    def __init__(self, mode="train", root_path="../TyphoonDatasets"):
        super(TyphoonDataset, self).__init__()
        if mode == "train":
            img_file_path = os.path.join(root_path, "img_train.npy")
            trace_file_path = os.path.join(root_path, "input_trace_train.npy")
            out_file_path = os.path.join(root_path, "output_trace_train.npy")
            self.img, self.trace, self.output = np.load(img_file_path), np.load(trace_file_path), np.load(out_file_path)
        elif mode == "test":
            img_file_path = os.path.join(root_path, "img_test.npy")
            trace_file_path = os.path.join(root_path, "input_trace_test.npy")
            out_file_path = os.path.join(root_path, "output_trace_test.npy")
            self.img, self.trace, self.output = np.load(img_file_path), np.load(trace_file_path), np.load(out_file_path)
        self.num = self.output.shape[0]
        self.img = self.img.transpose((0,3,1,2)) # N H W C -> N C H W
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.img[index, :, :].astype(np.float32), self.trace[index, :].astype(np.float32), self.output[index, :].astype(np.float32)


if __name__ == "__main__":
    dataset = TyphoonDataset(mode="train")
    print(dataset[0][0].shape)
    #train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    # for (img, _, y) in enumerate(train_loader):
    pass
