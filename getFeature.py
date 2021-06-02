import torch
import os
from typhoon_dataset import TyphoonDataset
import torch.nn.functional as F
from sknet_self_typhoon import SKNet50_Pure, SKNet50
from resnext import ResNeXt50
from senet import SENet50
import numpy as np
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    parser.add_argument('--res', '-r',type=str, default='../FuseforTyphoon/dataset/', help='result folder.')
    args = parser.parse_args()

    dataset = TyphoonDataset(mode="train")
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=False)

    net = SKNet50()
    net.load_state_dict(torch.load(args.load))
    net.cuda()

    def test():
        net.eval()
        features = None
        for step, (img, _, y) in enumerate(tqdm(test_loader)):
            img, y = img.cuda(), y.cuda()

            # forward
            _, feature = net(img)
            feature = feature.cpu().detach().numpy()
            if features is None:
                features = feature
            else:
                features = np.concatenate([features, feature], axis=0)
        np.save(args.res+"sknet_train.npy", features)
    
    test()
    

    
