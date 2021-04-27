import torch
import os
from typhoon_dataset import TyphoonDataset
import torch.nn.functional as F
from sknet_self_typhoon import SKNet50_Pure, SKNet50
import numpy as np
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    parser.add_argument('--res', '-r',type=str, default='./res/res.txt', help='result folder.')
    args = parser.parse_args()

    dataset = TyphoonDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=False)

    net = SKNet50()
    net.load_state_dict(torch.load(args.load))
    net.cuda()

    def test():
        net.eval()
        loss_avg = 0.0
        distance = 0.0
        for step, (img, _, y) in enumerate(test_loader):
            img, y = img.cuda(), y.cuda()

            # forward
            output = net(img)
            with open(args.res,'a') as f:
                np.savetxt(f, output.cpu().detach().numpy(), delimiter=",")
                f.close()
            loss = F.mse_loss(output, y)

            # distance
            length = y.shape[0]
            dealt = torch.pow((output-y), 2)
            del output
            distance += torch.sum(torch.sqrt(torch.sum(dealt.cpu(), dim=1))).item() / length
            # test loss average
            loss_avg += loss.item()
        return loss_avg / len(test_loader), distance / len(test_loader)
    
    loss, dis = test()
    print("loss is : {}, distance is {}".format(loss, dis))
    with open(args.res,'a') as f:
        f.write("\r\n loss is : {}, distance is {}".format(loss, dis))
        f.close()