import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import init
# from tensorboardX import SummaryWriter
from typhoon_dataset import TyphoonDataset
from resnext import ResNeXt
from sknet_typhoon import SKNet, SKNet50
from torch.optim import lr_scheduler
# from train import train_epoch, test
import argparse
import json



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    # parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    # parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='The Learning Rate.')
    # parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-3, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=16)
    # parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
    #                     help='Decrease learning rate at these epochs.')
    # parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./best/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    # parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    # parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    # parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    # parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./log', help='Log folder.')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # train valid split
    dataset = TyphoonDataset(mode="train")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                                        num_workers=args.prefetch, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_bs, shuffle=False,
                                                                        num_workers=args.prefetch, pin_memory=False)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

     # Init model, criterion, and optimizer
    # net = ResNeXt(10)
    net = SKNet50()

    # weight init
    #define the initial function to init the layer's parameters for the network
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data,0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.01)
            m.bias.data.zero_()

    net.apply(weights_init) #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。  
                #对所有的Conv层都初始化权重. 

    net.cuda()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # criterion = nn.MSELoss().cuda()

#######################TRAIN TEST FUNCTION########################################################################

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for stpe, (img, _, y) in enumerate(train_loader):
            img, y = img.cuda(), y.cuda()

            # forward
            output = net(img)

            # backward
            optimizer.zero_grad()
            loss = F.mse_loss(output, y)
            del output
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg += loss.item()

        state['train_loss'] = loss_avg / len(train_loader)


    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        distance = 0.0
        for step, (img, _, y) in enumerate(valid_loader):
            img, y = img.cuda(), y.cuda()

            # forward
            output = net(img)
            loss = F.mse_loss(output, y)

            # distance
            dealt = torch.pow(output - y, 2)
            del output
            distance += torch.sqrt(torch.sum(torch.sum(dealt.cpu(), 0))).item()/2.0
            # test loss average
            loss_avg += loss.item()

        state['test_loss'] = loss_avg / len(valid_loader)
        state['distance'] = distance / len(valid_loader)

#####################TRAIN TEST FUNCTION################################################################################

    # Main loop
    best_distance = 100
    for epoch in range(args.epochs):
        # current_lr = lr0 / 2**int(epoch/50)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr
        state['learning_rate'] = scheduler.get_last_lr()
        state["epoch"] = epoch
        train()
        test()
        scheduler.step()
        if state["distance"] < best_distance:
            best_distance = state["distance"]
            torch.save(net.state_dict(), os.path.join(args.save, 'model_{}.pytorch'.format(best_distance)))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best distance: {}".format(best_distance))
    
    log.close()
