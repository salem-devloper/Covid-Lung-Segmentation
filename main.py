import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from LungSegmentationDataset import LungSegDataset
import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt

from torch import optim
from tqdm import tqdm
from model.unet import UNet
import time
import copy
from DiceLoss import DiceLoss

def train_and_validate(net,criterion, optimizer, scheduler, dataloader,device,epochs, load_model = None):


    """load checkpoint pt"""
    if load_model:
        print("load model from", load_model)
        # net.load_state_dict(torch.load(load_model))
        checkpoint = torch.load(load_model)
        net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']


    history = {'train':{'epoch':[], 'loss' : [] , 'acc':[]},
               'val'  :{'epoch':[], 'loss' : [] , 'acc':[]}}

    best_acc = 0.92
    best_loss = 10000000000
    start = time.time()
    for epoch in range(epochs):
        if load_model:
            epoch += start_epoch
            epochs += start_epoch
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        # print(f"Epoch {epoch + 1}/{epochs} learning_rate : {scheduler.get_lr()[0]}")

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # set model to training mode
            else:
                print("-" * 10)
                net.eval() # set model to evaluate mode

            running_loss = 0.0
            running_correct = 0
            dataset_size = 0
            """Iterate over data"""
            data_iter = tqdm(enumerate(dataloader[phase]), total=len(dataloader[phase]))
            for batch_idx, sample in data_iter:
                imgs , true_masks = sample['image'],sample['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                # mask_type = torch.float32 if net.n_classes == 1 else torch.long
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # zero the parameter gradients
                optimizer.zero_grad()

                """forward"""
                with torch.set_grad_enabled(phase == 'train'):
                    masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks)
                    running_loss += loss.item()

                    """backward + optimize only if in training phase"""
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()



                """ statistics """
                dataset_size += imgs.size(0)
                running_loss += loss.item() * imgs.size(0)
                pred = torch.sigmoid(masks_pred) > 0.5
                running_correct += (pred == true_masks).float().mean().item() * imgs.size(0)
                running_acc = running_correct/dataset_size
                # if (batch_idx + 1) % 40 == 0:
                    # print(f'Batch {batch_idx}/{len(dataloader[phase])} Loss {loss.item()} Acc {running_acc}')
                    # print(f'Batch {batch_idx+1}/{len(dataloader[phase])} Loss {loss.item()}')

                data_iter.set_description(
                    f' {phase.capitalize()} - Loss: {running_loss/dataset_size:1.4f} Acc: {running_acc:1.4f}')


            """ statistics """
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_correct / dataset_size
            print('{} Loss {:.5f}\n{} Acc {:.2f}'
                  .format(phase, epoch_loss,phase,epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save({
                    'epoch':epoch + 1,
                    'model_state_dict':best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc
                },os.path.join(os.getcwd(),'checkpoint/best_checkpoint[epoch_{}].pt'.format(epoch + 1)))
                print("Achived best result! save checkpoint.")
                print("val acc = ", best_acc)
            history[phase]['epoch'].append(epoch)
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)

        scheduler.step(history['val']['acc'][-1])

        time_elapsed = time.time() - since
        print("One Epoch Complete in {:.0f}m {:.0f}s".format(time_elapsed//60 , time_elapsed%60))

        time_elapsed = time.time() - start
        min, sec = time_elapsed//60 , time_elapsed % 60


        # import json

        # with open('history.json', 'w') as fp:
        #     json.dump(history, fp)

        print("Total Training time {:.0f}min {:.0f}sec".format(min,sec))
    draw_plots(history)

def test(net,criterion,dataloader,device):
    running_loss = 0.0
    running_correct = 0
    dataset_size = 0
    """Iterate over data"""
    data_iter = tqdm(enumerate(dataloader['test']), total=len(dataloader['test']))
    for batch_idx, sample in data_iter:
        imgs, true_masks = sample['image'], sample['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        # mask_type = torch.float32 if net.n_classes == 1 else torch.long
        mask_type = torch.float32
        true_masks = true_masks.to(device=device, dtype=mask_type)

        # zero the parameter gradients
        # optimizer.zero_grad()

        """forward"""
        with torch.set_grad_enabled(False):
            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            running_loss += loss.item()

        """ statistics """
        dataset_size += imgs.size(0)
        running_loss += loss.item() * imgs.size(0)
        pred = torch.sigmoid(masks_pred) > 0.5
        running_correct += (pred == true_masks).float().mean().item() * imgs.size(0)
        running_acc = running_correct / dataset_size
        # if (batch_idx + 1) % 40 == 0:
        # print(f'Batch {batch_idx}/{len(dataloader[phase])} Loss {loss.item()} Acc {running_acc}')
        # print(f'Batch {batch_idx+1}/{len(dataloader[phase])} Loss {loss.item()}')

        data_iter.set_description(
            f' Test: - Loss: {running_loss / dataset_size:1.4f} Acc: {running_acc:1.4f}')
    """ statistics """
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_correct / dataset_size
    print('Test Loss {:.5f}\nTest Acc {:.2f}'
          .format(epoch_loss, epoch_acc))


def draw_plots(history):
    # list all data in history
    #history = {'train': {'epoch': [], 'loss': [], 'acc': []},
    #           'val': {'epoch': [], 'loss': [], 'acc': []}}
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history['train']['acc'])
    plt.plot(history['val']['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_plot.png')
    plt.show()

    # summarize history for loss
    plt.plot(history['train']['loss'])
    plt.plot(history['val']['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_plot.png')
    plt.show()

def get_args():

    parser = argparse.ArgumentParser(description = "U-Net for Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--gpu', type=str, default = '0')
    parser.add_argument('--n_workers', type =int , default = 4 , help = "The number of workers for dataloader")

    # arguments for training
    parser.add_argument('--img_size', type = int , default = 512,)
    parser.add_argument('--epochs', type=int , default = 100 )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--load_model', type=str, default=None, help='.pth file path to load model')
    return parser.parse_args()

def main():
    args = get_args()

    # set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    model = UNet(n_channels=1, n_classes=1).to(device)
    if len(args.gpu) > 1: # if multi-gpu
        model = torch.nn.DataParallel(model)

    """set img size
        - UNet type architecture require input img size be divisible by 2^N,
        - Where N is the number of the Max Pooling layers (in the Vanila UNet N = 5)
    """

    img_size = args.img_size #default: 512



    # set transforms for dataset
    import torchvision.transforms as transforms
    from my_transforms import RandomHorizontalFlip,RandomVerticalFlip,ColorJitter,GrayScale,Resize,ToTensor
    train_transforms = transforms.Compose([
        #Data Augmentations
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        #shear
        #rotation
        #scale
        #transformations to fit in Network
        GrayScale(),
        Resize(img_size),
        ToTensor(),

    ])
    eval_transforms = transforms.Compose([
        GrayScale(),
        Resize(img_size),
        ToTensor()
    ])


    # set Dataset and DataLoader
    train_dataset = LungSegDataset(transforms=train_transforms)
    val_dataset = LungSegDataset(split='val',transforms=eval_transforms)
    test_dataset = LungSegDataset(split = 'test',transforms=eval_transforms)

    from torch.utils.data import DataLoader
    dataloader = {'train' : DataLoader(dataset = train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True),
                  'val' :   DataLoader(dataset = val_dataset  , batch_size=args.batch_size, num_workers=args.n_workers),
                  'test':   DataLoader(dataset = test_dataset , batch_size=args.batch_size, num_workers=args.n_workers)}


    # checkpoint dir
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = args.load_model


    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=1e-5)

    # learning rate scheduler
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    # scheduler = StepLR(optimizer, step_size = 3 , gamma = 0.8)
    ## option 2.
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5,verbose=True)

    # # set criterion
    # if model.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()
    criterion = DiceLoss()
    #criterion = nn.BCEWithLogitsLoss()

    train_and_validate(net=model,criterion=criterion,optimizer=optimizer,dataloader=dataloader,device=device,epochs=args.epochs, scheduler=scheduler,load_model=checkpoint_path)
    # test()
    test(net=model,criterion=criterion,dataloader=dataloader,device=device)

if __name__ == '__main__':
    main()
