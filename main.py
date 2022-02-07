import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import model
from loss import BatchHardTripLoss
from torch.autograd import Variable
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from utils import SuperList, GenIdx, prepare_set, IdentitySampler, \
    prepare_dataset, prepare_data_ids,LFW_training_Data, AverageMeter
import numpy as np
import random
    
# Work on GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import the lfw dataset - With 15 img min per person
lfw_people = fetch_lfw_people(min_faces_per_person=15, resize=0.4)
nb_img_per_id_to_keep = 15
X, y, n_classes = prepare_dataset(lfw_people, nb_img_per_id_to_keep)

# Get train / val folds ids and get test ids
train_ids_lists, val_ids_lists, test_ids = prepare_data_ids(n_classes)

# Prepare training data
num_img_of_same_id_in_batch = 4
num_different_identities_in_batch = 8
batch_size = num_img_of_same_id_in_batch * num_different_identities_in_batch

global_img_pos = GenIdx(y)  # Get the images positions in list for each specific identity

folds = 5

# epochs = "nb_epoch_to_define"
epochs = 10

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Criterion / loss :
criterion_id = nn.CrossEntropyLoss().to(device)
criterion_tri = BatchHardTripLoss(batch_size=batch_size, margin= 0.3).to(device)



def train(epoch):
    # extract features
    # Compute loss
    # Update weights
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    Channel_exchange_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # Labels 1 and 2 are the same because the two inputs correspond to the same identity

        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

        data_time.update(time.time() - end)

        feat, out0 = net(inputs)

        loss_ce = criterion_id(out0, labels)
        loss_tri, batch_acc = criterion_tri(feat, labels)
        _, predicted = out0.max(1)


        correct += (batch_acc / 2)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_ce + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  # f'lr:{current_lr:.3f} '
                  f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  # f'Channel Exchange Loss : ({Channel_exchange_loss.val:.4f}) ({Channel_exchange_loss.avg:.4f})'
                  f'Accu: {100. * correct / total:.2f}')

    pass
def valid(epoch):
    # Prepare query gallery sets
    # Extract features
    # Compare features (Similarity measure)
    # Compute metrics (Rank-1, Rank-5, mAP)
    # Eventually save the model if this one is the best overall
    pass

for fold in range(folds):
    net = model(n_classes)

    # Optimizer :
    lr = 0.1
    optimizer = optim.SGD([
        {'params': net.parameters(), 'lr': lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

    for epoch in range(epochs):
        trainset = LFW_training_Data(X, global_img_pos, train_ids_lists[fold], nb_img_per_id_to_keep,\
                                     transform=transform_train)

        img_pos = GenIdx(trainset.train_labels)  # Get the images positions in list for each specific identity

        sampler = IdentitySampler(trainset.train_labels, img_pos, num_img_of_same_id_in_batch,
                                  num_different_identities_in_batch)
        trainset.cIndex = sampler.index

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,\
                                                  sampler=sampler, drop_last=True)

        for data, target in trainloader:
            print(data.shape)
            print(target.shape)
            break
        train(epoch)

        if epoch != 0 and epoch % 2 == 0:
            valid(epoch)
        break

# Eventually here test all models
