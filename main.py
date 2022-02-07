import time
import torch
import torch.nn as nn
import torch.optim as optim
from model import model
from loss import BatchHardTripLoss
from Evaluation import evaluation
from torch.autograd import Variable
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from utils import GenIdx,IdentitySampler, prepare_dataset, prepare_set, \
                    prepare_data_ids,LFW_training_Data, AverageMeter
import numpy as np

#### Fonction utiles ####

# Fonction d'entraînement
def train(epoch, criterion_id, criterion_tri, optimizer, trainloader, device, net):

    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    end = time.time()
    # Pour chaque batch
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        # Labels 1 and 2 are the same because the two inputs correspond to the same identity

        inputs = Variable(inputs.to(device))
        inputs = inputs.expand(-1, 3, 70, 57) # Expand on 3 channels

        labels = Variable(labels.to(device))

        data_time.update(time.time() - end)

        feat, out0 = net(inputs)

        loss_ce = criterion_id("A remplir")
        loss_tri, batch_acc = criterion_tri("A remplir")
        _, predicted = out0.max(1)


        correct += (batch_acc / 2)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_ce + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += labels.size(0)

        # Update loss for display
        train_loss.update(loss.item(), inputs.size(0))
        id_loss.update(loss_ce.item(), inputs.size(0))
        tri_loss.update(loss_tri.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print(inputs.size(0))
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  f'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  f'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  f'Accu: {100. * correct / total:.2f}')

# Fonction permettant la validation
def valid(epoch):
    # Prepare query gallery sets

    # Eventually save the model if this one is the best overall

    # Get the timer
    end = time.time()

    # Extract features from query and gallery
    query_gall_feat_pool, query_gall_feat= extract_query_gall_feat(query_gall_loader, n_query_gall, net=net)

    print(f"Feature extraction time : {time.time() - end}")
    start = time.time()

    # Compute the similarity (cosine)
    distmat_pool = np.matmul(query_gall_feat_pool, np.transpose(query_gall_feat_pool))
    distmat_fc = np.matmul(query_gall_feat, np.transpose(query_gall_feat))

    # Evaluation - Compute metrics (Rank-1, Rank-5, mAP)

    cmc_att, mAP_att, mINP_att  = evaluation(-distmat_pool, query_gallery_labels) # On avg_pool features
    cmc, mAP, mINP = evaluation(-distmat_fc, query_gallery_labels)                # On batch normed features

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


if __name__ == "__main__":

    # Work on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Import the lfw dataset - With 15 img min per person
    print("===> Loading Fetch Dataset \n")
    lfw_people = fetch_lfw_people(min_faces_per_person=15, resize=0.4)
    nb_img_per_id_to_keep = 15

    # Renvoie la base de données avec 15 images par personnes maximum
    # Vous devrez coder la fonction dans le fichier utils
    X, y, n_classes = prepare_dataset(lfw_people, nb_img_per_id_to_keep)


    # Produire les listes d'identitées qui correspondront aux folds d'entraînement / validation et
    # aux identitées de test.
    # train_ids_lists et val_ids_lists seront des listes de 5 listes d'identitées (Une liste pour chaque fold)
    # test_ids sera une liste de 21 identitées
    train_ids_lists, val_ids_lists, test_ids = prepare_data_ids(n_classes)

    # Prepare var - Batch size
    num_img_of_same_id_in_batch = 4
    num_different_identities_in_batch = 8
    batch_size = num_img_of_same_id_in_batch * num_different_identities_in_batch
    global_img_pos = GenIdx(y)  # Get the images positions in list for each specific identity

    folds = "A remplir"
    epochs = "A remplir"

    # img_h, img_w = 224, 224

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        # transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Définir vos fonctions de pertes (Cross-entropy + Fonction de pertes pour réseaux siamois)
    # Vous pouvez définir votre fonction de pertes pour réseaux siamois dans le fichier loss.py
    criterion_id = "A remplir"
    criterion_tri = "A remplir"

    for fold in range(folds):
        net = model(n_classes).to(device)

        # Définir votre optimizer :
        opt = "A remplir"

        # Preparez votre query / gallery set
        gallset, n_query_gall = prepare_set(val_ids_lists[fold], nb_img_per_id_to_keep, transform=transform_test,
                                            img_size=(224, 224))

        query_gall_loader = torch.utils.data.DataLoader(gallset, batch_size=32, shuffle=False)

        for epoch in range(epochs):

            trainset = LFW_training_Data(X, global_img_pos, train_ids_lists[fold], nb_img_per_id_to_keep, \
                                         transform=transform_train)

            img_pos = GenIdx(trainset.train_labels)  # Get the images positions in list for each specific identity

            sampler = IdentitySampler(trainset.train_labels, img_pos, num_img_of_same_id_in_batch,
                                      num_different_identities_in_batch)

            trainset.cIndex = sampler.index

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
                                                      sampler=sampler, drop_last=True)

            train(epoch, criterion_id, criterion_tri, opt, trainloader, device, net)

            # Call the validation every two epochs
            if epoch != 0 and epoch % 2 == 0:
                cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = valid(epoch)

                # Save model based on validation mAP or mINP ?
                if mAP > best_map:  # Usual saving

                    # best_acc = cmc_att[0]
                    best_map = mAP
                    best_minp = mINP
                    best_epoch = epoch
                    state = {
                        'net': net.state_dict(),
                        'cmc': cmc,
                        'mAP': mAP,
                        'mINP': mINP,
                        'epoch': epoch,
                    }
                    torch.save(state, f'model_fold({fold})_best.t')

                print('fc : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], mAP,
                                                                                                mINP))
                print('att : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4],
                                                                                                 mAP_att, mINP_att))

    # Tester ici ou dans un fichier à part votre modèle sur les données de test.


