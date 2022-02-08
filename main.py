import time
import torch
import torch.optim as optim
from loss import *
from model import model
from Evaluation import evaluation
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from utils import GenIdx,IdentitySampler, prepare_dataset, prepare_set, \
                    prepare_data_ids,LFW_training_Data
import numpy as np

#### Fonction utiles ####

# Fonction d'entraînement
def train(epoch, criterion, optimizer, trainloader, device, net):

    """

    Se réferer éventuellement au laboratoire 5

    """

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

    cmc_att, mAP_att  = evaluation(-distmat_pool, query_gallery_labels) # On avg_pool features
    cmc, mAP = evaluation(-distmat_fc, query_gallery_labels)                # On batch normed features

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, cmc_att, mAP_att


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


    # Produire les listes d'identités qui correspondront aux folds d'entraînement / validation et
    # aux identités de test.
    # train_ids_lists et val_ids_lists seront des listes de 5 listes d'identités (Une liste pour chaque fold)
    # test_ids sera une liste de 21 identités
    train_ids_lists, val_ids_lists, test_ids = prepare_data_ids(n_classes)

    # Prepare var - Batch size
    num_img_of_same_id_in_batch = 4
    num_different_identities_in_batch = 8
    batch_size = num_img_of_same_id_in_batch * num_different_identities_in_batch
    global_img_pos = GenIdx(y)  # Get the images positions in list for each specific identity

    folds = "A définir"
    epochs = "A définir"

    img_h, img_w = 288, 144

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_h, img_w)),
        transforms.Pad(10),
        transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])

    # Définir votre ou vos fonctions de pertes
    # Vous pouvez définir votre/vos fonction de pertes dans le fichier loss.py
    criterion_1 = "A définir"

    for fold in range(folds):
        net = model(n_classes).to(device)

        # Définir votre optimizer :
        opt = "A remplir"

        # Preparez votre query / gallery set pour la validation
        # A noter que query set = gallery set si vous comptez comparer chaque image de la base avec toute les autres
        query_gall_set, n_query_gall = prepare_set(X, nb_img_per_id_to_keep, val_ids_lists[fold], transform=transform_test)
        query_gall_loader = torch.utils.data.DataLoader(query_gall_set, batch_size=32, shuffle=False)


        best_map = 0
        for epoch in range(epochs):

            trainset = LFW_training_Data(X, global_img_pos, train_ids_lists[fold], transform=transform_train)

            img_pos = GenIdx(trainset.train_labels)  # Get the images positions in list for each specific identity

            # Permet de sampler la base de données par batch contenant le bon nombre d'identités et d'images par identités
            sampler = IdentitySampler(trainset.train_labels, img_pos, num_img_of_same_id_in_batch,
                                      num_different_identities_in_batch)
            trainset.cIndex = sampler.index

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
                                                      sampler=sampler, drop_last=True)

            train(epoch, criterion_1, opt, trainloader, device, net)

            # Call the validation every two epochs
            if epoch != 0 and epoch % 2 == 0:
                cmc, mAP, cmc_att, mAP_att = valid(epoch)

                # Save model based on validation mAP
                if mAP > best_map:  # Usual saving

                    best_m
                    ap = mAP
                    best_epoch = epoch
                    state = {
                        'net': net.state_dict(),
                        'cmc': cmc,
                        'mAP': mAP,
                        'epoch': epoch,
                    }
                    torch.save(state, f'model_fold({fold})_best.t')

                print('fc : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}'.format(cmc[0], cmc[4], mAP ))
                print('att : Rank-1: {:.2%} | Rank-5: {:.2%} | mAP: {:.2%}'.format(cmc_att[0], cmc_att[4], mAP_att))

    # Tester ici ou dans un fichier à part votre modèle sur les données de test.


