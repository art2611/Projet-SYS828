# Allow to get the complement of a list slice
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
import random

# Renvoie la base de données d'images sous X et les labels sous y
# Conserve uniquement 15 images par personnes maximum
def prepare_dataset(lfw_people, nb_img_per_id_to_keep):
    n_samples, h, w = lfw_people.images.shape

    X_init = lfw_people.images
    y_init = lfw_people.target

    """
    
    Code attendu :
    Votre base X, y à définir. 
    
    """

    X, y, n_classes = None, None

    return X, y, n_classes


def prepare_data_ids(n_classes):

    """

    Code attendu :
    Définissez train_ids_lists, val_ids_lists, test_ids_list

    """

    train_ids_lists, val_ids_lists, test_ids_list = None, None, None

    return train_ids_lists, val_ids_lists, test_ids_list


def GenIdx(train_label):
    color_pos = []
    unique_label_color = np.unique(train_label)
    for i in range(len(unique_label_color)):

        tmp_pos = [k for k, v in enumerate(train_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)
    return color_pos

# Sampler - Will select the right amount of ids and images per ids for the upcoming dataloader
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label : labels of each modalities
            color_pos : positions of each identity
            batch_num_identities: batch size
    """

    def __init__(self, train_labels, color_pos, num_of_same_id_in_batch, batch_num_identities):
        uni_label = np.unique(train_labels)
        self.n_classes = len(uni_label)
        N = len(train_labels)  # nb_img_per_id = 15
        for j in range(int(N / (batch_num_identities * num_of_same_id_in_batch)) + 1):
            # We choose randomly "num_different_identities_in_batch" identities :
            batch_idx = np.random.choice(uni_label, batch_num_identities, replace=False)
            for i in range(batch_num_identities):  # For each ids in a batch
                # We choose randomly "num_img_of_same_id_in_batch" images :
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                if j == 0 and i == 0:
                    index = sample_color
                else:
                    index = np.hstack((index, sample_color))
        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N

def prepare_set(ids, nb_images_per_id, images_X):
    # Query and gallery are the same since we want to compare query to all gallery image
    query_gallery_img = np.zeros((len(ids) * nb_images_per_id, 50, 37))

    for i, id in enumerate(ids):
        sublist = images_X[id * nb_images_per_id:id * nb_images_per_id + nb_images_per_id]

        if i == 0:
            query_gallery_img = sublist
            label_query_gallery = [id for _ in range(nb_images_per_id)]
        else:
            query_gallery_img = np.concatenate((query_gallery_img, sublist), axis=0)
            label_query_gallery.extend([id for _ in range(nb_images_per_id)])

    return query_gallery_img, np.array(label_query_gallery), len(label_query_gallery)



class LFW_training_Data(data.Dataset):
    def __init__(self, X, img_pos, img_ids_to_extract, nb_img_per_id, transform=None, colorIndex=None):

        """

        Code attendu :

        A partir des identitées d'entraînement pour un fold donné (img_ids_to_extract):
        self.train_images devra être la sous base de X contenant uniquement les ids de img_ids_to_extract
        self.train_labels devra être la sous base de labels. Il est conseillé de re-labeliser les images, en
        partant de 0. C'est à dire que les 15 premières images de la sous bases seront associées au nouveau label 0,
        les 15 suivantes au nouveau label 1 etc...

        """


        self.train_images = "X_to_keep"
        self.train_labels = "relabel"

        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex

    def __getitem__(self, index):
        #Dataset[i] return image with its corresponding label
        img, target = self.train_images[self.cIndex[index]], self.train_labels[self.cIndex[index]]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count