# Allow to get the complement of a list slice
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.utils.data as data
import random

def prepare_dataset(lfw_people, nb_img_per_id_to_keep):
    n_samples, h, w = lfw_people.images.shape

    X_init = lfw_people.images
    y_init = lfw_people.target

    n_features = X_init.shape[1]

    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    X = np.zeros((n_classes * nb_img_per_id_to_keep, 50, 37))
    y = [0 for _ in range(nb_img_per_id_to_keep)]
    for idx, label in enumerate(np.unique(y_init)):  # Pour chaque personne
        person_idx = label == y_init
        X_label = X_init[person_idx, :]
        random_selection = np.random.randint(X_label.shape[0], size=nb_img_per_id_to_keep) # Get 15 ids from the available ones

        if idx == 0:
            X = X_label[random_selection, :]
        else:
            X = np.concatenate((X, X_label[random_selection, :]), axis=0)
            y.extend([label for _ in range(nb_img_per_id_to_keep)])
    return (X, y, n_classes)


def prepare_data_ids(n_classes):
    n_classes_trainval = int(0.8 * n_classes) - 1  # We suppress one id so that it n_classes_trainval%5=0
    print(n_classes_trainval)
    # Get the number of class needed in each set
    n_classes_test = n_classes - n_classes_trainval
    n_classes_train = int(0.8 * n_classes_trainval)
    n_classes_val = n_classes_trainval - n_classes_train

    print(f"n_classes_train : {n_classes_train}")
    print(f"n_classes_val : {n_classes_val}")
    print(f"n_classes_test : {n_classes_test}")

    # Random class selection generation
    shuffled_classes = random.sample([i for i in range(n_classes)], n_classes)

    test_ids_list = shuffled_classes[:n_classes_test]
    trainval_classes = shuffled_classes[n_classes_test:]

    # Prepare trainval ids for 5 folds
    trainval_classes_super = SuperList(trainval_classes)
    train_ids_lists, val_ids_lists = [[] for _ in range(5)], [[] for _ in range(5)]
    for i in range(5):
        val_ids_lists[i] = trainval_classes_super[15 * i:15 * (i + 1)]
        train_ids_lists[i] = trainval_classes_super[15 * i:15 * (i + 1):'c']
    return train_ids_lists, val_ids_lists, test_ids_list

class SuperList(list):
    def __getitem__(self, val):
        if type(val) is slice and val.step == 'c':
            copy = self[:]
            copy[val.start:val.stop] = []
            return copy

        return super(SuperList, self).__getitem__(val)

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

def prepare_set(ids, nb_images_per_id, images_X, labels_y):
    # Query and gallery are the same since we want to compare query to all gallery image
    query_gallery_img = np.zeros((len(ids) * nb_images_per_id, 50, 37))

    for i, id in enumerate(ids):
        sublist = images_X[id * nb_images_per_id:id * nb_images_per_id + nb_images_per_id]

        if i == 0:
            query_gallery_img = sublist
            label_query_gallery = [id for _ in range(nb_images_per_id)]
        else:
            query_gallery_img = np.concatenate((query_gallery_img, sublist), axis=0)
            label_query_gallery.extend([label for _ in range(nb_images_per_id)])

    return query_gallery_img, np.array(label_query_gallery)



class LFW_training_Data(data.Dataset):
    def __init__(self, X, img_pos, img_ids_to_extract, nb_img_per_id, transform=None, colorIndex=None):

        nb_imgs_to_extract = len(img_ids_to_extract)
        relabel = []
        X_to_keep = None
        for idx, img_id_to_extract in enumerate(img_ids_to_extract):
            first_loc = img_pos[img_id_to_extract][0]
            last_loc = img_pos[img_id_to_extract][-1]
            if idx == 0:
                X_to_keep = X[first_loc:last_loc + 1]
                relabel = [idx for _ in range(nb_img_per_id)]
            else:
                X_to_keep = np.concatenate((X_to_keep, X[first_loc:last_loc+1]), axis=0)
                relabel.extend([idx for _ in range(nb_img_per_id)])

        self.train_images = X_to_keep
        self.train_labels = relabel

        self.transform = transform
        # Prepare index
        self.cIndex = colorIndex

    def __getitem__(self, index):
        #Dataset[i] return images from both modal and the corresponding label
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