import numpy as np

def evaluation(distmat, query__gallery_labels, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # Get the the matrix, which a row give the indices of the lowest scores by index in distmat
    indices = np.argsort(distmat, axis=1)
    matches = (query__gallery_labels[indices] == query__gallery_labels[:, np.newaxis]).astype(np.int32)
    # print(f"Matches : {matches}")
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_number in range(num_q):

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_number]

        # We get rid of the match with the exact same image
        remove = [False for i in range(num_g)]
        remove[q_number] = True
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_number][keep]  # binary vector, positions with value 1 are correct matches

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP