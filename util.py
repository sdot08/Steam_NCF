import pickle
import numpy as np
def find_cat(item, cat_id2cat, num_cat = 18):
    if item in cat_id2cat:
        cat = cat_id2cat[item]
    else:
        cat = [0] * num_cat
    return cat

def get_train_instances(train, num_negatives):
    cat_id2cat = pickle.load(open("./Data/dict_id2cat.p", "rb" ))
    user_input, item_input, cat_input, labels = [],[],[],[]
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        cat = find_cat(i, cat_id2cat)
        cat_input.append(cat)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            cat = find_cat(j, cat_id2cat)
            user_input.append(u)
            item_input.append(j)
            cat_input.append(cat)
            labels.append(0)
    return user_input, item_input, cat_input, labels