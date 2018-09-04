
import numpy as np
import scipy.sparse as sp
import pickle
import random
import sys
from hyperparams import Hyperparams as hp

mini = int(sys.argv[1])
gt = int(sys.argv[2])
if mini == 1:
    FILE_LENGTH = 500
else:
    FILE_LENGTH = 331590167

CHUNK_NUM = 4
#FILE_NAME = 'info.txt'


def split_file(file_length, chunk_num):
    # return:
    # boundary: list
    chunk_length = file_length / chunk_num
    boundary = [0]
    for i in range(chunk_num):
        boundary.append(chunk_length * (i + 1))
    return boundary

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



def read_data(filename, num_negatives = 20, mini = 0):
    file_handle = open(filename, "r")
    
    # split file
    splits_bd = split_file(FILE_LENGTH, CHUNK_NUM)

    unique_users_raw = set()
    unique_items_raw = set()
    unique_users = set()
    unique_items = set()
    item2itemid = {}
    user2userid = {}

    for idx, line in enumerate(file_handle.readlines()):
        if line != None and line != "":
            arr = line.split(",")
            u, i = int(arr[0]), int(arr[1])
            unique_users_raw.add(u)
            unique_items_raw.add(i)


    itemid = 0
    for item in unique_items_raw:
        item2itemid[item] = itemid 
        unique_items.add(itemid)
        itemid += 1
    userid = 0  
    for user in unique_users_raw:
        user2userid[user] = userid
        unique_users.add(userid)
        userid += 1

    for chunk_id, (chunk_start, chunk_end) in enumerate(zip(splits_bd[:-1], splits_bd[1:])):

        mat = {}
        testRatings = []
        testNegatives = []

        len_unique_items = len(unique_items)
        current_key = None
        items_set = set()

        file_handle.seek(0)
        for idx, line in enumerate(file_handle.readlines()):
            if idx >= chunk_start and idx <= chunk_end and line != None and line != "":
                arr = line.split(",")
                user, item, ptime = int(arr[0]), int(arr[1]), 0 if arr[3] == 'None' else int(arr[3])
                user = user2userid[user]
                item = item2itemid[item]
                if user == current_key:
                    items_set.add(item)
                    items_ptime_list.append((item, ptime))
                else:
                    if current_key:
                        for i in range(num_negatives):
                            negative_item = random.sample(unique_items, 1)[0]
                            while negative_item in items_set:
                                negative_item = random.sample(unique_items, 1)[0]
                            negatives.append(negative_item)
                        testNegatives.append(negatives)
                        test_item_index = np.random.choice(len(items_set))
                        test_item = items_ptime_list[test_item_index][0]
                        items_set.remove(test_item)
                        del items_ptime_list[test_item_index]
                        testRatings.append([current_key, test_item])
                        for i in items_ptime_list:
                            mat[current_key, i[0]] = i[1] 

                    items_set = set()
                    items_ptime_list = []
                    negatives = []
                    current_key = user
                    items_set.add(item)
                    items_ptime_list.append((item, ptime))
        for i in range(num_negatives):
            negative_item = random.sample(unique_items, 1)[0]
            while negative_item in items_set:
                negative_item = random.sample(unique_items, 1)[0]
            negatives.append(negative_item)
        testNegatives.append(negatives)
        test_item_index = np.random.choice(len(items_set))
        test_item = items_ptime_list[test_item_index][0]
        items_set.remove(test_item)
        del items_ptime_list[test_item_index]
        testRatings.append([current_key, test_item])
        for i in items_ptime_list:
            
            mat[current_key, i[0]] = i[1] 
        prepath = hp.prepath + hp.fn if gt == 1 else hp.prepath #yueqiu
        prepath = prepath + 'mini_' if mini == 1 else prepath #yueqiu

        save_object(mat, prepath +'mat_' + str(chunk_id) + '.p')
        save_object(testRatings, prepath +'testRatings_' + str(chunk_id) + '.p')
        save_object(testNegatives, prepath +'testNegatives_' + str(chunk_id) + '.p')
        del mat
        del testRatings
        del testNegatives

    save_object(unique_users, prepath +'unique_users.p')
    save_object(unique_items, prepath +'unique_items.p')  

    num_users = len(unique_users)
    num_items = len(unique_items)
    save_object(num_users, prepath +'num_users.p')
    save_object(num_items, prepath +'num_items.p')

    file_handle.close()
    
def main():
    prepath = '../data/'
    mini = int(sys.argv[1])
    if mini == 1:
        filename = prepath + 'info_mini2.txt'
    else:
        filename = prepath + 'info.txt'
    read_data(filename, mini = mini)



if __name__ == '__main__':
    main()
