'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import pickle
import random
import sys
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def read_data(filename, num_negatives = 20, mini = 0):
    
    num_users, num_items = 0, 0
    unique_users_raw = set()
    unique_items_raw = set()
    unique_users = set()
    unique_items = set()
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            u, i = int(arr[0]), int(arr[1])
            unique_users_raw.add(u)
            unique_items_raw.add(i)
            line = f.readline()
    item2itemid = {}
    user2userid = {}
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
    num_users = len(unique_users)
    num_items = len(unique_items)

    #mat = sp.dok_matrix((num_users+1, num_items -1 +1), dtype=np.float32)
    mat = {}
    len_unique_items = len(unique_items)
    current_key = None
    items_set = set()
    testRatings = []
    testNegatives = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(",")
            user, item, rating = int(arr[0]), int(arr[1]), 1.0
            user = user2userid[user]
            item = item2itemid[item]
            if user == current_key:
                if rating > 0:
                    items_set.add(item)
            else:
                if current_key:
                    for i in range(num_negatives):
                        negative_item = random.sample(unique_items, 1)[0]
                        #negative_item = items_set[int(np.random.random() * len_unique_items)]
                        while negative_item in items_set:
                            negative_item = random.sample(unique_items, 1)[0]
                            #negative_item = items_set[int(np.random.random() * len_unique_items)]
                        negatives.append(negative_item)
                    testNegatives.append(negatives)
                    test_item = random.sample(items_set, 1)[0]
                    #test_item = items_set[int(np.random.random() * len(items_set))]
                    items_set.remove(test_item)
                    testRatings.append([current_key, test_item])
                    for i in items_set:
                        mat[current_key, i] = 1.0  

                items_set = set()
                negatives = []
                current_key = user
                items_set.add(item)
            line = f.readline()  
    for i in range(num_negatives):
        negative_item = random.sample(unique_items, 1)[0]
        #negative_item = items_set[int(np.random.random() * len_unique_items)]
        while negative_item in items_set:
            negative_item = random.sample(unique_items, 1)[0]
            #negative_item = items_set[int(np.random.random() * len_unique_items)]
        negatives.append(negative_item)
    testNegatives.append(negatives)
    test_item = random.sample(items_set, 1)[0]
    #test_item = items_set[int(np.random.random() * len(items_set))]
    items_set.remove(test_item)
    testRatings.append([current_key, test_item])
    for i in items_set:
        mat[current_key, i] = 1.0
    prepath = '../data/'
    if mini == 1:
        prepath += 'mini_'
    save_object(mat, prepath +'mat.p')
    save_object(testRatings, prepath +'testRatings.p')
    save_object(testNegatives, prepath +'testNegatives.p')
    save_object(num_users, prepath +'num_users.p')
    save_object(num_items, prepath +'num_items.p')
    save_object(unique_users, prepath +'unique_users.p')
    save_object(unique_items, prepath +'unique_items.p')
    
def main():
    prepath = '../data/'
    mini = int(sys.argv[1])
    if mini == 1:
        filename = prepath +'info_mini2.txt'
    else:
        filename = prepath +'info.txt'
    read_data(filename, mini = mini)



if __name__ == '__main__':
    main()
