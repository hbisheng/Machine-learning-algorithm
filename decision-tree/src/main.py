import sys
import random
import copy
from ds import entry, node # data structure
from pre import load_attribute, load_data   # preprocessing procedure
from tools import freq_analy

if __name__ == "__main__":

    #1. loading in data
    attr_path = '../data/adult.names'
    train_path = '../data/adult.data'
    test_path = '../data/adult.test'
    blackbox_test_path = '../data/test.features'
    
    [label, attributes] = load_attribute(attr_path);
    train_entries_origin = load_data(attributes, train_path)
    test_entries = load_data(attributes, test_path)
    blackbox_entries = load_data(attributes, blackbox_test_path)
    
    ######################## Task one  ##################################
    print '############Doing Task 1###############'
    train_ratio = [0.05, 0.5, 1]
    task1_acc = []
    for ratio in train_ratio:
        t_index = range(0, len(train_entries_origin))
        random.shuffle(t_index)
        t_index = t_index[ 0 : int(ratio * len(t_index))]
        train_entries = []
        for t in t_index:
            train_entries.append(copy.deepcopy(train_entries_origin[t]))
        # train_entries save as temporal training dataset#
        ratio_acc = []
        for times in range(0, 5):
            print (ratio,times),
            # 2. fix missing values
            [pos, neg] = freq_analy(label, attributes, train_entries)
            fix_pos = {}
            fix_neg = {}
            for i in range(0, len(attributes)):
                if pos[i].has_key('?'):
                    # sort pos[i], get the biggest attr
                    sort_pos =  sorted(pos[i].items(), key = lambda d: d[1], reverse = True)
                    fix_pos[attributes[i]['name']] = sort_pos[0][0]
                if neg[i].has_key('?'):
                    # the same as above
                    sort_neg = sorted(neg[i].items(), key = lambda d: d[1], reverse = True)
                    fix_neg[attributes[i]['name']] = sort_neg[0][0]
            for entry in train_entries:
                for i in range(0, len(attributes)):
                    if(entry.attr[i] == '?'):
                        if(entry.label == label[0]):
                            entry.attr[i] = fix_pos[attributes[i]['name']]
                        else:
                            entry.attr[i] = fix_neg[attributes[i]['name']]
        
            # 3. Classification Starts from the root node
            valid_attr = range(0, len(attributes)) # only valid attributes will be considered 
            root = node(label, attributes, valid_attr, train_entries);
            root.build_tree()
    
            acc = 0
            for i in range(0, len(test_entries)):
                if root.classify(test_entries[i]) == test_entries[i].label:
                    acc += 1
            ratio_acc.append(1.0 * acc / len(test_entries))
            print 'Testing Acc', ratio_acc[-1]
        task1_acc.append(ratio_acc)
    
    # Also report Training Acc
    
    
    
    
    ######################## Task two  ##################################
    print '############Doing Task 2###############'
    train_ratio = [0.05, 0.5, 1]
    task2_acc = []
    for ratio in train_ratio:
        t_index = range(0, len(train_entries_origin))
        random.shuffle(t_index)
        t_index = t_index[ 0 : int(ratio * len(t_index))]
        
        train_entries = []
        validate_entries = []
        for t in range(0,len(t_index)):
            if t < len(t_index)*0.4:
                train_entries.append(copy.deepcopy(train_entries_origin[t_index[t]]))
            else:
                validate_entries.append(copy.deepcopy(train_entries_origin[t_index[t]]))
        # train_entries save as temporal training dataset
        ratio_acc = []
        for times in range(0, 5):
            print (ratio,times),
            # 2. fix missing values
            [pos, neg] = freq_analy(label, attributes, train_entries)
            fix_pos = {}
            fix_neg = {}
            for i in range(0, len(attributes)):
                if pos[i].has_key('?'):
                    # sort pos[i], get the biggest attr
                    sort_pos =  sorted(pos[i].items(), key = lambda d: d[1], reverse = True)
                    fix_pos[attributes[i]['name']] = sort_pos[0][0]
                if neg[i].has_key('?'):
                    # the same as above
                    sort_neg = sorted(neg[i].items(), key = lambda d: d[1], reverse = True)
                    fix_neg[attributes[i]['name']] = sort_neg[0][0]
            for entry in train_entries:
                for i in range(0, len(attributes)):
                    if(entry.attr[i] == '?'):
                        if(entry.label == label[0]):
                            entry.attr[i] = fix_pos[attributes[i]['name']]
                        else:
                            entry.attr[i] = fix_neg[attributes[i]['name']]
        
            # 3. Classification Starts from the root node
            valid_attr = range(0, len(attributes)) # only valid attributes will be considered 
            root = node(label, attributes, valid_attr, train_entries);
            root.build_tree()
            
            node.all_num = len(validate_entries)
            last_acc = 0 
            # on validate dataset
            while True:
                root.prunning_clear()
                acc = 0
                for i in range(0, len(validate_entries)):
                    if root.classify(validate_entries[i]) == validate_entries[i].label:
                        acc += 1
                now_acc = 1.0 * acc / len(validate_entries)
                if now_acc < last_acc:
                    break
                last_acc = now_acc
                node.all_acc = []
                node.all_node = []
                node.current_correct_num = acc
                root.pruning_finder(root)
                
                if len(node.all_acc) == 0:
                    break
                # ??why
                node.all_node[node.all_acc.index(max(node.all_acc))].is_leaf = True
                print 'Validating Acc', now_acc
            
            # on test dataset
            acc = 0
            for i in range(0, len(test_entries)):
                if root.classify(test_entries[i]) == test_entries[i].label:
                    acc += 1
            ratio_acc.append(1.0 * acc / len(test_entries))
            print 'Testing Acc', ratio_acc[-1]
        task2_acc.append(ratio_acc)
    
    ######################## Task Three  ##################################
    print '############Doing Task 3###############'
    t_index = range(0, len(train_entries_origin))
    train_entries = []
    validate_entries = []
    for t in range(0,len(t_index)):
        if t > len(t_index) * 0.4:
            train_entries.append(copy.deepcopy(train_entries_origin[t_index[t]]))
        else:
            validate_entries.append(copy.deepcopy(train_entries_origin[t_index[t]]))
    
    [pos, neg] = freq_analy(label, attributes, train_entries)
    fix_pos = {}
    fix_neg = {}
    for i in range(0, len(attributes)):
        if pos[i].has_key('?'):
                    # sort pos[i], get the biggest attr
            sort_pos =  sorted(pos[i].items(), key = lambda d: d[1], reverse = True)
            fix_pos[attributes[i]['name']] = sort_pos[0][0]
        if neg[i].has_key('?'):
                    # the same as above
            sort_neg = sorted(neg[i].items(), key = lambda d: d[1], reverse = True)
            fix_neg[attributes[i]['name']] = sort_neg[0][0]
    for entry in train_entries:
        for i in range(0, len(attributes)):
            if(entry.attr[i] == '?'):
                if(entry.label == label[0]):
                    entry.attr[i] = fix_pos[attributes[i]['name']]
                else:
                    entry.attr[i] = fix_neg[attributes[i]['name']]
                    
    # train_entries save as temporal training dataset
    valid_attr = range(0, len(attributes)) # only valid attributes will be considered 
    root = node(label, attributes, valid_attr, train_entries);
    root.build_tree()

    node.all_num = len(validate_entries)
    last_acc = 0 
    # on validate dataset
    while True:
        root.prunning_clear()
        acc = 0
        for i in range(0, len(validate_entries)):
            if root.classify(validate_entries[i]) == validate_entries[i].label:
                acc += 1
        now_acc = 1.0 * acc / len(validate_entries)
        if now_acc < last_acc:
            break
        last_acc = now_acc
        node.all_acc = []
        node.all_node = []
        node.current_correct_num = acc
        root.pruning_finder(root)

        if len(node.all_acc) == 0:
            break
        node.all_node[node.all_acc.index(max(node.all_acc))].is_leaf = True
        print 'Validating Acc', now_acc

    # on black-box test dataset
    output = open('2012011307.test.result','wb')
    acc = 0
    for entry in blackbox_entries:
        output.write(root.classify(entry) + '.\n')
    
    print 'Task one'
    print '5%', min(task1_acc[0]), sum(task1_acc[0])/len(task1_acc[0]), max(task1_acc[0])
    print '50%', min(task1_acc[1]), sum(task1_acc[1])/len(task1_acc[1]), max(task1_acc[1])
    print '100%', min(task1_acc[2]), sum(task1_acc[2])/len(task1_acc[2]), max(task1_acc[2]) 
    # Also report Training Acc
    print 'Task two'
    print '5%', min(task2_acc[0]), sum(task2_acc[0])/len(task2_acc[0]), max(task2_acc[0])
    print '50%', min(task2_acc[1]), sum(task2_acc[1])/len(task2_acc[1]), max(task2_acc[1])
    print '100%', min(task2_acc[2]), sum(task2_acc[2])/len(task2_acc[2]), max(task2_acc[2])