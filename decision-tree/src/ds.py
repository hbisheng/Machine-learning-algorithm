from tools import con_thr, eval_attr, major_category
import copy
import random

class entry:
    def __init__(self, attributes, attr_content):
        self.label = attr_content[-1].strip()
        self.attr = []
        for a in attr_content:
            self.attr.append(a.strip())
        if len(self.attr) > len(attributes):
            self.attr.pop() # the last piece is label
    
class node:
    current_correct_num = 0
    all_num = 0
    all_acc = []
    all_node = []
    # best_acc = 0
    # best_node = 0
    def __init__(self, label, attr, valid_attr, dataset):
        # Tree building variable
        self.label = label
        self.attr = attr
        self.valid_attr = valid_attr
        self.dataset = dataset
        
        # SubTree building variable
        self.decision_index = -1
        self.decision_thresh = -1
        self.subtree = []
        
        # Classification variable
        self.is_leaf = False # 
        self.category = None
        
        # Pre-prunning variable
        self.for_prunning = [0,0,0,0]
    def build_tree(self):
        
        # deal with continuous data
        label = self.label
        attributes = self.attr
        valid_attr = self.valid_attr
        # evaluate thresh and get the best of it
        
        pos_flag = False
        neg_flag = False
        for entry in self.dataset:
            if pos_flag and neg_flag:
                break
            if entry.label == label[0]:
                pos_flag = True
            else:
                neg_flag = True
        self.category = major_category(self.label, self.dataset)
                    
        if len(valid_attr) == 0 or (not pos_flag) or (not neg_flag):
            # terminate as a leaf node
            self.is_leaf = True
            return
        
        thresh = con_thr(attributes, self.dataset)    
        freq_all = {}
        for i in valid_attr:
            if len(attributes[i]['values']) == 1:
                assert thresh.has_key(attributes[i]['name'])
                # ans_bag = []
                    # for th in thresh[attributes[i]['name']]:
                        # ans_bag.append([0,0,0,0])
                freq_all[attributes[i]['name']] = [0,0,0,0]
            else:
                ans_bag = []
                for v in attributes[i]['values']:
                    ans_bag.append(0)
                    ans_bag.append(0)
                freq_all[attributes[i]['name']] = ans_bag
        
        # entry_n = 0
        for entry in self.dataset:
            # entry_n += 1
            # if entry_n % 1000 == 0:
                # print entry_n
            for i in valid_attr:
                if len(attributes[i]['values']) == 1:
                    # compare with each threshold    
                    # continue                    
                    # if (attributes[i]['name'] == 'fnlwgt'):
                        # continue
                    # th_cnt = -1
                    # for th in thresh[attributes[i]['name']]:
                        # th_cnt += 1
                        # loc = 0
                    loc = 0 # loc: (condition, label) 0:(1 1)  1:(0,1) 2:(1,0) 3:(0,0)  
                    if float(entry.attr[i]) > float(thresh[attributes[i]['name']]):
                        loc += 1
                    if entry.label != label[0]:
                        loc += 2
                    freq_all[attributes[i]['name']][loc] += 1
                else:                
                    assert entry.attr[i] in attributes[i]['values']
                    loc = attributes[i]['values'].index(entry.attr[i])
                    if entry.label == label[0]:
                        freq_all[attributes[i]['name']][loc] += 1
                    else:
                        freq_all[attributes[i]['name']][loc + len(attributes[i]['values'])] += 1
                        
        
        # evaluate each attribute by calculating ENTROPY 
        entropy_attr = []
        for i in valid_attr:
            entropy_attr.append( eval_attr(freq_all[attributes[i]['name']]) )
        
        # get the best attr
        self.decision_index = valid_attr[entropy_attr.index(min(entropy_attr))]
        
        # generating sub_tree
        
        sub_train = []    
        if len(attributes[self.decision_index]['values']) == 1:
            self.decision_thresh = thresh[attributes[self.decision_index]['name']]
            sub_train.append(list())
            sub_train.append(list())
            for entry in self.dataset:
                if float(entry.attr[self.decision_index]) < self.decision_thresh:
                    sub_train[0].append(entry)
                else:
                    sub_train[1].append(entry)
        else:
            for v in attributes[self.decision_index]['values']:
                sub_train.append(list()) 
            for entry in self.dataset:
                assert entry.attr[self.decision_index] in attributes[self.decision_index]['values']
                loc = attributes[self.decision_index]['values'].index(entry.attr[self.decision_index])
                sub_train[loc].append(entry)
                
        # for # value number
        sub_valid_attr = copy.deepcopy(self.valid_attr)
        sub_valid_attr.remove(self.decision_index)
        
        for sub_data in sub_train:
            tree = node(self.label, self.attr, sub_valid_attr, sub_data)
            tree.build_tree()
            self.subtree.append(tree)
        
    def count_node(self,level):
        if self.is_leaf == True:
            print 'leaf level', level
            return 1
        else:
            ans = 1
            for sub in self.subtree:
                ans += sub.count_node(level + 1)
            print 'Inner', level, ans
            return ans
    
    def classify(self, entry):
        # search until you get to the leaf node
        if self.is_leaf:
            return self.category
        
        parsing_label = None
        if len(self.attr[self.decision_index]['values']) == 1:
            if entry.attr[self.decision_index] == '?':
                parsing_label = self.subtree[random.randint(0,1)].classify(entry)  
            elif float(entry.attr[self.decision_index]) <= self.decision_thresh:
                parsing_label = self.subtree[0].classify(entry)
            else:
                parsing_label = self.subtree[1].classify(entry)
        else:    
            if entry.attr[self.decision_index] == '?':
                parsing_label = self.subtree[random.randint(0, len(self.subtree) - 1)].classify(entry)
            else:
                loc = self.attr[self.decision_index]['values'].index(entry.attr[self.decision_index])
                if(loc >= len(self.subtree)):
                    print loc, len(self.subtree)
                    assert False
                parsing_label =  self.subtree[loc].classify(entry)
        
        if parsing_label == self.label[0]:
            if entry.label == self.label[0]:
                self.for_prunning[0] += 1
            else:
                self.for_prunning[1] += 1
        else:    
            if entry.label == self.label[0]:
                self.for_prunning[2] += 1
            else:
                self.for_prunning[3] += 1
        
        return parsing_label
        
    def prunning_clear(self):
        self.for_prunning = [0,0,0,0]
        if self.is_leaf == False:
            for sub in self.subtree:
                sub.prunning_clear()
        
    def pruning_finder(self, root):
        if self.is_leaf == True:
            return
        new_acc = 0
        # print node.current_correct_num
        # print self.for_prunning
        # assert False
        if self.category == self.label[0]:
            new_acc = float(node.current_correct_num - self.for_prunning[3] + self.for_prunning[2]) / node.all_num 
        else:
            new_acc = float(node.current_correct_num - self.for_prunning[0] + self.for_prunning[1]) / node.all_num 
        node.all_acc.append(new_acc)
        node.all_node.append(self)
        for sub in self.subtree:
            sub.pruning_finder(root)
        
        # if node.prunning_found:
            # return
        """
        if self.subtree[0].is_leaf == False:
            for sub in self.subtree:
                sub.pruning_finder(root, best_node, best_acc, validate_dataset)
        self.is_leaf = True
        cnt = 0
        for i in range(0, len(validate_dataset)):
           if root.classify(validate_dataset[i]) == validate_dataset[i].label:
                cnt += 1
        acc =  1.0 * cnt / len(validate_dataset)
        print acc
        self.is_leaf = False
        """
        """
        if self.is_leaf == True:
            return
        
        self.is_leaf = True
        cnt = 0
        for i in range(0, len(validate_dataset)):
           if root.classify(validate_dataset[i]) == validate_dataset[i].label:
                cnt += 1
        acc =  1.0 * cnt / len(validate_dataset)
        if acc > node.best_acc:
            node.best_acc = acc
            node.best_node = self
            node.prunning_found = True
            print 'Found!', node.best_acc
            return
        print acc
        self.is_leaf = False
        for sub in self.subtree:
            sub.pruning_finder(root, validate_dataset)"""