import math

def validate_entry(attributes, e):
    if len(attributes) != len(e.attr):
        return False
    for i in range(0, len(attributes)):
        if e.attr[i] != '?':
            if len(attributes[i]['values']) == 1 and not e.attr[i].isdigit(): # continuous attr
                return False
            elif len(attributes[i]['values']) != 1 and e.attr[i] not in attributes[i]['values']:
                return False
    return True         

def freq_analy(label, attributes, entry_list): 
# Get the attributes frequencies
# Divide into two groups: Positive and Negative
    pos = []
    neg = []
    for i in range(0, len(attributes)):
        d = {} 
        e = {}
        pos.append(d)        
        neg.append(e)
    
    for entry in entry_list:
        for i in range(0, len(attributes)):
            if(entry.label == label[0]):
                dict = pos[i]
            else:
                dict = neg[i]
            if dict.has_key(entry.attr[i]):
                dict[entry.attr[i]] += 1
            else:
                dict[entry.attr[i]] = 1
    return (pos, neg)
  
# Get candidate threshold for continuous attributes  
def con_thr(attributes, data_set):
    threshold = {}
    for i in range(0, len(attributes)):
        if(len(attributes[i]['values']) == 1): # this is a continuous attr
            data_sorted = sorted(data_set, key = lambda d: float(d.attr[i]), reverse = True)
            candidate = []
            for j in range(0, len(data_set) - 1):
               if data_sorted[j].label != data_sorted[j+1].label and data_sorted[j].attr[i] != data_sorted[j+1].attr[i]:
                   candidate.append( (int(data_sorted[j].attr[i],10) + int(data_sorted[j + 1].attr[i],10) ) / 2.0) 
            # if len(data_set) == 49:
                # for d in data_sorted:
                    # print (d.attr[i],d.label), 
                # print attributes[i]['name']
                # print sum(candidate), len(candidate)
            if len(candidate) == 0:
                threshold[attributes[i]['name']] = float(data_sorted[0].attr[i]) + float(data_sorted[-1].attr[i]) / 2
            else:    
                threshold[attributes[i]['name']] = sum(candidate) / len(candidate)
    # assert False
    return threshold

def entropy(a, b):
    if a == 0 or b == 0:
        return 0
    p1 = float(a) / (a + b);    
    p2 = float(b) / (a + b);
    return - p1 * math.log(p1, 2) - p2 * math.log(p2, 2) 
    
def eval_attr(freq):
    pos = freq[:len(freq)/2]  #satisfy condition 
    neg = freq[len(freq)/2:]  #not satisfy
    
    ans = 0
    for i in range(0, len(pos)):
        ans += float(pos[i]+neg[i]) / sum(freq)* entropy(pos[i],neg[i])
    return ans
    
def major_category(label, dataset):
    pos_cnt = 0
    for entry in dataset:
        if entry.label == label[0]:
            pos_cnt += 1
    if pos_cnt > len(dataset) / 2:
        return label[0]
    else:
        return label[1]