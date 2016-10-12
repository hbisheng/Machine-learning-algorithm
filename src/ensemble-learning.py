import math
import random
from sklearn import cross_validation
from sklearn import svm
from sklearn import tree
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn import neighbors

def Bagging(classifier_name, fea_train, fea_test, label_train, label_test, bagging_iter = 3):
    if classifier_name == 'SVM':
        clf = svm.SVC()
    elif classifier_name == 'DTree':
        clf = tree.DecisionTreeClassifier()
    elif classifier_name == 'Baynes':
        clf = naive_bayes.BernoulliNB()
    elif classifier_name == 'KNN':
        clf = neighbors.KNeighborsClassifier()
    else:   
        return False
    prediction_all = [0 for i in range(0, len(label_test))]
    aver_acc = 0
    for i in range(0, bagging_iter):
        # bootstrap resampling from train data
        fea_resample = []
        label_resample = []
        for k in range(0, len(fea_train)):
            index = random.randint(0, len(fea_train)-1)
            fea_resample.append(fea_train[index])
            label_resample.append(label_train[index])            
        clf.fit(fea_resample, label_resample)
        predict = clf.predict(fea_test) # a sequence of 0 or 1
        acc = 0
        for ii in range(0,len(label_test)):
            acc += 1 if label_test[ii] == predict[ii] else 0
        aver_acc += 1.0 * acc / len(label_test)
        for j in range(0, len(predict)):
            if predict[j] == 1:
                prediction_all[j] += 1
    
    bagging_acc = 0
    for i in range(0,len(label_test)):
        bagging_acc += 1 if (label_test[i] == 1) == (prediction_all[i] >= 0.5*bagging_iter) else 0
    print "BAGGING", classifier_name, "BAGGING AVERAGE ACC", aver_acc / bagging_iter, "FINAL ACC: ", 1.0 * bagging_acc / len(label_test)
    return 1.0 * bagging_acc / len(label_test)
 
# implement AdaBoost.M1
def AdaBoost(classifier_name, fea_train, fea_test, label_train, label_test, boost_iter = 3):
    if classifier_name == 'SVM':
        clf = svm.SVC()
    elif classifier_name == 'DTree':
        clf = tree.DecisionTreeClassifier()
    elif classifier_name == 'Baynes':
        clf = naive_bayes.BernoulliNB()
    elif classifier_name == 'KNN':
        clf = neighbors.KNeighborsClassifier()
    else:   
        return False
    sample_weights = [1.0/len(fea_train) for i in fea_train]
    prediction_all = [0.0 for i in range(0, len(label_test))]
    
    aver_acc = 0
    for i in range(0, boost_iter):
    
        # resample according to weights
        fea_resample = []
        label_resample = []
        for k in range(0, len(fea_train)):
            rand_num = random.random()
            accum = 0
            for kk in range(0, len(sample_weights)):
                accum += sample_weights[kk]
                if accum > rand_num:
                    fea_resample.append(fea_train[kk])
                    label_resample.append(label_train[kk])
                    break
        
        clf.fit(fea_resample, label_resample)
        predict = clf.predict(fea_test) # a sequence of 0 or 1
        acc = 0
        error = 0
        for ii in range(0,len(label_test)):
            if label_test[ii] == predict[ii]:
                acc += 1
            else:
                error += sample_weights[ii]
        #print "ADABOOSTING", classifier_name, "ITER", i, "ACC: ", 1.0 * acc / len(label_test)
        aver_acc += 1.0 * acc / len(label_test)
        # update weights
        beta = 1.0 * error / (1 - error)
        for ii in range(0, len(label_test)):
            if label_test[ii] == predict[ii]:
                sample_weights[ii] *= beta  
                
        # normalize weights
        sample_sum = sum(sample_weights)
        for ii in range(0, len(sample_weights)):
            sample_weights[ii] /= sample_sum
        for ii in range(0, len(predict)):
            prediction_all[ii] += math.log((1. / beta)) * predict[ii]
            #print  prediction_all[ii], predict[ii]
            
    adaboosting_acc = 0
    for i in range(0,len(label_test)):
        adaboosting_acc += 1 if (label_test[i] == 1) == (prediction_all[i] >= 0) else 0
    print "ADABOOSTING", classifier_name, "ADABOOSTING AVERAGE ACC", aver_acc / bagging_iter, "FINAL ACC: ", 1.0 * adaboosting_acc / len(label_test)
    return 1.0 * adaboosting_acc / len(label_test)
    
def normalize_sample(fea_train, fea_test):
    return preprocessing.scale(fea_train), preprocessing.scale(fea_test)

def balance_sample(fea_train, label_train):
    pos_sample = label_train.count(1)
    neg_sample = len(label_train) - pos_sample
    num = min(pos_sample, neg_sample)
    
    fea_trun_train = []
    label_trun_train = []
    pos_cnt = 0
    neg_cnt = 0
    for i in range(0, len(fea_train)):
        if label_train[i] == 1 and pos_cnt < num:
            fea_trun_train.append(fea_train[i])
            label_trun_train.append(label_train[i])
            pos_cnt += 1
        elif label_train[i] != 1 and neg_cnt < num:
            fea_trun_train.append(fea_train[i])
            label_trun_train.append(label_train[i])
            neg_cnt += 1
    
    return fea_trun_train, label_trun_train

if __name__ == "__main__":    
    # load original data
    f = open('ContentNewLinkAllSample.csv','r') 
    text = f.readlines()
    f.close()
    # get rid of header
    del text[0] 
    # load data matrix
    fea_matrix = [] 
    label_list = []
    for line in text:
        fea_origin = line.split(',')
        assert len(fea_origin) == 235
        label = 1 if fea_origin[-1].strip() == 'spam' else -1
        del fea_origin[-1] # delete class label from raw feature
        fea = [float(i) for i in fea_origin]
        fea_matrix.append(fea)
        label_list.append(label)
    
    cv_num = 5
    bagging_iter = 10
    boost_iter = 10
    
    Bagging_SVM = []
    Bagging_Dtree = []
    
    Bagging_trun_SVM = []
    Bagging_trun_Dtree = []
    
    AdaBoost_SVM = []
    AdaBoost_DTree = []
    AdaBoost_norm_SVM = []
    AdaBoost_norm_DTree = []

    Bagging_KNN = []    
    AdaBoost_Baynes = []
    
    for i in range(0, cv_num):
        fea_train, fea_test, label_train, label_test = cross_validation.train_test_split(
            fea_matrix, label_list, test_size=0.2, random_state=int(random.random()*100))
        
        fea_norm_train, fea_norm_test = normalize_sample(fea_train, fea_test)
        fea_trun_train, label_trun_train = balance_sample(fea_train, label_train) # to deal with data inbalance

        Bagging_SVM.append(         Bagging('SVM',fea_train, fea_test, label_train, label_test, bagging_iter)) 
        Bagging_Dtree.append(       Bagging('DTree',fea_train, fea_test, label_train, label_test, bagging_iter))
        Bagging_trun_SVM.append(    Bagging('SVM',fea_trun_train, fea_test, label_trun_train, label_test, bagging_iter) ) 
        Bagging_trun_Dtree.append(  Bagging('DTree',fea_trun_train, fea_test, label_trun_train, label_test, bagging_iter) ) 
        
        AdaBoost_SVM.append(        AdaBoost('SVM',fea_train, fea_test, label_train, label_test, boost_iter))
        AdaBoost_DTree.append(      AdaBoost('DTree',fea_train, fea_test, label_train, label_test, boost_iter))
        AdaBoost_norm_SVM.append(   AdaBoost('SVM', fea_norm_train, fea_norm_test, label_train, label_test, boost_iter))
        AdaBoost_norm_DTree.append( AdaBoost('DTree', fea_norm_train, fea_norm_test, label_train, label_test, boost_iter))
        
        Bagging_KNN.append(         Bagging('KNN', fea_train, fea_test, label_train, label_test, bagging_iter))
        AdaBoost_Baynes.append(     AdaBoost('Baynes', fea_train, fea_test, label_train, label_test, bagging_iter))
        
        
    f = open('ans.txt','w')
    f.write(str(Bagging_SVM)+'\n')
    f.write(str(Bagging_Dtree)+'\n')
    f.write(str(Bagging_trun_SVM)+'\n')
    f.write(str(Bagging_trun_Dtree)+'\n')
    f.write(str(AdaBoost_SVM)+'\n')
    f.write(str(AdaBoost_DTree)+'\n')
    f.write(str(AdaBoost_norm_SVM)+'\n')
    f.write(str(AdaBoost_norm_DTree)+'\n')
    f.write(str(Bagging_KNN)+'\n')
    f.write(str(AdaBoost_Baynes)+'\n')
    f.close()
    
    
    