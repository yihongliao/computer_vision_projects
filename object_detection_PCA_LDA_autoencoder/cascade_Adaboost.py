#!/usr/bin/env python
# coding: utf-8

# In[248]:


import numpy as np 
import matplotlib.pyplot as plt
import cv2
import math
from scipy.spatial import distance
from scipy.linalg import null_space
from pathlib import Path
import os
from os import listdir
import re
from sklearn import preprocessing
import pickle


# In[249]:


def extract_features(folder_dir, label, num_data):
    features = []
    labels = []
    n = 0;
    for img in os.listdir(folder_dir):
        # check if the image ends with jpg
        if img.endswith(".png"):
            image = cv2.imread(folder_dir+'\\'+img)
            if(image is not None):
                if len(image.shape) > 2: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                feature = []    
                h, w = image.shape[0], image.shape[1]
                f_ws = range(2, int(w/2), 2)
                f_hs = range(2, int(h/2), 2)
                
#                 f_ws = [4]
#                 f_hs = [4]

                # extract horizontal features
                for f_w in f_ws:
                    for j in range(h):
                        for i in range(w - f_w + 1):
                            neg = np.sum(image[j, i:int((i+f_w)/2)]).astype(np.int32)
                            pos = np.sum(image[j, int((i+f_w)/2):(i+f_w)]).astype(np.int32)
                            feature.append(pos-neg)

                # extract vertical features
                for f_h in f_hs:
                    for i in range(w):
                        for j in range(h - f_h + 1):
                            neg = np.sum(image[j:int((j+f_h)/2), i]).astype(np.int32)
                            pos = np.sum(image[int((j+f_h)/2):(j+f_h), i]).astype(np.int32)
                            feature.append(pos-neg)
                            
                features.append(feature)              
                labels.append(label)
                n += 1
                if n % 200 == 0:
                    print(n)
                if num_data != -1:
                    if n == num_data:
                        break
    
    features = np.array(features)
    labels = np.array(labels)
#     features = preprocessing.normalize(features, axis = 0)
#     print(np.sum(features[:, 0]**2))

    return features, labels

def find_best_weak_classifier(features, labels, Dts, used_f_idx):
    
    best_classifier = []
    
    min_e = 2
    for f in range(len(features[0])):
        if f not in used_f_idx:
            feature = features[:, f]

            # sort features
            sorted_feature = np.array(sorted(feature))
            sorted_labels = np.array([x for _, x in sorted(zip(feature, labels))])
            sorted_Dts = np.array([x for _, x in sorted(zip(feature, Dts))])

            # calculate values for errors
            pos_dts = sorted_Dts*sorted_labels
            neg_dts = sorted_Dts*(1-sorted_labels)

            SP = np.cumsum(pos_dts)
            SN = np.cumsum(neg_dts)
            TP = np.sum(pos_dts)
            TN = np.sum(neg_dts)

            error_1 = SP + TN - SN
            error_2 = SN + TP - SP

            curr_min_e1 = np.min(error_1)
            curr_min_e2 = np.min(error_2)
            curr_min_e = np.minimum(curr_min_e1, curr_min_e2)

            if curr_min_e < min_e:
                min_e = curr_min_e
                f_idx = f
                if curr_min_e1 < curr_min_e2:
                    polarity = 1
                    threshold = sorted_feature[np.argmin(error_1)]
                    htx = feature > threshold
                else:
                    polarity = -1
                    threshold = sorted_feature[np.argmin(error_2)]
                    htx = feature <= threshold
                
    best_classifier = [f_idx, threshold, polarity, min_e, htx*1]

    return best_classifier
    

def adaboost(all_features, all_true_labels, true_idx, TP_target, FP_target, max_iter):
    features = all_features[true_idx, :]
    true_labels = all_true_labels[true_idx]
    
    N = len(features)
    Dts = (1/N)*np.ones(N)
    
    alphas = []
    classifiers = []
    used_f_idx = []
    TP = 0
    FP_rate = 1
    for t in range(max_iter):
        print(t)
        best_classifier = find_best_weak_classifier(features, true_labels, Dts, used_f_idx)
        classifiers.append(best_classifier)
        
        f_idx, threshold, polarity, min_e, htx = best_classifier
        used_f_idx.append(f_idx)
#         print("f: ", f_idx, " min_e: ", min_e)
        
        # apply all the weak classifiers
        d, TP_rate, pred_labels, reach_TP_target = apply_classifiers_with_targetTP(features, classifiers, alphas, true_labels, TP_target)
#         d, TP_rate, pred_labels, reach_TP_target = apply_classifiers(features, classifiers, alphas, true_labels, TP_target)
#         print("d: ", d)
        
        N = np.sum(true_labels == 0)
        if N == 0:
            break
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        FP_rate = FP / N
        
        print("TP_rate: ", TP_rate)
        print("FP_rate: ", FP_rate)
        
#         if reach_TP_target:     
        if FP_rate < FP_target:
            break

        # update probability distribution     
        # compute confidence parameters
        alpha = math.log((1-min_e) / (min_e + 1e-10))
        alphas.append(alpha)
        
#         numerator = Dts*np.exp(-alpha*true_labels*htx)
        numerator = Dts*np.exp(-2*alpha*(-2*abs(true_labels-htx)+1))
        Dts = numerator / np.sum(numerator)

    return classifiers, alphas, d, FP_rate

def create_cascade(features, labels, num_stages):
    TP_target = 0.95
    FP_target = 0.5
    new_idx = np.arange(features.shape[0])
    cascades = []
    classifications = np.zeros(features.shape[0])
    FPs = []
    FNs = []
    accuracies = []
    for k in range(num_stages):
        classifiers, alphas, d, FP_rate = adaboost(features, labels, new_idx, TP_target, FP_target, 100)
        cascades.append([classifiers, alphas, d, FP_rate])
        new_classifies = apply_classifiers_with_d(features, new_idx, classifiers, alphas, d)
        classifications[new_idx] = new_classifies
        new_idx = new_idx[np.where(new_classifies == 1)[0]]
        print("true_idx: ", len(new_idx))
        
        P = np.sum(labels == 1)
        N = np.sum(labels == 0)
        FP = np.sum(np.logical_and(classifications == 1, labels == 0))
        FN = np.sum(np.logical_and(classifications == 0, labels == 1))
        FP_rate = FP / N
        FN_rate = FN / P
        accuracy = calculate_accuracy(labels, classifications)
        
        FPs.append(FP_rate)
        FNs.append(FN_rate)
        accuracies.append(accuracy)
        print("FP: ", FP_rate, " FN: ", FN_rate, " accuracy: ", accuracy)
        
        ################
#         classifiers, alphas, d, FP_rate = adaboost(features, labels, new_idx, TP_target, FP_target, 10)
#         cascades.append([classifiers, alphas, d, FP_rate])
#         new_classifies = apply_classifiers_with_d(features, new_idx, classifiers, alphas, d)
#         classifications[new_idx] = new_classifies
#         neg_idx = new_idx[np.where(new_classifies == 0)[0]]
#         new_idx = new_idx[np.where(new_classifies == 1)[0]]
        
#         print("true_idx: ", len(new_idx))
#         print("neg_idx: ", len(neg_idx))
        
#         P = np.sum(labels == 1)
#         N = np.sum(labels == 0)
#         FP = np.sum(np.logical_and(classifications == 1, labels == 0))
#         FN = np.sum(np.logical_and(classifications == 0, labels == 1))
#         FP_rate = FP / N
#         FN_rate = FN / P
#         accuracy = calculate_accuracy(labels, classifications)
        
#         FPs.append(FP_rate)
#         FNs.append(FN_rate)
#         accuracies.append(accuracy)
#         print("FP: ", FP_rate, " FN: ", FN_rate, " accuracy: ", accuracy)
    return cascades
            
def test_cascade(features, labels, cascades):
    new_idx = np.arange(features.shape[0])
    classifications = np.zeros(features.shape[0])
    FPs = []
    FNs = []
    accuracies = []
    Ks = []
    K = 0
    for cascade in cascades:
        K += 1
        classifiers, alphas, d, FP_rate = cascade
        new_classifies = apply_classifiers_with_d(features, new_idx, classifiers, alphas, d)
        classifications[new_idx] = new_classifies
        new_idx = new_idx[np.where(new_classifies == 1)[0]]
        
        P = np.sum(labels == 1)
        N = np.sum(labels == 0)
        FP = np.sum(np.logical_and(classifications == 1, labels == 0))
        FN = np.sum(np.logical_and(classifications == 0, labels == 1))
        FP_rate = FP / N
        FN_rate = FN / P
        accuracy = calculate_accuracy(labels, classifications)
        
        FPs.append(FP_rate)
        FNs.append(FN_rate)
        accuracies.append(accuracy)
        Ks.append(K)
    return FPs, FNs, accuracies, Ks
        
        
def apply_classifiers_with_targetTP(features, classifiers, alphas, true_labels, TP_target):
    features = np.array(features)
    sumHs = np.zeros(features.shape[0])
    for classifier, alpha in zip(classifiers, alphas):
        f_idx, threshold, polarity, min_e, htx = classifier
        feature = features[:, f_idx]
        if polarity == 1:
            sumHs += alpha * (feature > threshold)*1
        else:
            sumHs += alpha * (feature <= threshold)*1
    print("zeros 1: ", np.sum(np.logical_and(true_labels == 1, sumHs == 0)))
    reach_target = False
    d = 0
    max_TP_rate = 0
    min_diff = 1
    cand_ds = sumHs
    for cand_d, true_label in zip(cand_ds, true_labels):
        if true_label == 1:
            pred_labels = (sumHs >= cand_d)*1
            P = np.sum(true_labels == 1)
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
            TP_rate = TP / P
#             if abs(TP_rate-TP_target) < min_diff:
#                 min_diff = abs(TP_rate-TP_target)
#                 max_TP_rate = TP_rate
#                 d = cand_d
#                 reach_target = True
            if TP_rate > max_TP_rate:
                max_TP_rate = TP_rate
                d = cand_d
                if TP_rate >= TP_target:     
                    reach_target = True
                
    pred_labels = (sumHs >= d)*1
    print("d: ", d)
    return d, max_TP_rate, pred_labels, reach_target
    
def apply_classifiers_with_d(all_features, true_idx, classifiers, alphas, d):
    
    features = all_features[true_idx, :]

    sumH = np.zeros(features.shape[0])
    for classifier, alpha in zip(classifiers, alphas):
        f_idx, threshold, polarity, min_e, htx= classifier
        feature = features[:, f_idx]
        if polarity == 1:
            sumH += alpha * (feature > threshold)*1
        else:
            sumH += alpha * (feature <= threshold)*1
    
    classifications = (sumH >= d)*1
    return classifications

def apply_classifiers(features, classifiers, alphas, true_labels, TP_target):
    features = np.array(features)
    sumH = np.zeros(features.shape[0])
    reach_TP_target = False
    
    for classifier, alpha in zip(classifiers, alphas):
        f_idx, threshold, polarity, min_e, htx= classifier
        feature = features[:, f_idx]
        if polarity == 1:
            sumH += alpha * (feature > threshold)*1
        else:
            sumH += alpha * (feature <= threshold)*1
            
    sum_alpha = np.sum(alphas)
    d = 0.5*sum_alpha
    pred_labels = (sumH >= d)*1
    
    P = np.sum(true_labels == 1)
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TP_rate = TP / P
    if TP_rate >= TP_target:     
        reach_TP_target = True
        
    return d, TP_rate, pred_labels, reach_TP_target

def calculate_accuracy(gt_labels, pred_labels):
    N = len(pred_labels)
    correct = 0
    for i in range(N):
        if pred_labels[i] == gt_labels[i]:
            correct += 1
    return correct/N

    

    


# In[250]:


if __name__ == '__main__':
    path = Path("C:/Users/yhosc/Desktop/ECE661/HW10/")
    outputPath = Path("C:/Users/yhosc/Desktop/ECE661/HW10")
    
    if os.path.exists(str(path / "train_data.pickle")):
        print("load training data")        
        with open('train_data.pickle', 'rb') as handle:
            train_features, train_labels = pickle.load(handle)
    else:
        print("creating training data...")              
        
        neg_features, neg_labels = extract_features(str(path / "CarDetection" / "train" / "negative"), 0, 50)         
        pos_features, pos_labels = extract_features(str(path / "CarDetection" / "train" / "positive"), 1, 50)
        train_features = np.concatenate((neg_features, pos_features))
        train_labels = np.concatenate((neg_labels, pos_labels))

        with open('train_data.pickle', 'wb') as handle:
            pickle.dump([train_features, train_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("training data saved")
        
    if os.path.exists(str(path / "test_data.pickle")):
        print("load testing data")        
        with open('test_data.pickle', 'rb') as handle:
            test_features, test_labels = pickle.load(handle)
    else:
        print("creating testing data...")              
        neg_features, neg_labels = extract_features(str(path / "CarDetection" / "test" / "negative"), 0, 50)      
        pos_features, pos_labels = extract_features(str(path / "CarDetection" / "test" / "positive"), 1, 50)
        test_features = np.concatenate((neg_features, pos_features))
        test_labels = np.concatenate((neg_labels, pos_labels))

        with open('test_data.pickle', 'wb') as handle:
            pickle.dump([test_features, test_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("testing data saved")
        
    if os.path.exists(str(path / "cascades.pickle")):
        print("load cascades")        
        with open('cascades.pickle', 'rb') as handle:
            cascades = pickle.load(handle)
    else:
        cascades = create_cascade(train_features, train_labels, 10)
        with open('cascades.pickle', 'wb') as handle:
            pickle.dump(cascades, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    FPs, FNs, accuracies, Ks = test_cascade(test_features, test_labels, cascades)
    print(FPs, FNs, accuracies, Ks)


# In[251]:


plt.plot(Ks, FPs, Ks, FNs)
plt.xticks(Ks)
plt.xlabel('K')
plt.legend(["FP", "FN"], loc ="lower right")
plt.savefig('adaboost.png')
plt.show()

