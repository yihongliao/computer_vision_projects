#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import torch
from torch import nn, optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[2]:


def load_dataset(folder_dir):
    # get the path/directory
    images = []
    labels = []
    i = 0;
    for img in os.listdir(folder_dir):
        # check if the image ends with jpg
        if img.endswith(".png"):
            image = cv2.imread(folder_dir+'\\'+img)
            
            if(image is not None):
                label = img.split('_')[0]
                if len(image.shape) > 2: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_vec = image.flatten()
                
                # normalize images
                image_vec = image_vec / np.linalg.norm(image_vec)
                
                images.append(image_vec)
                labels.append(label)
#                 i = i+1
#                 if i % 20 == 0:
#                     print(i)
    return np.transpose(np.stack(images)), labels

def PCA(images, p):
    N = images.shape[1]
    m = np.mean(images, axis=1)
    xm = np.transpose(np.transpose(images) - m)
    CT = (1/N)*np.transpose(xm).dot(xm)
    
    w, v = np.linalg.eig(CT)
    sorted_v = np.transpose([vp for _, vp in sorted(zip(w , v), key = lambda wv: wv[0], reverse = True)])
    
    eig_vecs = xm.dot(sorted_v)
    eig_vecs = eig_vecs / np.linalg.norm(eig_vecs)

    return eig_vecs[:, 0:p], m

def LDA(images, labels, p):
    n = images.shape[0]
    
    # global mean
    m = np.mean(images, axis=1)
    
    # in class mean
    mi = []
    SW_vec = []
    unique_labels = np.unique(labels)
    C = unique_labels.shape[0]
    for unique_label in unique_labels:
        sum_image = np.zeros(n)
        ci = 0
        for i, label in enumerate(labels):
            if label == unique_label:
                ci += 1
                sum_image += images[:, i]
        mi_tmp = sum_image / ci
        for i, label in enumerate(labels):
            if label == unique_label:
                SW_vec.append(images[:, i] - mi_tmp)
        mi.append(mi_tmp)
    SW_vec = np.transpose(SW_vec)
    mi = np.transpose(np.stack(mi))
    
    
    xm = np.transpose(np.transpose(mi) - m)
    SBT = (1/C)*np.transpose(xm).dot(xm)
    
    w, v = np.linalg.eig(SBT)
    sorted_w = sorted(w , reverse = True)
    sorted_v = np.transpose([vp for _, vp in sorted(zip(w , v), key = lambda wv: wv[0], reverse = True)])
    
    eig_vecs0 = xm.dot(sorted_v)
    eig_vecs0 = eig_vecs0 / np.linalg.norm(eig_vecs0)

    
    Y = eig_vecs0[:, :-1]
    DB = np.eye(C-1) * sorted_w[:-1]
    Z = Y.dot(np.sqrt(np.linalg.inv(DB)))
    
    ZT_SW_Z = np.dot(np.dot(np.transpose(Z), SW_vec), np.transpose(np.dot(np.transpose(Z), SW_vec)))
    
    w, v = np.linalg.eig(ZT_SW_Z)
    sorted_w = sorted(w , reverse = True)
    sorted_v = np.transpose([vp for _, vp in sorted(zip(w , v), key = lambda wv: wv[0], reverse = True)])
    
    eig_vecs1 = Z.dot(sorted_v)
    eig_vecs1 = eig_vecs1 / np.linalg.norm(eig_vecs1)

    return eig_vecs1[:, 1:p+1], m
    


def project_to_subspace(ws, images, m):
    images = np.transpose(np.transpose(images) - m)
    return np.transpose(ws).dot(images)

def predict_labels(images, trainImgs, trainLabels, ws, m):
    N = images.shape[1]
    trainVecs = np.transpose(project_to_subspace(ws, trainImgs, m))
    testVecs = project_to_subspace(ws, images, m)
    predicted_labels = []
    for i in range(N):
        testVec = testVecs[:, i]
        dists = np.sqrt(np.sum((trainVecs - testVec) ** 2, 1))
        predicted_label = trainLabels[np.argmin(dists)]
        predicted_labels.append(predicted_label)
    return predicted_labels

def calculate_accuracy(gt_labels, pred_labels):
    N = len(pred_labels)
    correct = 0
    for i in range(N):
        if pred_labels[i] == gt_labels[i]:
            correct += 1
    return correct/N
    


# In[3]:


class DataBuilder(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_list = [f for f in os.listdir(path) if f.endswith('.png')]
        self.label_list = [int(f.split('_')[0]) for f in self.image_list]
        self.len = len(self.image_list)
        self.aug = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        fn = os.path.join(self.path, self.image_list[index])
        x = Image.open(fn).convert('RGB')
        x = self.aug(x)
        return {'x': x, 'y': self.label_list[index]}

    def __len__(self):
        return self.len


class Autoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, encoded_space_dim * 2)
        )
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 4 * 4 * 64),
            nn.LeakyReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu, logvar = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
        return mu, logvar

    def decode(self, z):
        x = self.decoder_lin(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


class VaeLoss(nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, xhat, x, mu, logvar):
        loss_MSE = self.mse_loss(xhat, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD


def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(trainloader):
        optimizer.zero_grad()
        mu, logvar = model.encode(data['x'])
        z = model.reparameterize(mu, logvar)
        xhat = model.decode(z)
        loss = vae_loss(xhat, data['x'], mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainloader.dataset)))
    
def predict_labels_autoenc(X_test, X_train, y_train):
    N = X_test.shape[0]

    predicted_labels = []
    for i in range(N):
        testVec = X_test[i, :]
        dists = np.sqrt(np.sum((X_train - testVec) ** 2, 1))
        predicted_label = trainLabels[np.argmin(dists)]
        predicted_labels.append(predicted_label)
    return predicted_labels


# In[4]:


if __name__ == '__main__':
    path = Path("C:/Users/yhosc/Desktop/ECE661/HW10/")
    outputPath = Path("C:/Users/yhosc/Desktop/ECE661/HW10")
    
    trainImgs, trainLabels = load_dataset(str(path / "FaceRecognition" / "train"))
    testImgs, testLabels = load_dataset(str(path / "FaceRecognition" / "test"))
    
    ps = [3, 8, 16]
    accuracies_PCA = []
    accuracies_LDA = []
    
    for p in ps:
        ws, m = PCA(trainImgs, p)
        predicted_labels = predict_labels(testImgs, trainImgs, trainLabels, ws, m)
        accuracy = calculate_accuracy(testLabels, predicted_labels)
        accuracies_PCA.append(accuracy)
        
        ws, m =  LDA(trainImgs, trainLabels, p)
        predicted_labels = predict_labels(testImgs, trainImgs, trainLabels, ws, m)
        accuracy = calculate_accuracy(testLabels, predicted_labels)
        accuracies_LDA.append(accuracy)
        
    print(accuracies_PCA)
    print(accuracies_LDA)


# In[5]:


# Change these
ps = [3, 8, 16]
training = False
TRAIN_DATA_PATH = str(path / "FaceRecognition" / "train")
EVAL_DATA_PATH = str(path / "FaceRecognition" / "test")
OUT_PATH = str(path)

accuracies_autoenc = []
for p in ps:
    LOAD_PATH = str(path) + f'/weights/model_{p}.pt'
    model = Autoencoder(p)

    trainloader = DataLoader(
        dataset=DataBuilder(TRAIN_DATA_PATH),
        batch_size=1,
    )
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()

    X_train, y_train = [], []
    for batch_idx, data in enumerate(trainloader):
        mu, logvar = model.encode(data['x'])
        z = mu.detach().cpu().numpy().flatten()
        X_train.append(z)
        y_train.append(data['y'].item())
    X_train = np.stack(X_train)
    y_train = np.array(y_train)

    testloader = DataLoader(
        dataset=DataBuilder(EVAL_DATA_PATH),
        batch_size=1,
    )
    X_test, y_test = [], []
    for batch_idx, data in enumerate(testloader):
        mu, logvar = model.encode(data['x'])
        z = mu.detach().cpu().numpy().flatten()
        X_test.append(z)
        y_test.append(data['y'].item())
    X_test = np.stack(X_test)
    y_test = np.array(y_test)
    
    predicted_labels = predict_labels_autoenc(X_test, X_train, y_train)
    accuracy = calculate_accuracy(testLabels, predicted_labels)
    accuracies_autoenc.append(accuracy)
print(accuracies_autoenc)


# In[10]:


plt.plot(ps, accuracies_PCA, ps, accuracies_LDA, ps, accuracies_autoenc)
plt.xticks(ps)
plt.xlabel('p')
plt.ylabel('accuracy')
plt.legend(["PCA", "LDA", "Autoencoder"], loc ="lower right")
plt.savefig('classification.png')
plt.show()

