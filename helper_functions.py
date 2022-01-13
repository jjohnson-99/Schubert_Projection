import numpy as np
import matplotlib.pyplot as plt
import time

from itertools import permutations

def dUU(U_1, U_2, r):
  u,s,vt = np.linalg.svd(U_1.T @ U_2)

  for i in range(len(s)):
    if s[i] - 1 > 1e-5:
      raise Exception('s[',i,'] = ', s[i])
    elif s[i] > 1:
      s[i] = 1

  d = sum([np.arccos(s[i])**2 for i in range(r)])

  #print(u,s,vt)
  assert d >= 0
  return d

 #@title Default title text
def evaluate(predict, truth, cluster):
  labels = [i for i in range(cluster)]
  p = permutations(labels)

  predict = np.array(predict)
  truth = np.array(truth)
  assert predict.shape == truth.shape

  err = 1
  for permuted_label in p:
    #print("Permutation:", permuted_label)
    new_predict = np.zeros(len(predict), dtype = int)

    for i in range(len(labels)):
      new_predict[predict == labels[i]] = int(permuted_label[i])

    err_temp = np.sum(new_predict != truth) / len(predict)

    #print('predict:', new_predict)
    #print('truth:', truth)

    err = min(err, err_temp)
    #print("Error Rate:", err_temp)

  return err
