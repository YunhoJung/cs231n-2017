import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    loss_contributors_cnt = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0: # 다를 때?
        loss += margin
        
        # incorrect class gradient
        dW[:,j] += X[i]
        
        # count contributor terms to loss function ?
        loss_contributors_cnt += 1
    # correct class gradient ?
    dW[:, y[i]] += (-1) * loss_contributors_cnt * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization to the gradient
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # scores : a numpy array(N,C) containing scores
#   scores = X.dot(W)
  s = X.dot(W)
  
  
  # Read correct scores into a column array of height N
#   correct_score = scores[list(range(num_train)), y].reshape(num_train, -1)
#   correct_score = scores[np.arange(num_train), y].reshape(num_train, -1)
  correct_score = s[list(range(num_train)), y]
  correct_score = correct_score.reshape(num_train, -1)

  #? Substract correct scores from score matrix as a column vector from every cell
#   scores_diff = scores - correct_score
#   scores += 1 - correct_score
  s += 1 - correct_score
  
  # Add margin
#   scores_diff += 1
  
  #? for correct scores not to contribute to loss function
#   scores_diff[np.arange(num_train), y] = 0
#   scores[np.arange(num_train), y] = 0
  s[list(range(num_train)), y] = 0
    
  # loss function
#   loss = np.sum(np.fmax(scores_diff,0))
#   loss = np.sum(np.fmax(scores, 0))
#   loss /= num_train
#   loss += reg * np.sum(W * W)
  loss = np.sum(np.fmax(s, 0)) / num_train
  print(reg)
  print(np.sum(W*W))
  loss += reg * np.sum(W * W)

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
# #   X_mask = np.zeros(scores_diff.shape)
#   X_mask = np.zeros(scores.shape)
# #   X_mask[scores_diff > 0] = 1
#   X_mask[scores > 0] = 1
#   X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
# #   scores_diff[scores_diff > 0] = 1
# #   correct_label_vals = scores_diff.sum(axis=1) * -1
# #   scores_diff[np.arange(num_train), y] = correct_label_vals
    
#   dW = X.T.dot(X_mask)
#   dW /= num_train
#   dW += 2 * reg * W

  X_mask = np.zeros(s.shape)
  X_mask[s > 0] = 1
  X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
  dW = X.T.dot(X_mask)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
