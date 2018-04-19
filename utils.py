import numpy as np
import itertools
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torch.autograd import Variable


# using yaml instaed of pickle for compatibility reasons
# for yaml usage check: https://stackoverflow.com/a/36096428/4078182
def load_office(source_name, target_name, data_folder, feature_type):
    if data_folder is None:
        data_folder = './pkl/'
    source_file = data_folder + source_name + '_' + feature_type + '.yml'
    target_file = data_folder + target_name + '_' + feature_type + '.yml'
    with open(source_file, 'r') as f:
        source = yaml.load(f)
    with open(target_file, 'r') as f:
        target = yaml.load(f)
    xs = source['train']
    ys = source['train_labels']
    xt = target['train']
    yt = target['train_labels']
    xt_test = target['test']
    yt_test = target['test_labels']
    return xs, ys, xt, yt, xt_test, yt_test

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=50, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = (int) (batch_count * batch_size)
        end = (int) (start + batch_size)
        batch_count += 1
        yield [d[start:end] for d in data]

def weight_init(x):
    for y in x.modules():
        if isinstance(y, nn.Linear):
            nn.init.xavier_normal(y.weight.data)
            # nn.init.kaiming_normal(y.weight.data)
            nn.init.constant(y.bias.data, 0.1)

def one_hot_encode(labels, num_classes):
    ones = torch.sparse.torch.eye(num_classes)
    if labels.is_cuda:
        ones = ones.cuda()
    return ones.index_select(0, labels.data)


class L2_Loss(nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
    def forward(self, x):
        return torch.sum(x ** 2) / 2.
    
def adjust_learning_rate(optimizer=None, p=None, alpha=10, beta=0.75, n_0=0.01):
    '''
    Update learning rate according to schedule given in Multi-Adversarial Domain Adaptation paper.
    '''
    n_p = n_0 / np.power(1. + alpha * p, beta)
    print("new lr:", n_p)
    for param_group in optimizer.param_groups:
        param_group['lr'] = n_p
    
class softmax_cross_entropy_with_logits(nn.Module):
    def __init__(self, num_classes=None):
        super(softmax_cross_entropy_with_logits, self).__init__()
        assert num_classes!=None, "Provide number of classes"
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
    def forward(self, logits=None, labels=None):
        ones = torch.sparse.torch.eye(self.num_classes)
        if labels.is_cuda:
            ones = ones.cuda()
        ones = Variable(ones, requires_grad=False)
        labels_one_hot = torch.index_select(ones, 0, labels)
        logits_softmax = self.softmax(logits)
        return (-torch.sum(labels_one_hot * torch.log(logits_softmax), dim=1)).mean()
        