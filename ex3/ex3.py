# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
from tqdm.notebook import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

LOSS = "loss"
ACCURACY = "accuracy"

BATCH_SIZE = 64
N_EPOCHS = 20
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    average = np.zeros(embedding_dim)
    count = 0
    for word in sent.text:
        if word in word_to_vec:
            average = np.add(average, word_to_vec[word])
            count += 1
    if count == 0:
        count = 1
    return torch.Tensor(average / count)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    hot_vector = np.zeros(size)
    hot_vector[ind] = 1
    return hot_vector


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    average = np.zeros(len(word_to_ind))
    count = 0
    for word in sent.text:
        count = count + 1
        average = np.add(average, get_one_hot(len(word_to_ind), word_to_ind[word]))
    return torch.Tensor(average / count)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words = dict()
    i = 0
    for word in words_list:
        if word not in words:
            words[word] = i
            i = i + 1
    return words


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embeddings = np.zeros((seq_len, embedding_dim))
    count = 0
    for word in sent.text:
        if count < seq_len:
            if word in word_to_vec:
                embeddings[count] = word_to_vec[word]
            count += 1
        else:
            break
    return embeddings


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True,
                 dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
                               for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if n_layers > 1:
            self.model = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout,
                                 bidirectional=True,
                                 batch_first=True)
        else:
            self.model = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=0, bidirectional=True,
                                 batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def lstm_output(self, text):
        output, (hn, cn) = self.model(text.float())
        vec = hn.view(self.n_layers, 2, text.size(0), self.hidden_dim)
        vec = torch.cat((vec[:, 0, :, :], vec[:, 1, :, :]), 0)[-1]
        return vec

    def forward(self, text):
        vec = self.lstm_output(text)
        if self.n_layers > 1:
            vec = self.dropout(vec)
        return self.linear(vec)

    def predict(self, text):
        with torch.no_grad():
            vec = self.lstm_output(text)
            vec = self.linear(vec)
            return torch.round(self.activation(vec))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            x = self.linear(x)
            x = self.activation(x)
            return torch.round(x)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return torch.sum((preds == y).float()) / preds.shape[0]


def train_epoch(model, data_iterator, optimizer, criterion, epoch):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    device = get_available_device()
    model = model.to(device)
    model.train()
    with tqdm(data_iterator, leave=False) as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.to(device), target.view((target.shape[0], 1)).to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, target).to(device)
            accuracy = binary_accuracy(model.predict(data), target).item()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), acc=accuracy)

    return evaluate(model, data_iterator, criterion)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """

    def weighted_average(result_list):
        values, weights = zip(*result_list)
        return np.average(values, weights=weights)

    device = get_available_device()
    model = model.to(device)
    model.eval()

    test_loss, test_accuracy = [], []

    with torch.no_grad():
        for local_batch, local_labels in data_iterator:
            local_batch = local_batch.to(device)
            local_labels = local_labels.view((local_labels.shape[0], 1)).to(device)
            predictions = model(local_batch)
            loss = criterion(predictions, local_labels).to(device)
            accuracy = binary_accuracy(model.predict(local_batch), local_labels)
            test_loss += [(loss, len(local_labels))]
            test_accuracy += [(accuracy, len(local_labels))]

        return weighted_average(test_loss), weighted_average(test_accuracy)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    device = get_available_device()
    model = model.to(device)
    model.eval()

    sigm = nn.Sigmoid()
    output, logits = [], []

    with torch.no_grad():
        for local_batch, local_labels in data_iter:
            predictions = model(local_batch.to(device))
            logits += predictions
            output += torch.round(sigm(predictions))

        return torch.tensor(output, dtype=torch.float32), \
               torch.tensor(logits, dtype=torch.float32)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = F.binary_cross_entropy_with_logits
    train_results, validation_results = {LOSS: [], ACCURACY: []}, {LOSS: [], ACCURACY: []}

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}: ", end=' ')
        train_iter = data_manager.get_torch_iterator(data_subset=TRAIN)
        train_loss, train_accuracy = train_epoch(model, train_iter, optimizer, criterion, epoch)
        train_results[LOSS].append(train_loss)
        train_results[ACCURACY].append(train_accuracy)
        print(f"train_loss: {train_loss}, train_accuracy: {train_accuracy}", end=', ')

        validation_iter = data_manager.get_torch_iterator(data_subset=VAL)
        validation_loss, validation_accuracy = evaluate(model, validation_iter, criterion)
        validation_results[LOSS].append(validation_loss)
        validation_results[ACCURACY].append(validation_accuracy)
        print(f"validation_loss: {validation_loss}, validation_accuracy: {validation_accuracy}")

    return train_results, validation_results


def best_model_results(model, data_manager):
    print(f"{'-' * 16}\n| Test results |\n{'-' * 16}")
    results = []
    test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
    predictions, logits = get_predictions_for_data(model, test_iterator)
    actual_labels = torch.tensor(data_manager.get_labels(TEST), dtype=torch.float32)
    whole_set_loss = F.binary_cross_entropy_with_logits(logits, actual_labels).item()
    whole_set_accuracy = binary_accuracy(predictions, actual_labels).item()
    results += [(whole_set_loss, whole_set_accuracy)]
    print(f"Loss on Entire Test Set: {whole_set_loss}")
    print(f"Accuracy on Entire Test: {whole_set_accuracy}")

    test_set = data_manager.torch_datasets[TEST]
    negated_polarity = data_loader.get_negated_polarity_examples(test_set.data)
    rare_words = data_loader.get_rare_words_examples(test_set.data,
                                                     data_manager.sentiment_dataset)
    special_subsets = [negated_polarity, rare_words]
    for i in range(2):
        name = "Negated Polarity" if i == 0 else "Rare Words"
        subset_preds, subset_logits = predictions[special_subsets[i]], logits[special_subsets[i]]
        subset_labels = actual_labels[special_subsets[i]]
        subset_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_labels).item()
        subset_accuracy = binary_accuracy(subset_preds, subset_labels).item()
        results += [(subset_loss, subset_accuracy)]
        print(f"Loss on {name}: {subset_loss}")
        print(f"Accuracy on {name}: {subset_accuracy}")


def plot_results(model_title, results):
    train_results, validation_results = results
    for idx, function in enumerate([LOSS, ACCURACY], 1):
        plt.subplot(210 + idx)
        n_epochs = np.arange(len(train_results[function]))
        # n_epochs = np.arange(N_EPOCHS)
        plt.plot(n_epochs, train_results[function], 'b--', label='Train')
        plt.plot(n_epochs, validation_results[function], 'r:', label='Validation')
        if idx == 2:
            plt.xlabel('Epoch', fontname='Georgia')
        plt.ylabel(function.title(), fontname='Georgia')
        # plt.xlim([0, 20])
        plt.xticks(np.arange(0, 25, 5))
        plt.ylim([0, plt.ylim()[1] + 0.1]) if function == LOSS else plt.ylim([0, 1])
        plt.legend()

    plt.suptitle(model_title, fontname='Georgia')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.7)
    plt.show()


def print_title(title: str):
    width = len(title) + 6
    print(f"{'#' * width}\n#  {title}  #\n{'#' * width}")


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    title = "log_linear_with_one_hot"
    print_title(title)
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    model = LogLinear(data_manager.get_input_shape()[0]).to(get_available_device())
    train_results = train_model(model, data_manager, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
    plot_results(title, train_results)
    best_model_results(model, data_manager)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    title = "log_linear_with_w2v"
    print_title(title)
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(data_manager.get_input_shape()[0]).to(get_available_device())
    train_results = train_model(model, data_manager, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
    plot_results(title, train_results)
    best_model_results(model, data_manager)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    title = "lstm_with_w2v"
    print_title(title)
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=BATCH_SIZE,
                               embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100,
                 n_layers=1, dropout=0.5).to(get_available_device())
    train_results = train_model(model, data_manager, n_epochs=4, lr=0.001,
                                weight_decay=WEIGHT_DECAY)
    plot_results(title, train_results)
    best_model_results(model, data_manager)


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    train_lstm_with_w2v()
