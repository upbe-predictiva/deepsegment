import datetime
import os
import pickle
import random
import re
import string

import tensorflow as tf
from progressbar import progressbar
from seqeval.metrics import f1_score
from seqtag_keras.models import BiLSTMCRF
from seqtag_keras.trainer import Trainer
from seqtag_keras.utils import load_glove
from seqtag_keras.wrapper import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
except:
    pass


def bad_sentence_generator(sent, remove_punctuation=None):
    """
        Returns sentence with completely/ partially removed punctuation.

        Parameters:
        sent (str): Sentence on which the punctuation removal operation is performed.

        remove_punctuation (int): removing punctuation completely if remove_punctuation ==0 or ==1, removing punctuation till a randomly selected point if remove_punctuation ==2

        Returns:
        str: Sentence with modified punctuation

    """

    if not remove_punctuation:
        remove_punctuation = random.randint(0, 3)

    break_point = random.randint(1, len(sent)-2)
    lower_case = random.randint(0, 2)

    if remove_punctuation <= 1:
        # removing punctuation completely if remove_punctuation ==0 or ==1
        sent = re.sub('['+string.punctuation+']', '', sent)

    elif remove_punctuation == 2:
        # removing punctuation till a randomly selected point if remove_punctuation ==2
        if random.randint(0, 1) == 0:
            sent = re.sub('['+string.punctuation+']', '', sent[:break_point]) + sent[break_point:]
        # removing punctuation after a randomly selected point if remove_punctuation ==2
        else:
            sent = sent[:break_point] + re.sub('['+string.punctuation+']', '', sent[break_point:])

    if lower_case <= 1:
        # lower casing sentence
        sent = sent.lower()

    return sent


def generate_data(lines, max_sents_per_example=6, n_examples=1000):
    """
        Generates training data for deepsegment from list of sentences.

        Parameters:
        lines (list): Base sentences for data generation.

        max_sents_per_example (int): Maximum number of sentences to be combined to form a single paragraph.

        n_examples (int): Number of training examples to be generated.

        Returns:
        list, list: Training data and corresponding labels in BIOU format.

    """
    x, y = [], []

    for current_i in progressbar(range(n_examples)):
        x.append([])
        y.append([])

        chosen_lines = []
        for _ in range(random.randint(1, max_sents_per_example)):
            chosen_lines.append(random.choice(lines))

        chosen_lines = [bad_sentence_generator(line, remove_punctuation=random.randint(0, 3)) for line in chosen_lines]

        for line in chosen_lines:
            words = line.strip().split()
            for word_i, word in enumerate(words):
                x[-1].append(word)
                label = 'O'
                if word_i == 0:
                    label = 'B-sent'
                y[-1].append(label)

    return x, y


def train(x, y, vx, vy, epochs, batch_size, save_folder, glove_path):
    """
        Trains a deepsegment model.

        Parameters:
        x (list): x generated from generate_data

        y (list): y generated from generate_data

        vx (list): x generated from generate_data

        vy (list): y generated from generate_data


        epochs (int): Max number of epochs.

        batch_size (int): batch_size

        save_folder (str): path for the directory where checkpoints should be saved.

        glove_path (str): path to 100d word vectors.


    """

    embeddings = load_glove(glove_path)

    checkpoint_path = os.path.join(save_folder, 'checkpoint')
    final_weights_path = os.path.join(save_folder, 'weights', 'final_weights')
    params_path = os.path.join(save_folder, 'params')
    utils_path = os.path.join(save_folder, 'utils')

    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, mode='max', monitor='f1')
    earlystop = EarlyStopping(patience=3, monitor='f1', mode='max')

    model = Sequence(embeddings=embeddings)

    model.fit(x, y, x_valid=vx, y_valid=vy, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, earlystop])

    model.save(weights_file=final_weights_path, params_file=params_path, preprocessor_file=utils_path)


lang_code_mapping = {
    'english': 'en',
    'french': 'fr',
    'italian': 'it'
}


def finetune(lang_code, x, y, vx, vy, name=None, epochs=5, batch_size=16, lr=0.0001):
    """
        Finetunes an existing deepsegment model.

        Parameters:
        x (list): x generated from generate_data

        y (list): y generated from generate_data

        vx (list): x generated from generate_data

        vy (list): y generated from generate_data

        name (str): name of the fintuned checkpoint. (params, utils will be same for finetuned and base models.)


        epochs (int): Max number of epochs.

        batch_size (int): batch_size

        lr (float): initial learning rate.

    """

    if not name:
        name = str(datetime.datetime.now()).split()
        name = '-'.join(name)
        print('Name not provided. The checkpoint will be named checkpoint_' + name)

    if lang_code in lang_code_mapping:
        lang_code = lang_code_mapping[lang_code]

    home = os.path.expanduser("~")
    lang_path = os.path.join(home, '.DeepSegment_' + lang_code)
    checkpoint_path = os.path.join(lang_path, 'checkpoint')
    utils_path = os.path.join(lang_path, 'utils')
    params_path = os.path.join(lang_path, 'params')

    p = pickle.load(open(utils_path, 'rb'))

    model = BiLSTMCRF(char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      word_embedding_dim=100,
                      char_embedding_dim=25,
                      word_lstm_size=100,
                      char_lstm_size=25,
                      fc_dim=100,
                      dropout=0.2,
                      embeddings=None,
                      use_char=True,
                      use_crf=True)

    model = model.build()
    model.compile(optimizer=Adam(learning_rate=lr))

    model.load_weights(checkpoint_path)

    temp_vx = p.transform(vx)
    lengths = map(len, vy)
    y_pred = model.predict(temp_vx)
    y_pred = p.inverse_transform(y_pred, lengths)
    orig_score = f1_score(vy, y_pred)
    print('Scores before finetuning: ')
    print(orig_score)
    temp_vx = None
    del temp_vx

    trainer = Trainer(model, preprocessor=p)

    checkpoint_path = checkpoint_path + '_' + name
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, mode='max', monitor='f1')
    earlystop = EarlyStopping(patience=3, monitor='f1', mode='max')

    trainer.train(x, y, vx, vy,
                  epochs=epochs, batch_size=batch_size,
                  verbose=1, callbacks=[checkpoint, earlystop],
                  shuffle=True)
