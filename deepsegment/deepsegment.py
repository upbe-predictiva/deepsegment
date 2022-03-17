import glob
import logging
import os
import pickle

import numpy as np
from seqtag_keras.wrapper import Sequence

is_tfserving_installed = True

try:
    import grpc
    import tensorflow as tf
    from tensorflow.python.saved_model import signature_constants
    from tensorflow_serving.apis import (predict_pb2,
                                         prediction_service_pb2_grpc)
except Exception as ex:
    is_tfserving_installed = False
    logging.warn("Tensorflow serving is not installed. Cannot be used with tesnorflow serving docker images.")
    logging.warn("Run pip install tensorflow-serving-api==1.12.0 if you want to use with tf serving.")


def chunk(l, n):
    """
    Chunk a list l into chunks of equal size n

    Parameters:
    l (list): List (of any items) to be chunked.
    n (int): size of each chunk.

    Returns:
    list: Return list os lists (chunks)

    """
    chunked_l = []
    for i in range(0, len(l), n):
        chunked_l.append(l[i:i + n])

    if not chunked_l:
        chunked_l = [l]

    return chunked_l


def predict_response_to_array(response, output_tensor_name):
    """
    Convert response from tf-serving to np array (keras model.predict format)
    """
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    return np.reshape(response.outputs[output_tensor_name].float_val, shape)


def get_tf_serving_respone(seqtag_model, x):
    """
    Make GRPC call to tfserving server and read it's output.

    """
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = seqtag_model
    request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs["word-input"].CopyFrom(tf.contrib.util.make_tensor_proto(x[0], dtype="int32", shape=None))
    request.inputs["char-input"].CopyFrom(tf.contrib.util.make_tensor_proto(x[1], dtype="int32", shape=None))
    response = stub.Predict(request, 20)
    preds = predict_response_to_array(response, "prediction")
    preds = [np.argmax(_tags, axis=1).tolist() for _tags in preds]
    return preds


class DeepSegment():
    seqtag_model = None
    data_converter = None

    def __init__(self, weights_path=None, params_path=None, utils_path=None, tf_serving=False, checkpoint_name=None):
        """
        Initialize deepsegment

        Parameters:

        lang_code (str): Name or code of the language. (default is english)

        checkpoint_path (str): If using with custom models, pass the custom model checkpoint path and set lang_code=None

        params_path (str): See checkpoint_path.

        utils_path (str): See checkpoint_path.

        tf_serving (bool): If using with tf_serving docker image, set to True.

        checkpoint_name (str): If using with finetuned models use this.

        """
        # if not tf_serving:
        #     self.seqtag_model = model_from_json(open(params_path).read(), custom_objects={
        #                                         'CRFModelWrapper': CRFModelWrapper})
        #     self.seqtag_model.load_weights(checkpoint_path)

        # elif tf_serving:
        #     if not is_tfserving_installed:
        #         raise RuntimeError(
        #             "Tensorflow serving is not installed. Cannot be used with tesnorflow serving docker images.")
        #     self.seqtag_model = 'deepsegment_model'

        # self.data_converter = pickle.load(open(utils_path, 'rb'))
        self.data_converter, self.seqtag_model = Sequence.load(
            weights_file=weights_path,
            preprocessor_file=utils_path,
            params_file=params_path).get_transformer_and_model()

    def segment(self, sents, batch_size=32):
        """
        segment a list of sentences or single sentence

        Parameters:
        sents (list or str): List (or single) of sentences to be segmented.

        Returns:
        list: Return list or list of lists of segmented sentenes.

        """
        if not self.seqtag_model:
            print('Please load the model first')

        string_output = False
        if isinstance(sents, str):
            logging.warn("Batch input strings for faster inference.")
            string_output = True
            sents = [sents]

        sents = [sent.strip().split() for sent in sents]

        max_len = len(max(sents, key=len))
        if max_len >= 40:
            logging.warn("Consider using segment_long for longer sentences.")

        encoded_sents = self.data_converter.transform(sents)

        if not isinstance(self.seqtag_model, type('')):
            all_tags = self.seqtag_model.predict(encoded_sents, batch_size=batch_size)

        else:
            all_tags = get_tf_serving_respone(self.seqtag_model, encoded_sents)

        segmented_sentences = [[] for _ in sents]
        for sent_index, (sent, tags) in enumerate(zip(sents, all_tags)):
            segmented_sent = []
            for i, (word, tag) in enumerate(zip(sent, tags)):
                if tag == 2 and i > 0 and segmented_sent:
                    segmented_sent = ' '.join(segmented_sent)
                    segmented_sentences[sent_index].append(segmented_sent)
                    segmented_sent = []

                segmented_sent.append(word)
            if segmented_sent:
                segmented_sentences[sent_index].append(' '.join(segmented_sent))

        if string_output:
            return segmented_sentences[0]

        return segmented_sentences

    def segment_long(self, sent, n_window=None):
        """
        Segment a longer text

        Parameters:
        sent (str): Input text.
        n_window (int): window size (words) for iterative segmentation.

        Returns:
        list: Return list of sentences.
        """
        if not n_window:
            logging.warn("Using default n_window=10. Set this parameter based on your data.")
            n_window = 10

        if isinstance(sent, list):
            logging.error("segment_long doesn't support batching as of now. Batching will be added in a future release.")
            return None

        segmented = []
        sent = sent.split()
        prefix = []
        while sent:
            current_n_window = n_window - len(prefix)
            if current_n_window <= 0:
                current_n_window = n_window

            window = prefix + sent[:current_n_window]
            sent = sent[current_n_window:]
            segmented_window = self.segment([' '.join(window)])[0]
            segmented += segmented_window[:-1]
            prefix = segmented_window[-1].split()

        if prefix:
            segmented.append(' '.join(prefix))

        return segmented
