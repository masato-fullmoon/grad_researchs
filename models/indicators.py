from functools import partial
from keras import backend as K
import tensorflow as tf

class LearningIndicators(object):
    def __init__(self, modeltype='classify', num_classes=None):
        assert modeltype in ('classify','regression','autoencoder'),\
                'Mismatch model type.'
        if (modeltype == 'classify') and (num_classes is not None):
            self.class_label = num_classes
            self.metrics = self.classify_metrics()
        elif modeltype == 'regression':
            self.metrics = self.regression_metrics()
        elif modeltype == 'autoencoder':
            self.metrics = self.ae_metrics()

    def classify_metrics(self):
        metrics = ["accuracy"]

        # the metrics a class label
        func_list = [self.__class_accuracy, self.__class_precision,
                self.__class_recall, self.__class_f_measure]
        name_list = ["acc", "precision", "recall", "f_measure"]
        for i in range(self.class_label):
            for func, name in zip(func_list, name_list):
                func = partial(func, i)
                func.__name__ = "{}-{}".format(name, i)
                metrics.append(func)

        # total metrics
        metrics.append(self.__average_accuracy)
        metrics.append(self.__macro_precision)
        metrics.append(self.__macro_recall)
        metrics.append(self.__macro_f_measure)

        return metrics

    def regression_metrics(self):
        return ['accuracy',
                'mae',
                self.__rms,
                self.__r2,
                self.__rmse,
                self.__mce]

    def ae_metrics(self):
        return ['accuracy',
                self.__dice_coef,
                self.__jaccard_index,
                self.__overlap_coef]

    # ---------- classfication indicators
    def __normalize_y_pred(self, y_pred):
        return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])

    def __class_true_positive(self, class_label, y_true, y_pred):
        y_pred = self.__normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true[:, class_label]+y_pred[:, class_label],2), K.floatx())

    def __class_accuracy(self, class_label, y_true, y_pred):
        y_pred = self.__normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true[:, class_label], y_pred[:, class_label]), K.floatx())

    def __class_precision(self, class_label, y_true, y_pred):
        y_pred = self.__normalize_y_pred(y_pred)

        return K.sum(self.__class_true_positive(class_label, y_true, y_pred))/\
                (K.sum(y_pred[:, class_label])+K.epsilon())

    def __class_recall(self, class_label, y_true, y_pred):
        return K.sum(self.__class_true_positive(class_label, y_true, y_pred))/\
                (K.sum(y_true[:, class_label])+K.epsilon())

    def __class_f_measure(self, class_label, y_true, y_pred):
        precision = self.__class_precision(class_label, y_true, y_pred)
        recall = self.__class_recall(class_label, y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    def __true_positive(self, y_true, y_pred):
        y_pred = self.__normalize_y_pred(y_pred)

        return K.cast(K.equal(y_true+y_pred, 2), K.floatx())

    def __micro_precision(self, y_true, y_pred):
        y_pred = self.__normalize_y_pred(y_pred)

        return K.sum(self.__true_positive(y_true, y_pred))/(K.sum(y_pred)+K.epsilon())

    def __micro_recall(self, y_true, y_pred):
        return K.sum(self.__true_positive(y_true, y_pred))/(K.sum(y_true)+K.epsilon())

    def __micro_f_measure(self, y_true, y_pred):
        precision = self.__micro_precision(y_true, y_pred)
        recall = self.__micro_recall(y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    def __average_accuracy(self, y_true, y_pred):
        class_count = y_pred.shape[-1]
        class_acc_list = [self.__class_accuracy(i, y_true, y_pred) for i in range(class_count)]
        class_acc_matrix = K.concatenate(class_acc_list, axis=0)

        return K.mean(class_acc_matrix, axis=0)

    def __macro_precision(self, y_true, y_pred):
        class_count = y_pred.shape[-1]

        return K.sum([self.__class_precision(i, y_true, y_pred) for i in range(class_count)]) \
                / K.cast(class_count, K.floatx())

    def __macro_recall(self, y_true, y_pred):
        class_count = y_pred.shape[-1]

        return K.sum([self.__class_recall(i, y_true, y_pred) for i in range(class_count)]) \
                / K.cast(class_count, K.floatx())

    def __macro_f_measure(self, y_true, y_pred):
        precision = self.__macro_precision(y_true, y_pred)
        recall = self.__macro_recall(y_true, y_pred)

        return (2.*precision*recall)/(precision+recall+K.epsilon())

    # ---------- regression indicators
    def __rms(self, y_true, y_pred):
        return ((K.mean(y_true-y_pred)**2))**0.5

    def __r2(self, y_true, y_pred):
        return 1.-(K.sum((y_true-y_pred)**2)/(K.sum((y_true-K,mean(y_true))**2)+K.epsilon()))

    def __rmse(self, y_true, y_pred):
        return K.sqrt(K.mean((y_true-y_pred)**2))

    def __mce(self, y_true, y_pred):
        return K.mean(K.abs(y_true-y_pred))

    # ---------- autoencoder indicators
    def __dice_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.sum(y_true)+K.sum(y_pred)+K.epsilon())

    def __jaccard_index(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.sum(y_true)+\
                K.sum(y_pred)-K.sum(y_true*y_pred)+K.epsilon())

    def __overlap_coef(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        return (2.*K.sum(y_true*y_pred))/(K.min([K.sum(y_true),K.sum(y_pred)])+K.epsilon())
