import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

'''
TensorBoard Data will be stored in './runs' path
'''


class Logger:
    def __init__(self, model):
        """
        @Param model_name (str): name of the model
        """
        self.writer = SummaryWriter(comment=model)
        # self.writer_dict = {metric: SummaryWriter(comment=metric) for metric in metrics}


    def log(self, metrics, epoch):
        """
        """
        for metric_name, metric_val in metrics.items():
            if isinstance(metric_val, float):
                self.writer.add_scalar(
                    '%s' % (metric_name), metric_val, epoch
                )
            else:
                self.writer.add_scalar(
                    '%s/mean' % (metric_name), metric_val.mean(), epoch
                )
                self.writer.add_scalar(
                    '%s/std' % (metric_name), metric_val.std(), epoch
                )
        # for metric_name, metric_val in metrics.items():
        #     if isinstance(metric_val, float):
        #         self.writer_dict[metric_name].add_scalar(
        #             '%s/%s' % (metric_name, model_name), metric_val, epoch
        #         )
        #     else:
        #         self.writer_dict[metric_name].add_scalar(
        #             '%s/%s/mean' % (metric_name, model_name), metric_val.mean(), epoch
        #         )
        #         self.writer_dict[metric_name].add_scalar(
        #             '%s/%s/std' % (metric_name, model_name), metric_val.std(), epoch
        #         )


    def display_status(self, epoch, num_epochs, total_loss, accs, f1s, aucs, recalls):
        """
        @Param epoch (int): epoch for which the value is being logged
        @Param num_epochs (int): the total number of epochs
        @Param total_loss: total_loss
        @Param accs: accs
        @Param f1s: f1_score
        @Param aucs: aucs
        @Param recalls: recall_score

        Displays epoch data
        """
        print('Epoch: [{}/{}]'.format(
            epoch, num_epochs))
        print('\tLoss: ', total_loss)
        print('\tACC: %0.4f +- %0.4f' %
                (accs.mean(), accs.std()))
        print('\tF1: %0.4f +- %0.4f' %
                (f1s.mean(), f1s.std()))
        print('\tAUC: %0.4f +- %0.4f' %
                (aucs.mean(), aucs.std()))
        print('\tRecall: %0.4f +- %0.4f' %
                (recalls.mean(), recalls.std()))


    def close(self):
        self.writer.close()

    # Private Functionality
    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
