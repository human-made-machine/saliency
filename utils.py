import os
import time
from datetime import timedelta

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GCS compatibility
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import tensorflow as tf


class History:
    """This class represents the training history of a model. It can load the
       prior history when training continues, keeps track of the training and
       validation error, and finally plots them as a curve after each epoch.
    """

    def __init__(self, n_train_batches, n_valid_batches,
                 dataset, path, device):
        self.train_history = []
        self.valid_history = []

        self._prior_epochs = 0

        self._train_error = 0
        self._valid_error = 0

        self._n_train_batches = n_train_batches
        self._n_valid_batches = n_valid_batches

        self._path = path
        self._id = (dataset, device)

        self._get_prior_history()

    def _get_prior_history(self):
        train_file = self._path + "train_%s_%s.txt" % self._id
        if tf.io.gfile.exists(train_file):
            with tf.io.gfile.GFile(train_file, "r") as file:
                for line in file.readlines():
                    self.train_history.append(float(line))

        valid_file = self._path + "valid_%s_%s.txt" % self._id
        if tf.io.gfile.exists(valid_file):
            with tf.io.gfile.GFile(valid_file, "r") as file:
                for line in file.readlines():
                    self.valid_history.append(float(line))

        self.prior_epochs = len(self.train_history)

    def update_train_step(self, train_error):
        self._train_error += train_error

    def update_valid_step(self, valid_error):
        self._valid_error += valid_error

    def get_mean_train_error(self, reset=True):
        mean_train_error = self._train_error / self._n_train_batches

        if reset:
            self._train_error = 0

        return mean_train_error

    def get_mean_valid_error(self, reset=True):
        mean_valid_error = self._valid_error / self._n_valid_batches

        if reset:
            self._valid_error = 0

        return mean_valid_error

    def save_history(self):
        mean_train_loss = self.get_mean_train_error(False)
        mean_valid_loss = self.get_mean_valid_error(False)

        self.train_history.append(mean_train_loss)
        self.valid_history.append(mean_valid_loss)

        tf.io.gfile.makedirs(self._path)

        train_file = self._path + "train_%s_%s.txt" % self._id
        with tf.io.gfile.GFile(train_file, "a") as file:
            file.write("%f\n" % self.train_history[-1])

        valid_file = self._path + "valid_%s_%s.txt" % self._id
        with tf.io.gfile.GFile(valid_file, "a") as file:
            file.write("%f\n" % self.valid_history[-1])

        if len(self.train_history) > 1:
            axes = plt.figure().gca()

            x_range = np.arange(1, len(self.train_history) + 1)

            plt.plot(x_range, self.train_history, label="train", linewidth=2)
            plt.plot(x_range, self.valid_history, label="valid", linewidth=2)

            plt.legend()
            plt.xlabel("epochs")
            plt.ylabel("error")

            locations = plticker.MultipleLocator(base=1.0)
            axes.xaxis.set_major_locator(locations)

            # Save plot to GCS or local filesystem
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            curve_file = self._path + "curve_%s_%s.png" % self._id
            with tf.io.gfile.GFile(curve_file, "wb") as file:
                file.write(buf.read())
            plt.close()


class Progbar:
    """This class represents a progress bar for the terminal that visualizes
       the training progress for each epoch, estimated time of accomplishment,
       and then summarizes the training and validation loss together with the
       elapsed time.
    """

    def __init__(self, n_train_data, n_train_batches,
                 batch_size, n_epochs, prior_epochs):
        self._train_time = 0
        self._valid_time = 0

        self._start_time = time.time()
        self._training_start_time = time.time()

        self._batch_size = batch_size

        self._n_train_data = n_train_data
        self._n_train_batches = n_train_batches

        self._n_epochs = n_epochs
        self._prior_epochs = prior_epochs
        self._completed_epochs = 0
        self._epoch_times = []

        self._target_epoch = str(n_epochs + prior_epochs).zfill(2)
        self._current_epoch = str(prior_epochs + 1).zfill(2)

    def _flush(self):
        epoch_total_time = self._train_time + self._valid_time
        self._epoch_times.append(epoch_total_time)
        self._completed_epochs += 1

        self._train_time = 0
        self._valid_time = 0

        self._start_time = time.time()

        current_epoch_int = int(self._current_epoch) + 1
        self._current_epoch = str(current_epoch_int).zfill(2)

    def _format_time(self, seconds):
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return "%ds" % seconds
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return "%dm %ds" % (minutes, secs)
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return "%dh %dm" % (hours, minutes)

    def _estimate_total_remaining(self, current_batch):
        """Estimate remaining time for entire training run."""
        epochs_remaining = self._n_epochs - self._completed_epochs - 1

        if self._epoch_times:
            avg_epoch_time = np.mean(self._epoch_times)
        else:
            if current_batch > 0:
                batch_time = self._train_time / current_batch
                avg_epoch_time = batch_time * self._n_train_batches * 1.1
            else:
                return None

        batch_train_time = self._train_time / max(current_batch, 1)
        current_epoch_remaining = (self._n_train_batches - current_batch) * batch_train_time

        total_remaining = current_epoch_remaining + (epochs_remaining * avg_epoch_time)
        return total_remaining

    def update_train_step(self, current_batch):
        current_batch += 1

        self._train_time = time.time() - self._start_time
        batch_train_time = self._train_time / current_batch

        eta_epoch = (self._n_train_batches - current_batch) * batch_train_time
        eta_epoch_str = self._format_time(int(np.ceil(eta_epoch)))

        eta_total = self._estimate_total_remaining(current_batch)
        if eta_total is not None:
            eta_total_str = self._format_time(int(np.ceil(eta_total)))
        else:
            eta_total_str = "..."

        progress_line = "=" * (20 * current_batch // self._n_train_batches)

        current_instance = current_batch * self._batch_size
        current_instance = np.clip(current_instance, 0, self._n_train_data)

        progress_frac = "%i/%i" % (current_instance, self._n_train_data)

        information = (self._current_epoch, self._target_epoch,
                       progress_line, progress_frac, eta_epoch_str, eta_total_str)

        progbar_output = "Epoch %s/%s [%-20s] %s (ETA: %s | Total: %s)" % information

        print(progbar_output, end="\r", flush=True)

    def update_valid_step(self):
        self._valid_time = time.time() - self._start_time - self._train_time

    def write_summary(self, mean_train_loss, mean_valid_loss):
        train_time = str(timedelta(seconds=np.ceil(self._train_time)))
        valid_time = str(timedelta(seconds=np.ceil(self._valid_time)))

        total_elapsed = time.time() - self._training_start_time
        total_elapsed_str = self._format_time(int(np.ceil(total_elapsed)))

        epochs_remaining = self._n_epochs - self._completed_epochs - 1
        epoch_time = self._train_time + self._valid_time

        if epochs_remaining > 0:
            if self._epoch_times:
                avg_epoch = np.mean(self._epoch_times + [epoch_time])
            else:
                avg_epoch = epoch_time
            eta_remaining = epochs_remaining * avg_epoch
            eta_str = self._format_time(int(np.ceil(eta_remaining)))
            progress_info = "Elapsed: %s | Remaining: %s" % (total_elapsed_str, eta_str)
        else:
            progress_info = "Total time: %s" % total_elapsed_str

        train_information = (mean_train_loss, train_time)
        valid_information = (mean_valid_loss, valid_time)

        train_output = "\n\tTrain loss: %.6f (%s)" % train_information
        valid_output = "\tValid loss: %.6f (%s)" % valid_information

        print(train_output, flush=True)
        print(valid_output, flush=True)
        print("\t%s" % progress_info, flush=True)

        self._flush()
