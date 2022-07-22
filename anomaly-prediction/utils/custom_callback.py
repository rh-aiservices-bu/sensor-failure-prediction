from keras.callbacks import Callback


class CustomCallback(Callback):
    job_size = None
    progress_counter = 0
    train_loss = []

    def on_train_begin(self, logs=None):
        print("start Training")
        yield "event: initialize\ndata: " + str(CustomCallback.job_size) + "\n\n"

    def on_train_batch_end(self, batch, logs=None):
        CustomCallback.train_loss.append(logs['loss'])
        print("End of batch {}".format(CustomCallback.progress_counter))
        CustomCallback.progress_counter += 1
        yield "event: inprogress\ndata: " + str(CustomCallback.progress_counter) + "\n\n"

    def on_train_end(self, logs=None):
        print("End of training")
        yield "event: jobfinished\ndata: " + "\n\n"
