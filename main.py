import argparse
import os

import numpy as np
import tensorflow as tf

import config
import data
import download
import model
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# Enable memory growth for GPUs (required for tensorflow-metal stability)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def detect_metal_gpu():
    """Detect and log available Metal GPU devices on macOS.

    Returns:
        bool: True if Metal GPU is available, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(">> Metal GPU detected: %d device(s) available" % len(gpus))
        for gpu in gpus:
            print("   - %s" % gpu.name)
        return True
    else:
        print(">> WARNING: No Metal GPU detected. Training will use CPU.")
        print("   Ensure tensorflow-metal is installed: pip install tensorflow-metal")
        return False


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")

    # Use GCS output path if set, otherwise use local paths
    if config.GCS_OUTPUT_PATH:
        # GCS path format: gs://bucket/path/
        results_path = config.GCS_OUTPUT_PATH.rstrip("/") + "/"
    else:
        results_path = current_path + "/results/"

    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths


def train_model(dataset, paths, device):
    """The main function for executing network training. It loads the specified
       dataset iterator, saliency model, and helper classes. Training is then
       performed by iterating over all batches for a number of epochs. After
       validation on an independent set, the model is saved and the training
       history is updated.

    Args:
        dataset (str): Denotes the dataset to be used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    train_ds, valid_ds = data.get_dataset_iterator("train", dataset, paths["data"])

    msi_net = model.MSINET()

    # Build the model by calling it once
    dummy_input = tf.zeros((1, 240, 320, 3))
    msi_net(dummy_input)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.PARAMS["learning_rate"])

    # Restore weights if available
    msi_net.restore(dataset, paths, device)

    n_train_data = getattr(data, dataset.upper()).n_train
    n_valid_data = getattr(data, dataset.upper()).n_valid

    n_train_batches = int(np.ceil(n_train_data / config.PARAMS["batch_size"]))
    n_valid_batches = int(np.ceil(n_valid_data / config.PARAMS["batch_size"]))

    history = utils.History(n_train_batches,
                            n_valid_batches,
                            dataset,
                            paths["history"],
                            device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)

    print(">> Start training on %s..." % dataset.upper())

    for epoch in range(config.PARAMS["n_epochs"]):
        # Training loop
        for batch, (input_images, ground_truths, _, _) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                predicted_maps = msi_net(input_images, training=True)
                error = model.kld_loss(ground_truths, predicted_maps)

            gradients = tape.gradient(error, msi_net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, msi_net.trainable_variables))

            history.update_train_step(error.numpy())
            progbar.update_train_step(batch)

        # Validation loop
        for batch, (input_images, ground_truths, _, _) in enumerate(valid_ds):
            predicted_maps = msi_net(input_images, training=False)
            error = model.kld_loss(ground_truths, predicted_maps)

            history.update_valid_step(error.numpy())
            progbar.update_valid_step()

        msi_net.save_weights(dataset, paths["latest"], device)

        history.save_history()

        progbar.write_summary(history.get_mean_train_error(),
                              history.get_mean_valid_error())

        if history.valid_history[-1] == min(history.valid_history):
            msi_net.save_weights(dataset, paths["best"], device)
            msi_net.export_saved_model(dataset, paths["best"], device)

            print("\tBest model!", flush=True)


def test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    test_ds = data.get_dataset_iterator("test", dataset, paths["data"])

    model_name = "model_%s_%s" % (dataset, device)
    saved_model_path = paths["best"] + model_name

    # Try to load SavedModel first, fall back to weights
    if os.path.isdir(saved_model_path):
        loaded_model = tf.saved_model.load(saved_model_path)
        infer = loaded_model.signatures["serving_default"]
        use_saved_model = True
    else:
        # Check for weights
        weights_path = paths["best"] + model_name + ".weights.h5"
        if not os.path.isfile(weights_path):
            weights_path = paths["weights"] + model_name + ".weights.h5"
            if not os.path.isfile(weights_path):
                download.download_pretrained_weights(paths["weights"], model_name)

        msi_net = model.MSINET()
        # Build model
        dummy_input = tf.zeros((1, 240, 320, 3))
        msi_net(dummy_input)
        msi_net.load_weights(weights_path)
        use_saved_model = False

    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    for input_images, original_shape, file_path in test_ds:
        if use_saved_model:
            result = infer(input=input_images)
            predicted_maps = result["output"]
        else:
            predicted_maps = msi_net(input_images, training=False)

        output_file = data.postprocess_saliency_map(predicted_maps[0],
                                                    original_shape[0])

        path = file_path[0][0].numpy().decode("utf-8")

        filename = os.path.basename(path)
        filename = os.path.splitext(filename)[0]
        filename += ".jpeg"

        os.makedirs(paths["images"], exist_ok=True)

        with open(paths["images"] + filename, "wb") as file:
            file.write(output_file.numpy())


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    phases_list = ["train", "test"]

    datasets_list = ["salicon", "mit1003", "cat2000",
                     "dutomron", "pascals", "osie", "fiwi",
                     "fixationadd1000"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    parser.add_argument("-p", "--path", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    args = parser.parse_args()

    paths = define_paths(current_path, args)

    # Detect Metal GPU if device is set to "metal"
    if config.PARAMS["device"] == "metal":
        detect_metal_gpu()

    if args.phase == "train":
        train_model(args.data, paths, config.PARAMS["device"])
    elif args.phase == "test":
        test_model(args.data, paths, config.PARAMS["device"])


if __name__ == "__main__":
    main()
