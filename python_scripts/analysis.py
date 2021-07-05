import numpy as np
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
import sys, os


sys.path.append("../imported_code/svcca")
import cca_core, pwcca

"""
For centroid RSA: 1st-level RDMs should be 10x10, not 1000x1000 (average before RSA-ing).
CCAs should be averaged category-wise before reshaping.
"""


def correlate(
    method: str,
    path_to_instances: str,
    x_predict,
    consistency="exemplar",
    cocktail_blank=False,
):
    """
    Pre: ***HARDCODED*** 10 instances at specified path with 9 layers each, 1000 images
    Post: returns 90x90 correlation matrix using RSA, SVCCA or PWCCA
    """
    # Get necessary functions
    preprocess_func, corr_func = get_funcs(method)
    # Get instances
    instances = os.listdir(path_to_instances)
    print("**** Load and Preprocess Acts ****")
    # Load up all acts into layer * instance grid
    all_acts = [[], [], [], [], [], [], [], [], []]
    # TODO: remove the limiter when u do full experiment, right now limit to 10 for testing
    limiter_idx = 0  # Remove
    for instance in instances:
        if limiter_idx == 10:
            break  # Remove
        # Skip any non-model files that may have snuck in
        if ".h5" not in instance and ".pb" not in instance:
            continue
        print("*** Working on", instance, "***")
        K.clear_session()
        model = load_model(path_to_instances + instance)
        acts_list = get_acts(model, range(9), x_predict, cocktail_blank)
        # Loop through layers
        layer_num = 0
        for acts in acts_list:
            print("* Preprocessing...")
            acts = preprocess_func(acts, consistency)
            all_acts[layer_num].append(acts)
            layer_num += 1
        limiter_idx += 1  # Remove

    # Now do correlations
    print("**** Done gathering representations, now correlations ****")
    num_networks = 10
    correlations = np.zeros((90, 90))
    # Run Analysis
    for i in range(correlations.shape[0]):
        # Only perform on lower triangle, avoid repeats
        # NOTE: Lower triangle b/c yields better separation with PWCCA
        for j in range(i + 1):
            print("Correlation", str(i), ",", str(j))
            # Decode into the org scheme we want (i.e. layer * instance)
            layer_i = i // num_networks
            network_i = i % num_networks
            layer_j = j // num_networks
            network_j = j % num_networks
            acts1 = all_acts[layer_i][network_i]
            acts2 = all_acts[layer_j][network_j]

            correlations[i, j] = corr_func(acts1, acts2)

    # Fill in other side of graph with reflection
    correlations += correlations.T
    for i in range(correlations.shape[0]):
        correlations[i, i] /= 2

    print("Done!")
    return correlations


def correlate_trajectory(
    epochs_scheme: str,
    method: str,
    path_to_instances: str,
    category: str,
    x_predict,
    consistency="exemplar",
    cocktail_black=False,
):
    """
    Pre:  Arbitrary number of instances and corresponding acts at specified paths
    Post: returns (num_epochs*num_instances) ^ 2 correlation matrix using RSA, SVCCA or PWCCA
    """
    # Establish epochs scheme and corresponding acts list
    assert epochs_scheme in ["first_ten", "fifties"]
    if epochs_scheme == "first_ten":
        epochs = range(10)
        all_acts = [[], [], [], [], [], [], [], [], [], []]
    else:
        epochs = [0, 49, 99, 149, 199, 249, 299, 349]
        all_acts = [[], [], [], [], [], [], [], []]
    # Which category are we testing
    assert category in ["weights", "shuffle", "both"]
    if category == "weights":
        path_to_acts = "../outputs/representations/acts/weights/"
    elif category == "shuffle":
        path_to_acts = "../outputs/representations/acts/shuffle_seed/"
    else:
        path_to_acts = "../outputs/representations/acts/both/"
    # Get necessary functions
    preprocess_func, corr_func = get_funcs(method)
    # Get instance names
    instance_names = os.listdir(path_to_instances)
    # TODO: remove the limiter when u do full experiment, right now limit to 10 for testing
    limiter_idx = 0  # Remove
    for name in instance_names:
        if limiter_idx == 10:
            break  # Remove
        # Skip any non-model files that may have snuck in
        if ".h5" not in name:
            continue
        # Grab the number from the instance name
        i = name[:-3] if category == "both" else name[9:-3]
        # Get all the acts from the relevant epochs of that instance number
        epoch_index = 0
        for e in epochs:
            # Need to take [0] at the end because they were stored as arrays of 1
            if category == "weights":
                acts = np.load(path_to_acts + "i" + i + "e" + str(e) + ".npy")[
                    0
                ]
            elif category == "shuffle":
                acts = np.load(path_to_acts + "s" + i + "e" + str(e) + ".npy")[
                    0
                ]
            else:
                acts = np.load(path_to_acts + i + "e" + str(e) + ".npy")[0]
            print("* Preprocessing...")
            acts = preprocess_func(acts, consistency)
            all_acts[epoch_index].append(acts)
            epoch_index += 1

        limiter_idx += 1  # Remove

    # Now do correlations
    print("**** Done gathering representations, now correlations ****")
    num_instances = len(all_acts[0])
    correlations = np.zeros(
        (num_instances * len(epochs), num_instances * len(epochs))
    )
    # Run analysis
    for i in range(correlations.shape[0]):
        # Only perform on lower triangle, avoid repeats
        # NOTE: Lower triangle b/c yields better separation with PWCCA
        for j in range(i + 1):
            print("Correlation", str(i), ",", str(j))
            # Decode into the org scheme we want (i.e. layer * instance)
            epoch_i = i // num_instances
            instance_i = i % num_instances
            epoch_j = j // num_instances
            instance_j = j % num_instances
            acts1 = all_acts[epoch_i][instance_i]
            acts2 = all_acts[epoch_j][instance_j]
            print("Acts1:", acts1.shape)
            print("Acts2:", acts2.shape)
            correlations[i, j] = corr_func(acts1, acts2)

    # Fill in other side of graph with reflection
    correlations += correlations.T
    for i in range(correlations.shape[0]):
        correlations[i, i] /= 2

    print("Done!")
    return correlations


def get_funcs(method):
    assert method in ["RSA", "SVCCA", "PWCCA"], "Invalid correlation method"
    if method == "RSA":
        return preprocess_rsa, do_rsa
    elif method == "SVCCA":
        return preprocess_svcca, do_svcca
    elif method == "PWCCA":
        return preprocess_pwcca, do_pwcca


"""
Preprocessing functions
"""


def preprocess_rsa(acts, consistency="exemplar"):
    # Note: Hardcoded on 10 categories
    categories = 10
    if len(acts.shape) > 2:
        imgs, h, w, channels = acts.shape
        acts = np.reshape(acts, newshape=(imgs, h * w * channels))
    if consistency == "centroid":
        centroid_acts = np.empty((categories, acts.shape[1]))
        # imgs/categories should be a clean divide
        imgs_per_cat = int(imgs / categories)
        for i in range(0, imgs, imgs_per_cat):
            centroid_acts[int(i / imgs_per_cat)] = np.mean(
                acts[i : i + imgs_per_cat], axis=0
            )
        acts = centroid_acts
    rdm = get_rdm(acts)
    return rdm


# TODO: merge with interpolate
def preprocess_svcca(acts, interpolate=False):
    if len(acts.shape) > 2:
        acts = np.mean(acts, axis=(1, 2))
    # Transpose to get shape [neurons, datapoints]
    threshold = get_threshold(acts.T)
    # Mean subtract activations
    cacts = acts.T - np.mean(acts.T, axis=1, keepdims=True)
    # Perform SVD
    _, s, V = np.linalg.svd(cacts, full_matrices=False)

    svacts = np.dot(s[:threshold] * np.eye(threshold), V[:threshold])
    return svacts


# TODO: merge with interpolate
def preprocess_pwcca(acts, interpolate=False):
    if len(acts.shape) > 2:
        acts = np.mean(acts, axis=(1, 2))

    return acts


def preprocess_cka(acts):
    """
    Changes to sample by neuron shape as needed. Uses global average pooling.
    """
    if len(acts.shape) > 2:
        acts = np.squeeze(np.apply_over_axes(np.mean, acts, [1, 2]))

    return acts


"""
Correlation analysis functions
"""


def do_rsa_from_acts(acts1, acts2):
    """
    Pre: acts must be shape (neurons, datapoints)
    """
    rdm1 = get_rdm(acts1)
    rdm2 = get_rdm(acts2)
    return do_rsa(rdm1, rdm2)


def do_rsa(rdm1, rdm2):
    """
    Pre: RDMs must be same shape
    """
    assert rdm1.shape == rdm2.shape, (
        "rdm1: " + str(rdm1.shape) + " rdm2: " + str(rdm2.shape)
    )
    num_imgs = rdm1.shape[0]
    # Only use upper-triangular values
    rdm1_flat = rdm1[np.triu_indices(n=num_imgs, k=1)]
    rdm2_flat = rdm2[np.triu_indices(n=num_imgs, k=1)]
    # Return squared spearman coefficient
    return spearmanr(rdm1_flat, rdm2_flat)[0] ** 2


def do_svcca(acts1, acts2):
    """
    Pre: acts must be shape (neurons, datapoints) and preprocessed with SVD
    """
    svcca_results = cca_core.get_cca_similarity(
        acts1, acts2, epsilon=1e-10, verbose=False
    )
    return np.mean(svcca_results["cca_coef1"])


def do_pwcca(acts1, acts2):
    """
    Pre: acts must be shape (neurons, datapoints)
    """
    # acts1.shape cannot be bigger than acts2.shape for pwcca
    if acts1.shape <= acts2.shape:
        return np.mean(pwcca.compute_pwcca(acts1.T, acts2.T, epsilon=1e-10)[0])
    return np.mean(pwcca.compute_pwcca(acts2.T, acts1.T, epsilon=1e-10)[0])


def do_linearCKA(acts1, acts2):
    """
    Pre: acts must be shape (datapoints, neurons)
    """

    def _linearKernel(acts):
        return np.dot(acts, acts.T)

    def _centerMatrix(n):
        return np.eye(n) - (np.ones((n, n)) / n)

    def _hsic(x, y):
        n = x.shape[0]
        centeredX = np.dot(_linearKernel(x), _centerMatrix(n))
        centeredY = np.dot(_linearKernel(y), _centerMatrix(n))
        return np.trace(np.dot(centeredX, centeredY)) / ((n - 1) ** 2)

    return _hsic(acts1, acts2) / (
        (_hsic(acts1, acts1) * _hsic(acts2, acts2)) ** (1 / 2)
    )


def correspondence_test(
    model1: Model,
    model2: Model,
    dataset: np.ndarray,
    preproc_fun: list,
    sim_fun: list,
):
    """
    Return a list of indices for each model1 layer's most similar layer in 
    model2 for each similarity function given the dataset. If multiple 
    similarity functions are given, return a dictionary.
    """
    # Get a model that outputs at every layer and get representations
    outModel = make_allout_model(model1)
    mainReps = outModel.predict(dataset)

    # Do the same for model 2
    outModel = make_allout_model(model2)
    altReps = outModel.predict(dataset)

    # Create dict for results
    results = {fun.__name__: [] for fun in sim_fun}

    # Loop through layers of model1
    for i, rep in enumerate(mainReps):
        winners = {fun.__name__: [-1, 0] for fun in sim_fun}
        for layer, altRep in enumerate(altReps):
            output = multi_analysis(rep, altRep, preproc_fun, sim_fun)

            for fun, sim in output.items():
                if sim > winners[fun][1]:
                    # New winner
                    winners[fun] = [layer, sim]

        # Save winners for this layer
        for fun, layerIdx in results.items():
            results[fun] += [winners[fun][0]]

        print(f"Layer {i} winners {winners}", flush=True)

    # Just return list if there's only one
    if len(results.keys()) == 1:
        results = results[sim_fun[0].__name__]

    return results


"""
Helper functions
"""


def get_acts(model, layer_arr, x_predict, cocktail_blank):
    """
    Pre: model exists, layer_arr contains valid layer numbers, x_predict is organized
    Post: Returns list of activations over x_predict for relevant layers in this particular model instance
    """
    inp = model.input
    acts_list = []

    for layer in layer_arr:
        print("Layer", str(layer))
        out = model.layers[layer].output
        temp_model = Model(inputs=inp, outputs=out)
        # Predict on x_predict, transpose for spearman
        print("Getting activations...")
        acts = temp_model.predict(x_predict)
        if cocktail_blank:
            # subtracting the mean activation pattern across all images from each network unit
            acts -= np.mean(acts, axis=0)
        acts_list.append(acts)

    return acts_list


def get_threshold(acts):
    start = 0
    end = acts.shape[0]
    return_dict = {}
    ans = -1
    while start <= end:
        mid = (start + end) // 2
        # Move to right side if target is
        # greater.
        s = np.linalg.svd(
            acts - np.mean(acts, axis=1, keepdims=True), full_matrices=False
        )[1]
        # Note: normally comparing floating points is a bad bad but the precision we need is low enough
        if np.sum(s[:mid]) / np.sum(s) <= 0.99:
            start = mid + 1
        # Move left side.
        else:
            ans = mid
            end = mid - 1

    print(
        "Found",
        ans,
        "/",
        acts.shape[0],
        "neurons accounts for",
        np.sum(s[:ans]) / np.sum(s),
        "of variance",
    )

    return ans


def get_rdm(acts):
    """
    Pre: acts must be flattened
    """
    # print('shape:', acts.shape)
    num_imgs = acts.shape[0]
    # print('num_images =', num_imgs)
    return np.corrcoef(acts, acts)[0:num_imgs, 0:num_imgs]


def make_allout_model(model):
    """
    Creates a model with outputs at every layer that is not dropout.
    """
    inp = model.input

    modelOuts = [
        layer.output for layer in model.layers if "dropout" not in layer.name
    ]

    return Model(inputs=inp, outputs=modelOuts)


"""
Large scale analysis functions
"""


def multi_analysis(rep1, rep2, preproc_fun, sim_fun):
    """
    Perform similarity analysis between rep1 and rep2 once for each method as
    indicated by first applying a preproc_fun then the sim_fun. preproc_fun 
    should be a list of functions to run on the representations before applying
    the similarity function at the same index in the list sim_fun. Elements of
    preproc_fun can be None wherein the representations are not preprocessed 
    before being passed to the paired similarity function.
    """
    assert len(preproc_fun) == len(sim_fun)

    # Loop through each pair
    simDict = {}
    for preproc, sim in zip(preproc_fun, sim_fun):
        # Preprocess each set of representations
        rep1Preproc = preproc(rep1)
        rep2Preproc = preproc(rep2)

        # Get similarity between reps
        try:
            simDict[sim.__name__] = sim(rep1Preproc, rep2Preproc)
        except Exception as e:
            simDict[sim.__name__] = np.nan
            print(f"{sim.__name__} produced an error, saving nan.")
            print(e)

    return simDict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform some type of analysis, intended to be used in HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        help="type of analysis to run",
        choices=["correspondence"],
    )
    parser.add_argument(
        "--model_index",
        "-i",
        type=int,
        help="model index to select weight and shuffle seeds",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, help="shuffle seed of the main model"
    )
    parser.add_argument(
        "--weight_seed", type=int, help="weight seed of the main model"
    )
    parser.add_argument(
        "--model_seeds",
        type=str,
        default="../outputs/masterOutput/modelSeeds.csv",
        help="file location for csv file with model seeds",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="../outputs/masterOutput/dataset.npy",
        help="npy file path for the image dataset to use for analysis",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="../outputs/masterOutput/models",
        help="directory for all of the models",
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset", flush=True)
    dataset = np.load(args.dataset_file)
    print(f"dataset shape: {dataset.shape}", flush=True)

    # Get model
    if args.model_index is not None:
        import pandas as pd

        modelIdx = args.model_index

        # Load csv and get model parameters
        modelSeeds = pd.read_csv(args.model_seeds)
        weightSeed = modelSeeds.loc[
            modelSeeds["index"] == modelIdx, "weight"
        ].item()
        shuffleSeed = modelSeeds.loc[
            modelSeeds["index"] == modelIdx, "shuffle"
        ].item()

        # Load main model
        modelName = f"w{weightSeed}s{shuffleSeed}.pb"
        modelPath = os.path.join(args.models_dir, modelName)
        model = load_model(modelPath)
    elif args.shuffle_seed is not None and args.weight_seed is not None:
        weightSeed = args.weight_seed
        shuffleSeed = args.shuffle_seed

    # Load main model
    modelName = f"w{weightSeed}s{shuffleSeed}.pb"
    modelPath = os.path.join(args.models_dir, modelName)
    model = load_model(modelPath)
    layerN = len(model.layers)
    print(f"Model loaded: {modelName}", flush=True)
    model.summary()

    import time

    # Now do analysis
    if args.analysis == "correspondence":
        print("Performing correspondence analysis.", flush=True)
        preprocFuns = [preprocess_rsa, preprocess_pwcca, preprocess_cka]
        simFuns = [do_rsa, do_pwcca, do_linearCKA]

        # Loop through all models and check for correspondence
        allModels = os.listdir(args.models_dir)
        for mdlDir in allModels:
            print(f"Working on model: {mdlDir}", flush=True)
            startTime = time.time()
            tmpModel = load_model(os.path.join(args.models_dir, mdlDir))

            results = correspondence_test(
                model, tmpModel, dataset, preprocFuns, simFuns
            )
            print("--- %s seconds ---" % (time.time() - startTime), flush=True)
