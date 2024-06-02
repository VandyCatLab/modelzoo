import numpy as np
import os
import glob
import pandas as pd
import numba as nb
import itertools



@nb.jit(nopython=True, parallel=True)
def preprocess_eucRsaNumba(acts):
    assert acts.ndim in (2, 4)

    if acts.ndim == 4:
        imgs, x, y, chans = acts.shape
        newShape = x * y * chans
        newActs = acts.reshape((imgs, newShape))
        # Preallocate array for RDM
        result = np.zeros((imgs, imgs))
        # Loop through each image and calculate euclidean distance
        for i in nb.prange(imgs):
            for j in nb.prange(imgs):
                result[i, j] = np.linalg.norm(newActs[i] - newActs[j])
                result[j, i] = result[i, j]
    else:
        imgs = acts.shape[0]
        # Preallocate array for RDM
        result = np.zeros((imgs, imgs))
        # Loop through each image and calculate euclidean distance
        for i in nb.prange(imgs):
            for j in nb.prange(imgs):
                result[i, j] = np.linalg.norm(acts[i] - acts[j])
                result[j, i] = result[i, j]

    return result


@nb.jit(nopython=True, parallel=True)
def nb_cov(x, y):
    # Concatenate x and y
    x = np.concatenate((x, y), axis=0)

    # Subtract feature mean from each feature
    for i in nb.prange(x.shape[0]):
        x[i, :] -= x[i, :].mean()

    # Dot product
    result = np.dot(x, x.T)

    # Normalization
    factor = x.shape[1] - 1
    result *= np.true_divide(1, factor)

    return result


@nb.jit(nopython=True, parallel=True)
def nb_cor(x, y):
    # Get covariance matrix
    c = nb_cov(x, y)

    # Get diagonal to normalize into correlation matrix
    d = np.sqrt(np.diag(c))

    # Divide by rows
    for i in nb.prange(d.shape[0]):
        c[i, :] /= d

    # Tranpose and divide it again
    c = c.T

    for i in nb.prange(d.shape[0]):
        c[i, :] /= d

    # Transpose back
    c = c.T


    return c


@nb.jit(nopython=True, parallel=True)
def do_rsaNumba(rdm1, rdm2):
    """
    Pre: RDMs must be same shape
    """
    imgs = rdm1.shape[0]

    # Only use upper-triangular values
    upperTri = np.triu_indices(n=imgs, k=1)

    rdm1_flat = np.empty((1, upperTri[0].shape[0]), dtype="float32")
    rdm2_flat = np.empty((1, upperTri[0].shape[0]), dtype="float32")
    for n in nb.prange(upperTri[0].shape[0]):
        i = upperTri[0][n]
        j = upperTri[1][n]
        rdm1_flat[0, n] = rdm1[i, j]
        rdm2_flat[0, n] = rdm2[i, j]

    # Return pearson coefficient
    return nb_cor(rdm1_flat, rdm2_flat)[0, 1]



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
        choices=["correspondence", "getReps", "seedSimMat", "itemSimMat"],
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
        default=None,
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
    parser.add_argument(
        "--reps_dir",
        type=str,
        default="../outputs/masterOutput/representations",
        help="directory for representations",
    )
    parser.add_argument(
        "--simSet",
        type=str,
        default="all",
        help="which set of similarity functions to use",
    )
    parser.add_argument(
        "--layer_index",
        "-l",
        type=_split_comma_str,
        help="which layer to use, must be positive here, split by comma",
    )
    parser.add_argument(
        "--simMatType",
        type=str,
        help="what to name the output similarity matrix file",
    )
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="if set, add this proportion of noise to the representations based on their standard deviation",
    )
    args = parser.parse_args()

    np.random.seed(2022)
    # Now do analysis
    if args.analysis == "correspondence":
        print("Performing correspondence analysis.", flush=True)

        preprocFuns, simFuns, analysisNames = get_funcs(args.simSet)

        # Get model
        if args.model_seeds is not None:
            _, modelName, _ = get_model_from_args(args, return_model=False)
            modelName = modelName.split(".")[0]
        else:
            modelList = glob.glob(os.path.join(args.reps_dir, "*.npy"))
            # Get mode
            modelPath = modelList[args.model_index]
            modelName = modelPath.split("/")[-1].split(".")[0].split('-Reps')[0]

        # List model representations and make combinations
        reps = glob.glob(args.reps_dir + "/*")
        for rep in reps:
            if "w" in rep and "s" in rep:
                reps = [
                    rep.split("/")[-1] for rep in reps if "w" in rep and "s" in rep
                ]
            else:
                reps = [
                    rep.split("/")[-1].split(".")[0].split('-Reps')[0] for rep in reps
                ]
        # reps = [
        #     rep.split("/")[-1] for rep in reps if "w" in rep and "s" in rep else
        # ]
        repCombos = list(itertools.combinations(reps, 2))
        repCombos = [x for x in repCombos if x[0] == modelName]

        # Prepare dataframes
        if args.output_dir is not None:
            fileName = f"{args.output_dir}/{modelName}Correspondence.csv"
        else:
            fileName = f"../outputs/masterOutput/correspondence/{modelName}Correspondence.csv"
        if os.path.exists(fileName):
            # Load existing dataframe
            print("path exists")
            winners = pd.read_csv(fileName, index_col=0)
        else:
            numLayers = len(
                glob.glob(f"{args.reps_dir}/{reps[0]}/{reps[0]}l*.npy")
            )
            winners = pd.DataFrame(
                sum([[combo] * numLayers for combo in repCombos], []),
                columns=["model1", "model2"],
            )
            winners[analysisNames] = -1

        # Find the winners
        for model1, model2 in repCombos:
            print(f"Comparing {model1} and {model2}", flush=True)
            if np.all(
                winners.loc[
                    (winners["model1"] == model1)
                    & (winners["model2"] == model2),
                    analysisNames,
                ]
                == -1
            ):
                winners.loc[
                    (winners["model1"] == model1)
                    & (winners["model2"] == model2),
                    analysisNames,
                ] = correspondence_test(
                    model1, model2, preprocFuns, simFuns, names=analysisNames
                )

                print("Saving results", flush=True)
                winners.to_csv(
                    fileName
                )
            else:
                print("This pair is complete, skipping", flush=True)

    elif args.analysis == "getReps":
        print("Getting representations each non-dropout layer", flush=True)

        # Load dataset
        print("Loading dataset", flush=True)
        dataset = np.load(args.dataset_file)
        print(f"dataset shape: {dataset.shape}", flush=True)

        # Run it!
        get_reps_from_all(args.models_dir, dataset, args.output_dir)
    elif args.analysis == "seedSimMat":
        print("Creating model similarity matrix.", flush=True)
        preprocFuns, simFuns, simNames = get_funcs(args.simSet)

        for layer in args.layer_index:
            for preprocFun, simFun, simName in zip(
                preprocFuns, simFuns, simNames
            ):
                print(f"Working on layer {layer} with {simFun.__name__}")
                simMat = get_seed_model_sims(
                    args.model_seeds,
                    args.reps_dir,
                    layer,
                    preprocFun,
                    simFun,
                    args.noise,
                )

                if args.output_dir is None:
                    np.save(
                        f"../outputs/masterOutput/similarities/simMat_l{layer}_{simName}.npy",
                        simMat,
                    )
                else:
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    np.save(
                        os.path.join(
                            args.output_dir, f"simMat_l{layer}_{simName}.npy"
                        ),
                        simMat,
                    )
    elif args.analysis == "itemSimMat":
        print(
            "Creating model similarity matrix for item weighting differences.",
            flush=True,
        )
        preprocFun, simFun, simNames = get_funcs(args.simSet)

        get_unstruct_model_sims(
            args.reps_dir,
            args.layer_index,
            preprocFun,
            simFun,
            simNames,
            args.simMatType,
            args.output_dir,
            args.noise,
        )
    else:
        # can delete if needed
        rep_dir = "../hubReps/all_reps/"

        # Find all models, doesn't  remove -Reps.npy
        models = glob.glob(os.path.join(rep_dir, '*.npy'))

        # Create dataframe for model similarities
        #master_array = np.load('../correspondences/master_numpy_new.npy', allow_pickle=True)
        sims = pd.DataFrame(columns=["model1", "model2", "eucRsa"])

        testing = True
        if testing is True:
            print('testing is true')
            testmodels = ['../hubReps/all_reps/vgg19_bn-Reps.npy', '../hubReps/all_reps/maxvit_tiny_rw_224-Reps.npy']

            rep1 = np.load(
                os.path.join(testmodels[0])
            )
            rep2 = np.load(
                os.path.join(testmodels[1])
            )

            # Get similarities
            rep1_proc = preprocess_eucRsaNumba(rep1)
            rep2_proc = preprocess_eucRsaNumba(rep2)
            print(rep1_proc)
            print()
            print()
            print(rep2_proc)

            sim = do_rsaNumba(rep1_proc, rep2_proc)
            print(sim)

            combo_names = [comb.split('/')[-1].split('-Reps')[0] for comb in testmodels]

            # Add to dataframe
            sims.loc[len(sims)] = list(combo_names) + [sim]
            np.save(
                os.path.join(
                    args.output_dir, f"master_numpy_testing.npy"
                ),
                sims,
            )

        if testing is False:
            #sims.to_csv(os.path.join(args.output_dir, 'correspondences.csv'))
            for combo in itertools.combinations(models, 2):
                print("Comparing models:", combo)
                combo_names = [comb.split('/')[-1].split('-Reps')[0] for comb in combo]
                if len(np.intersect1d(np.where(sims.model1 == combo_names[0]), np.where(sims.model2 == combo_names[1]))) != 0:
                    print('Sim Exists 0, Skipping.')
                elif len(np.intersect1d(np.where(sims.model1 == combo_names[1]), np.where(sims.model2 == combo_names[0]))) != 0:
                    print('Sim Exists 1, Skipping.')
                else:
                    # Get reps
                    rep1 = np.load(
                        os.path.join(combo[0])
                    )
                    rep2 = np.load(
                        os.path.join(combo[1])
                    )

                    # Get similarities
                    rep1_proc = preprocess_eucRsaNumba(rep1)
                    rep2_proc = preprocess_eucRsaNumba(rep2)

                    sim = do_rsaNumba(rep1_proc, rep2_proc)

                    # Add to dataframe
                    sims.loc[len(sims)] = list(combo_names) + [sim]
                    np.save(
                        os.path.join(
                            args.output_dir, f"master_numpy_new.npy"
                        ),
                        sims,
                    )


        # arr = []
        # for file in os.listdir(args.output_dir):
        #     temp = np.load(f'{args.output_dir}/{file}', allow_pickle=True)
        #     arr.append(temp)
        #
        # df = pd.DataFrame(arr)
        # print(df)

#htop see cores over ssh
    #    sims
