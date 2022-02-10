import analysis
import tensorflow as tf
import numpy as np
import pandas as pd
import os


def permuteTest(nImgs=None, outputPath=None):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[12]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()
    repMean = np.mean(rep_flat)
    repSD = np.std(rep_flat)

    if outputPath is None:
        permutePath = "../outputs/masterOutput/permuteSims.csv"
    else:
        permutePath = outputPath

    if os.path.exists(permutePath):
        print("Using existing permutation test results")
        permuteData = pd.read_csv(permutePath)
    else:
        preprocFuns = [
            analysis.preprocess_peaRsaNumba,
            analysis.preprocess_eucRsaNumba,
            analysis.preprocess_speRsaNumba,
            analysis.preprocess_svcca,
            analysis.preprocess_ckaNumba,
        ]
        simFuns = [
            analysis.do_rsaNumba,
            analysis.do_rsaNumba,
            analysis.do_rsaNumba,
            analysis.do_svcca,
            analysis.do_linearCKANumba,
        ]
        analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]
        colNames = analysisNames + ["analysis"]
        nPermutes = 1000

        permuteData = pd.DataFrame(columns=colNames)

        # Random permutation
        print("Doing random simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep1 = np.random.choice(rep_flat, size=repShape, replace=False)
            rep2 = np.random.choice(rep_flat, size=repShape, replace=False)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["random"]
        permuteData = permuteData.append(df)

        # Noised permutation
        print("Doing noise simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repNoise = rep + np.random.normal(scale=repSD, size=repShape)
            repNoise = repNoise.astype(np.float32)

            sims = analysis.multi_analysis(
                rep, repNoise, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["noise"]
        permuteData = permuteData.append(df)

        # Cut a neuron
        print("Doing ablation simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repCut = np.copy(rep)
            repCut[:, np.random.choice(rep.shape[1])] = 0

            sims = analysis.multi_analysis(
                rep, repCut, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[permute] = list(sims.values()) + ["ablate"]
        permuteData = permuteData.append(df)
        permuteData.to_csv(permutePath)


def sizeRatioTest(nImgs=None, outputPath=None):
    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    model = tf.keras.models.load_model(modelPath)
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    rep = model.predict(imgset).flatten()

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]
    colNames = analysisNames + ["sample", "features"]

    permuteSims = pd.DataFrame(columns=colNames)

    nMax = imgset.shape[0]
    ratiosRange = np.arange(0.05, 1, 0.05)
    nPermute = 1000

    for ratio in ratiosRange:
        print(f"Analyzing img:samples ratio: {ratio}")
        repShape = (nMax, int(nMax * ratio))
        for permute in range(nPermute):
            if permute % 100 == 0:
                print(f"Permutation at {permute}")

            # Sample reps
            rep1 = np.random.choice(rep, size=repShape)
            rep2 = np.random.choice(rep, size=repShape)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            sims["sample"] = repShape[0]
            sims["features"] = repShape[1]
            permuteSims = pd.concat(
                (
                    permuteSims,
                    pd.DataFrame.from_dict(sims, orient="index").transpose(),
                )
            )

    if outputPath is None:
        outPath = "../outputs/masterOutput/ratioSims.csv"
    else:
        outPath = outputPath
    permuteSims.to_csv(outPath, index=False)


def bigSizeRatioTest():
    # load dataset
    imgset = np.load("../outputs/masterOutput/bigDataset.npy")
    model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3))
    model.compile(metrics=["top_k_categorical_accuracy"])
    print(f"Model loaded: MobileNetV3Small", flush=True)
    rep = model.predict(imgset).flatten()

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]
    colNames = analysisNames + ["sample", "features"]

    permuteSims = pd.DataFrame(columns=colNames)

    nMax = imgset.shape[0]
    ratiosRange = np.arange(0.05, 1, 0.05)
    nPermute = 100

    for ratio in ratiosRange:
        print(f"Analyzing img:samples ratio: {ratio}")
        repShape = (nMax, int(nMax * ratio))
        for permute in range(nPermute):
            if permute % 10 == 0:
                print(f"Permutation at {permute}")

            # Sample reps
            rep1 = np.random.choice(rep, size=repShape)
            rep2 = np.random.choice(rep, size=repShape)

            sims = analysis.multi_analysis(
                rep1, rep2, preprocFuns, simFuns, names=analysisNames
            )
            sims["sample"] = repShape[0]
            sims["features"] = repShape[1]
            permuteSims = pd.concat(
                (
                    permuteSims,
                    pd.DataFrame.from_dict(sims, orient="index").transpose(),
                )
            )
    permuteSims.to_csv("../outputs/masterOutput/bigRatioSims.csv", index=False)


def parametricAblation(minNeuron=3, maxNeuron=10, nImgs=None, outputPath=None):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[12]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()
    repMean = np.mean(rep_flat)
    repSD = np.std(rep_flat)

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]
    colNames = analysisNames + ["Neurons"]
    nPermutes = 1000

    permuteData = pd.DataFrame(columns=colNames)
    if outputPath is None:
        outPath = "../outputs/masterOutput/ablateSims.csv"
    else:
        outPath = outputPath
    print("Doing ablation simulation")
    for permute in range(nPermutes):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}")

        df = pd.DataFrame(
            columns=colNames, index=range(minNeuron, maxNeuron + 1)
        )
        for nNeurons in range(minNeuron, maxNeuron + 1):
            rep = np.random.choice(rep_flat, size=repShape, replace=False)
            repCut = np.copy(rep)
            repCut = repCut[
                :, np.random.choice(rep.shape[1], size=nNeurons, replace=False)
            ]

            sims = analysis.multi_analysis(
                rep, repCut, preprocFuns, simFuns, names=analysisNames
            )
            df.loc[nNeurons] = list(sims.values()) + [nNeurons]
        permuteData = permuteData.append(df)
    permuteData.to_csv(outPath)


def parametricNoise(
    minNoise=0.0,
    maxNoise=1,
    step=0.1,
    permutations=10000,
    seed=None,
    nImgs=None,
    outputPath=None,
):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Only use a subset of the images
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Dataset information
    num_imgs = imgset.shape[0]
    dim = imgset.shape[1]
    correlations = []

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[12]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()
    repMean = np.mean(rep_flat)
    repSD = np.std(rep_flat)

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]

    colNames = analysisNames + ["Noise"]

    permuteData = pd.DataFrame(columns=colNames)
    noiseRange = np.arange(minNoise, maxNoise + 0.1, step)

    # Set seed if available
    if seed is not None:
        print(f"Setting seed {seed}", flush=True)
        np.random.seed(seed)
        outPath = f"../outputs/masterOutput/noiseSims{seed}.csv"
    else:
        if outputPath is None:
            outPath = "../outputs/masterOutput/noiseSims.csv"
        else:
            outPath = outputPath
    print("Doing parametric noise simulation")
    for permute in range(permutations):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}", flush=True)

        rep = np.random.choice(rep_flat, size=repShape, replace=False)

        df = pd.DataFrame(columns=colNames, index=noiseRange)
        for noise in noiseRange:
            repNoise = rep + np.random.normal(
                scale=repSD * noise, size=repShape
            )
            repNoise = repNoise.astype(np.float32)

            sims = analysis.multi_analysis(
                rep,
                repNoise,
                preprocFuns,
                simFuns,
                verbose=False,
                names=analysisNames,
            )
            df.loc[noise] = list(sims.values()) + [noise]
        permuteData = permuteData.append(df)

    permuteData.to_csv(outPath)


def sanity_check(nImgs=None, outputPath=None):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

    # Subset dataset
    if nImgs is not None:
        imgset = imgset[:nImgs]

    # Set model to output reps at layer
    inp = model.input
    layer = model.layers[12]
    out = layer.output

    tmpModel = tf.keras.Model(inputs=inp, outputs=out)

    # Get reps for originals and flatten
    rep_orig = tmpModel.predict(imgset)
    repShape = rep_orig.shape
    rep_flat = rep_orig.flatten()

    preprocFuns = [
        analysis.preprocess_peaRsaNumba,
        analysis.preprocess_eucRsaNumba,
        analysis.preprocess_speRsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    analysisNames = ["peaRsa", "eucRsa", "speRsa", "svcca", "cka"]

    colNames = analysisNames
    nPermutes = 1000

    permuteData = pd.DataFrame(columns=colNames, index=range(nPermutes))

    if outputPath is None:
        outPath = "../outputs/masterOutput/sanityCheckSims.csv"
    else:
        outPath = outputPath
    print("Performing sanity check, permuting features.")
    for permute in range(nPermutes):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}")

        rep = np.random.choice(rep_flat, size=repShape, replace=False)
        repPermute = np.copy(rep)
        repPermute = repPermute[
            :, np.random.choice(rep.shape[1], size=repShape[1], replace=False)
        ]

        sims = analysis.multi_analysis(
            rep, repPermute, preprocFuns, simFuns, names=analysisNames
        )
        permuteData.loc[permute] = list(sims.values())

    permuteData.to_csv(outPath)


if __name__ == "__main__":
    # Setup argparse
    import argparse

    parser = argparse.ArgumentParser(
        description="Do some simulation anlaysis, inteded to be used on HPC"
    )
    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        required=True,
        help="analysis to perform",
        choices=["noise", "simulations", "sanity", "ablate", "sizeRatio"],
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        help="seed for random number generator",
        default=None,
    )
    parser.add_argument(
        "--nImgs",
        type=int,
        default=None,
        help="number images from the dataset to use",
    )
    parser.add_argument(
        "--outputPath",
        type=str,
        default=None,
        help="path to output file, if not specified will use default",
    )
    args = parser.parse_args()

    if args.analysis == "noise":
        parametricNoise(
            maxNoise=4.0,
            step=0.01,
            permutations=100,
            seed=args.seed,
            nImgs=args.nImgs,
            outputPath=args.outputPath,
        )
    elif args.analysis == "simulations":
        permuteTest(nImgs=args.nImgs, outputPath=args.outputPath)
    elif args.analysis == "sanity":
        sanity_check(nImgs=args.nImgs, outputPath=args.outputPath)
    elif args.analysis == "ablate":
        parametricAblation(nImgs=args.nImgs, outputPath=args.outputPath)
    elif args.analysis == "sizeRatio":
        sizeRatioTest(nImgs=args.nImgs, outputPath=args.outputPath)
    else:
        raise ValueError(f"Unknown analysis: {args.analysis}")
