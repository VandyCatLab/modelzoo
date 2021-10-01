import analysis
import tensorflow as tf
import numpy as np
import pandas as pd
import os


def permuteTest():
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

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

    permutePath = "../outputs/masterOutput/permuteSims.csv"
    if os.path.exists(permutePath):
        print("Using existing permutation test results")
        permuteData = pd.read_csv(permutePath)
    else:
        preprocFuns = [
            analysis.preprocess_rsaNumba,
            analysis.preprocess_svcca,
            analysis.preprocess_ckaNumba,
        ]
        simFuns = [
            analysis.do_rsaNumba,
            analysis.do_svcca,
            analysis.do_linearCKANumba,
        ]
        colNames = [fun.__name__ for fun in simFuns] + ["analysis"]
        nPermutes = 10000

        permuteData = pd.DataFrame(columns=colNames)

        # Random permutation
        print("Doing random simulation")
        df = pd.DataFrame(columns=colNames, index=range(nPermutes))
        for permute in range(nPermutes):
            if permute % 100 == 0:
                print(f"-- Permutation at {permute}")

            rep1 = np.random.choice(rep_flat, size=repShape, replace=False)
            rep2 = np.random.choice(rep_flat, size=repShape, replace=False)

            sims = analysis.multi_analysis(rep1, rep2, preprocFuns, simFuns)
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

            sims = analysis.multi_analysis(rep, repNoise, preprocFuns, simFuns)
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

            sims = analysis.multi_analysis(rep, repCut, preprocFuns, simFuns)
            df.loc[permute] = list(sims.values()) + ["ablate"]
        permuteData = permuteData.append(df)
        permuteData.to_csv(permutePath)


def sizeRatioTest():
    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    model = tf.keras.models.load_model(modelPath)
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    rep = model.predict(imgset).flatten()

    preprocFuns = [
        analysis.preprocess_rsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]
    colNames = [fun.__name__ for fun in simFuns] + ["sample", "features"]

    permuteSims = pd.DataFrame(columns=colNames)

    nMax = imgset.shape[0]
    ratiosRange = np.arange(0.05, 1, 0.05)
    nPermute = 10000

    for ratio in ratiosRange:
        print(f"Analyzing img:samples ratio: {ratio}")
        repShape = (nMax, int(nMax * ratio))
        for permute in range(nPermute):
            if permute % 100 == 0:
                print(f"Permutation at {permute}")

            # Sample reps
            rep1 = np.random.choice(rep, size=repShape)
            rep2 = np.random.choice(rep, size=repShape)

            sims = analysis.multi_analysis(rep1, rep2, preprocFuns, simFuns)
            sims["sample"] = repShape[0]
            sims["features"] = repShape[1]
            permuteSims = pd.concat(
                (
                    permuteSims,
                    pd.DataFrame.from_dict(sims, orient="index").transpose(),
                )
            )
    permuteSims.to_csv("../outputs/masterOutput/ratioSims.csv", index=False)


def parametricAblation(minNeuron=3, maxNeuron=10):
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

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
        analysis.preprocess_rsaNumba,
        analysis.preprocess_rsa,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_rsa,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]

    colNames = [fun.__name__ for fun in simFuns] + ["Neurons"]
    nPermutes = 10000

    permuteData = pd.DataFrame(columns=colNames)
    outPath = "../outputs/masterOutput/ablateSims.csv"
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

            sims = analysis.multi_analysis(rep, repCut, preprocFuns, simFuns)
            df.loc[nNeurons] = list(sims.values()) + [nNeurons]
        permuteData = permuteData.append(df)
    permuteData.to_csv(outPath)


def sanity_check():
    modelPath = "../outputs/masterOutput/models/w0s0.pb"
    print("Loading model")
    model = tf.keras.models.load_model(modelPath)

    # load dataset
    imgset = np.load("../outputs/masterOutput/dataset.npy")

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
        analysis.preprocess_rsaNumba,
        analysis.preprocess_svcca,
        analysis.preprocess_ckaNumba,
    ]
    simFuns = [
        analysis.do_rsaNumba,
        analysis.do_svcca,
        analysis.do_linearCKANumba,
    ]

    colNames = [fun.__name__ for fun in simFuns]
    nPermutes = 10000

    permuteData = pd.DataFrame(columns=colNames, index=range(nPermutes))
    outPath = "../outputs/masterOutput/sanityCheckSims.csv"
    print("Performing sanity check, permuting features.")
    for permute in range(nPermutes):
        if permute % 100 == 0:
            print(f"-- Permutation at {permute}")

        rep = np.random.choice(rep_flat, size=repShape, replace=False)
        repPermute = np.copy(rep)
        repPermute = repPermute[
            :, np.random.choice(rep.shape[1], size=repShape[1], replace=False)
        ]

        sims = analysis.multi_analysis(rep, repPermute, preprocFuns, simFuns)
        permuteData.loc[permute] = list(sims.values())

    permuteData.to_csv(outPath)


if __name__ == "__main__":
    sanity_check()
