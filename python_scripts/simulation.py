import analysis
import tensorflow as tf
import numpy as np
import pandas as pd


if __name__ == "__main__":
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
    ratiosRange = np.arange(0.01, 1, 0.01)
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
