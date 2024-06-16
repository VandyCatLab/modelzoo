import os

import numpy as np
import numba as nb
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
import click
import pandas as pd
import pickle

import datasets


numpy2ri.activate()


# MARK: CLI
@click.group()
def cli():
    pass


@cli.command()
@click.option("--dim_min", type=int, default=2, help="Minimum number of dimensions")
@click.option("--dim_max", type=int, default=11, help="Maximum number of dimensions")
@click.option("--eps", type=float, default=1.0, help="Convergence criterion")
@click.option("--itmax", type=int, default=1500, help="Maximum number of iterations")
@click.option("--verbose", default=True, is_flag=True, help="Print output")
@click.option("--overwrite", default=False, is_flag=True, help="Overwrite existing")
def computeindscal(
    dim_min: int = 2,
    dim_max: int = 11,
    eps: float = 1.0,
    itmax: int = 1500,
    verbose: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Compute INDSCAL with dimenions in the range [minDims, maxDims). Most
    arguments are passed to the INDSCAL function.
    """
    dataNames = list(datasets._DATA_DIRS.keys())

    # Load the first dataset to get info
    data = pd.read_csv(f"../data_storage/sims/{dataNames[0]}.csv", index_col=0)
    modelNames = data.columns
    tmp = 1 - data.to_numpy() ** 2
    tmp[tmp < 0] = 0
    data = [tmp]

    for dataset in dataNames[1:]:
        tmp = pd.read_csv(f"../data_storage/sims/{dataset}.csv", index_col=0)

        # Reorder columns to match original
        tmp = tmp[modelNames]

        # Reorder rows to match original
        tmp = tmp.reindex(index=modelNames)

        # Check if the columns are correct
        if not np.array_equal(tmp.columns, modelNames):
            raise ValueError(f"Columns do not match for {dataset}")

        tmp = 1 - tmp.to_numpy() ** 2
        tmp[tmp < 0] = 0

        data += [tmp]

    for nDims in range(dim_min, dim_max):
        # Check if pickle of this solution exists
        outFile = f"../data_storage/indscal/indscal_d{nDims}_eps{eps}_itmax{itmax}.pkl"
        if os.path.exists(outFile) and not overwrite:
            click.echo(f"Solution already exists, skipping {nDims} dimensions")
            continue

        print(f"Running {nDims} dimensions")
        out = indscalR(data, nDims, eps=eps, itmax=itmax, verbose=verbose)

        # Save the solution
        with open(outFile, "wb") as f:
            pickle.dump(out, f)


# MARK: EucRSA Numba
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


# MARK: R Functions
def indscalR(
    delta: np.ndarray,
    ndim: int,
    eps: float = 1.0,
    itmax: int = 1500,
    verbose: bool = True,
) -> dict:
    """
    Return INDSCAL results where X is the distance cube and nfac is the number
    of factors. This function is a wrapper around the R indscal function.
    """
    smacof = importr("smacof")
    npConverter = default_converter + numpy2ri.converter

    with npConverter.context():
        solution = smacof.indscal(
            delta=delta, ndim=ndim, eps=eps, itmax=itmax, verbose=verbose, type="ratio"
        )

    # Just keep the relevant data, these objects are huge
    conf = np.stack([value for key, value in list(solution["conf"].items())], axis=2)
    cweights = np.stack(
        [value for key, value in list(solution["cweights"].items())], axis=2
    )
    out = {
        "gspace": solution["gspace"],
        "cweights": cweights,
        "conf": conf,
        "stress": solution["stress"],
    }
    return out


if __name__ == "__main__":
    cli()