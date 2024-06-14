# Model Zoo

This repository requires an annoying combination of very specific dependencies
to ensure that Tensorflow and Pytorch (and their respect model hubs) will work
together.

## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
The requirements.txt probably works but if it doesn't, manually installing the
important packages should work, provided the exact right versions are installed.

Note that we use `torch==2.3.1+cu118` and `tensorflow==2.14.1` because both of
these use CUDA 11.8, which should be supported by the `nvidia-driver-545`. CUDA
11.8 should still probably be manually installed to avoid issues of dependencies
crashing together (even if installing Tensorflow with the CUDA extras 
`tensorflow[and-cuda]==2.14.1`). Importantly, Tensorflow Hub must be pinned
for installation (e.g., `tensorflow-hub==0.15.0`) as later version will not work
as it depends on the latest version of Keras, which is not compatible with our
specific version of tensorflow. 