This batch was trained with a few improvements over version 1

1. Better accuracy: More correct setup of networks and data preprocessing
2. Trained on CPU: GPU introduces some unaccounted-for variance while we want to solely isolate init seed
3. Incorporated trajectory snapshots
4. No shuffle: Again, good for isolating variance

Mistakes that need to be addressed in version 3

1. Grabbed the wrong layer. Need to retrain and get layer before the 10-deep conv layer
2. Maybe in future version (i.e. not urgent) but an exact replication of the paper's models would be nice
3. Is the variance thing fixed? Somehow training with seed 0 is different now than it was before...
