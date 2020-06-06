Run on ACCRE, ensure the same nodelist for each instance

Run 3 instances, same seed, same hash, shuffle = True. Also run 1 instnace with different seed.

Hypothesis: Setting the hash should ensure the shuffle is the same every instance. Different seed should be the
only output that is different.
