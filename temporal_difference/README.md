# Prerequisites
The scripts in this repo were built using the following:

* Python 3.7.4
* matplotlib 3.1.1
* numpy 1.17.2

Any version of Python 3 should work however

# Experiment 1
Experiment 1 is ran from **experiment1.py**. It has command line arguments to alter the execution. The defaults however are already configured to rerun the experiment exactly how it is presented in my paper. It will load the exact training set used and all other defaults.

```
python experiment1.py
```

To randomly generate a new training set, simply pass "false" to the CLI arguement **-l** or **--load_training_set**

```
python experiment1.py -l false
```
or 
```
python experiment1.py --load_training_set false
```

Other arguments are as follows:
* \[-t]\[--training_sets] - Number of training sets to be generated: Default = 100
* \[-s]\[--sequences] - Number of sequences per training set: Default = 10
* \[-a]\[--alpha] - Learning Rate alpha: Default = 0.2
* \[-v]\[--initial_values] - Initial values of states: Default = 0.0
* \[-g]\[--gamma] - Gamma value: Default = 1.0

### Note
Experiment 1 was coded as a multiprocess algorithm. It uses the multiprocess library and spawns a new process for every training set created. Using all defaults, it will spawn 100 individual processes.

Also note that if you attempt to generate new training data, there is no limit on how long the sequences can be. In the event of an extremely long transition, an overflow event can happen and will throw the error
```
RuntimeWarning: invalid value encountered in double_scalars
  values[pos] = val + alpha * update * eligibility[pos]
```

Simply kill the process with ctrl + z and and rerun the code

# Experiment 2
Experiment 2 is ran from **experiment2.py**. It has command line arguments to alter the execution. The defaults however are already configured to rerun the experiment exactly how it is presented in my paper. It will load the exact training set used and all other defaults

```
python experiment2.py
```

To randomly generate a new training set, simply pass "false" to the CLI arguement **-l** or **--load_training_set**

```
python experiment1.py -l false
```
or 
```
python experiment1.py --load_training_set false
```

Other arguments are as follows:
* \[-t]\[--training_sets] - Number of training sets to be generated: Default = 100
* \[-s]\[--sequences] - Number of sequences per training set: Default = 10

Experiment 2 creates 2 graphs: Figure 4 and 5 from Sutton's paper. The first graph to appear will be figure 4 recreation. After closing this figure, the figure 5 recreation will appear
