## Dependencies

* Intel's distribution for python ([IDP](https://software.intel.com/en-us/distribution-for-python))
* Intel's Math kernel library ([MKL](https://software.intel.com/en-us/mkl))
* Pickle
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [Cython](http://cython.org/)

Assuming that [Anaconda](https://www.anaconda.com/download) is installed on the machine, a virtual environment (named idp_full) that satisfies the dependencies for training is created by running:
```bash
./create_environment.sh
```
If you have an older version of Anaconda, the creation of the environment will crash, since some of the libraries need to be updated. Thus, you might need to run:
```bash
conda update conda
```
before running `create_environment.sh`.
___
## Training

#### Training requirements

In order to run the training/evaluation code, in addition to the datasets that contain the documents themselves, the algorithm needs the following data:
* training_concepts: they define the concepts used in training. (The file is needed in training)
* validation_concepts: they define the concepts used for validation. The validation concepts are also excluded from training. (The file is needed in training and evaluation)
* target_concepts: they defines the concepts that make up the target space (retrieval search space) in the validation phase. (The file is needed in evaluation)

Additionally the training uses some code written in cython, which needs to be compiled. Hence you need to run the `setup.py` script before training, using:

```bash
python setup.py build_ext --inplace
```

#### Directory structure

The scripts rely on data being stored according to a certain structure, thus it is essential that the right file paths are entered in the beginning of the `utils.py` script.

In addition to that, if the data directory structure is not already generated, run the script `create_folder_structure.sh` to generate it. The previous needs to be passed the home directory for the data as a parameter. If you haven't updated the parameter in `utils.py`,  run the script using the default parameters by:

```bash
./create_folder_structure.sh ./..
```

#### Data

Download the training and evaluation [data](https://zenodo.org/record/3359814#.XUdP85MzZQI). Unzip it in the data folder.

#### Training, validation and target concept files

Generate training concepts and validation concepts file by running:

```bash
./generate_training_validaiton_concepts.sh
```
___
Once you have generated the training and validation concepts files, the experiment wrapper should be updated with the desired training/validation dataset files, as well as the training parameters' search range, and sequentially run, using:
```bash
python experiment_wrapper.py
```
___
## Evaluation

The evaluation is done on the cross-lingual document retrieval task. All the details on the possibilities, and the usage of the evaluator class are given in the `eval_example.ipynb` notebook.

### Baseline embeddings

In the experiments we use the state-of-the-art at the time of writing as baseline (details [here](https://arxiv.org/pdf/1710.04087.pdf)). The script `get_baseline_embeddings.sh` is used to retrieve embeddings for a desired language. Alternatively the baseline embeddings that are needed for this paper are available for downloading [here](https://zenodo.org/record/3359814#.XUdP85MzZQI).

*To run the notebook above, make sure you have downloaded the needed baseline embeddings.*
___
## Scripts included in the directory and their purpose

#### Python scripts
* `cr5.py`: Helper library with function that help you easily load the pre-trained models (link on the previous page) and use them.

* `example.py`: Example on how to use the `cr5.py` library.

* `counting_words_per_concepts.py`: As the name implies, this script counts the number of unique words for all articles in a language. The number of words from the full vocabulary that is considered in the counting is passed as an argument. 
Generates an object that contains the unique word counts for each concept in a given language.

* `training_concepts.py`: Filters documents based on the number of unique words in them (file generated from `counting_words_per_concepts.py`), as well as the number of languages that a given concept has an article in.
Generates a training concepts object that defines the concepts that should be used in training for the language codes passed as argument, based on the chosen criteria.

* `training_concepts_no_intersection.py`: Similar as previously, with the addition of the parameter intersection_to_exclude that represents a pair of languages whose intersection is excluded from the training concepts.
Generates a training concepts object that defines the concepts that should be used in training for the language codes passed as argument, based on the chosen criteria.

* `validation_concepts.py`: Samples concepts from a training_concepts object (a file structured as the output from `training_concepts.py` script) to generate evaluation sets for every pair of languages in the training_concepts object. 
Generates a validation concepts object that defines the held-out set of concepts used in the evaluation stage.

* `target_concepts.py`: Samples concepts from a training_concepts object (a file structured as the output from `training_concepts.py` script) to generate a target concepts sets for every language that defines the target space in the evaluation phase.
Generates a target concepts object that defines the target space (retrieval search space) in the evaluation stage.

* `utils.py`: Module that defines the home directory with its structure and groups together all the functions with global usability. 

* `data_class.py`: A class wrapper around all of the important functions involved in filtering, manipulating and loading of the data used in training, as well as in evaluation.

* `operations_wrapper.py`: A class wrapper for all the optimised routines used in the training phase. It contains everything needed from the computational aspect of the problem solution.

* `logger.py`: Defines a class, that is used to coordinate and enforce a systematic logging schema, for the experiments parameters and their training progress, in order to facilitate the results search, retrieval and result parsing for all runs.

* `experiment_run.py`: Represents one run of the training pipeline, with the desired parameters passed in the constructor as an argument.

* `experiment_wrapper.py`: For training and concepts validation files given as parameters, it performs a grid search over a predefined parameter space i.e. performs multiple runs using the same training and validation concept files, but with different parameters. It dumps the results in a very structured way that makes it really easy to retrieve results and evaluate.

* `evaluator.py`: A class that contains all of the functions needed for evaluation. It enables individual evaluation (one combination of parameters) by only supplying a results object produced by an experiment run. Additionally, it provides multiple runs evaluation (different combinations of parameters using the same data) by supplying the logs file for the experiment (equivalent to the identifier of an experiment).
* `multiply_cython.pyx`: Cython implementation of a multiplication function used in training.
* `setup.py`: Script used to compile the cython file and produce a python importable object
* `generate_concept_id_concept_name_mapping.py` : Generates a mapping from the generated concept id to the user friendly concept name. The resulting dictionary is used in the evaluation phase, in order to make the results more interpretable. *In order for the evaluator to work, the dictionary must exist in the data folder.*
* `test_matrix_operations.py`: Contains unit tests for the routines in the operations wrapper.
* `old_data_class`: Obsolete. Initial implementation of the data_class. Kept because it is used in test_matrix_operations.

#### Bash scripts
* `create_environment.sh`: Creates an environment that satisfies all the dependencies, and ensures high performance on machines with high number of cores.
* `create_folder_structure.sh`: Creates the needed folder structure. (If not generated)
* `generate_training_validaiton_target_concepts.sh`: Generates everything you need in order to run the experiments for multiple languages. 
*Can serve as a useful example of the steps needed before training.*
* `get_baseline_embeddings.sh`: As the name implies, retrieves baseline embeddings for a given language.

___

## Any questions or suggestions?
You are welcome to contact me at martin.josifoski@epfl.ch. 
