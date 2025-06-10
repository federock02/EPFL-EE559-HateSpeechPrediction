## 1. Building the Docker Image

From the `docker` folder:

- Place the `requirements.txt` file and `Dockerfile` inside a folder named after your Docker image (e.g., `ee559_docker_env`).
- Open a terminal inside this folder and run the following command to build the image:

```bash
docker build --platform linux/amd64 . \
  --tag registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1 \
  --build-arg LDAP_GROUPNAME=rcp-runai-course-ee-559_AppGrpU \
  --build-arg LDAP_GID=84650 \
  --build-arg LDAP_USERNAME=<username> \
  --build-arg LDAP_UID=<uid>P_GROUPNAME=rcp-runai-course-ee-559_AppGrpU --build-arg LDAP_GID=84650 --build-arg LDAP_USERNAME=<username> --build-arg LDAP_UID=<uid>
```

## 2. Pushing the Image to the Registry

After the image has been built successfully, push it to the EPFL registry:

```bash
    docker push registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1
```


## 3. Running the Python Script

You can launch the Python script either in a training job or an interactive job using the command:

```bash
    python3 ~/Predictor/predictor_....py --<arguments>
```

All the arguments that can be added to the command are:
- `--dataset_path`: all the `.csv` files in the folder indicated in this argument will be used for the training and evaluation (required)
- `--result_path`: the logs, checkpoints, outputs and errors will be stored in the folder indicated by this argument (required)
- `--debug`: including this flag argument runs the training with a limited number of data samples, with the number indicated in the `cfg.yaml` in the dataset folder (optional)
- `--no_freeze`: flag to indicate that the training be be run with all the layers of the transformers in training mode, so it does not freeze the first pretrained layers (optional)
- `--load_model_dir`: indicates the path of the trained model that needs to be loaded, either for finetuning or for testing (optional)
- `--finetune`: flag that indicates that the model loaded will be finetuned (optional)
- `--test`: flag to activate the testing mode (optional)

# Base Training

The possible architectures that can be trained are:
- `peredictor_BERT.py`: uses *BERT* encoder as base text transformer
- `predictor_BERT_RNN.py`: uses *BERT* encoder as base text transformer, followed by a *biLSTM*
- `predictor_GPT2.py`: uses *GPT2* decoder as base text transformer
- `predictor_GPT2_RNN.py`: uses *GPT2* decoder as base text transformer, followed by a *biLSTM*

The hyperparameters used in the training process are expected to be in a `cfg.yaml` file, in the folder indicated as `--dataset_path`.

The dataset that are expected in this phase consist of classification datasets, labeled with binary labels. 1 is the positive class, 0 is the negative class. The phrases to train on are sourced from the `text` column in the `.csv` files, and will be cut in all the possible sub-prefixes of various length during the dataset loading phase; the labels are sourced from the `label` column.

# Finetuning

Once one of the previous models has been trained, it can be finetuned using a dataset labeled with probabilities. To finetune a trained model, it must be loaded by passing the path to the results folder for the previously trained model as argument after `--load_model_dir`, followed by the flag `--finetune`.

The phrases should again be in the `text` column of the `.csv` file, and this time they are expected to be prefixed of phrases, with the associated probability labels in the `label` column. There should also be a column `weight` indicating the percentage of words in the considered prefix compared to the length of the whole phrase.

To train each one of the models listed above, the corresponding `..._hadcrafted.py` model should be used, in order to guarantee that the trained weights are loaded in the right architecture for finetuning. The possible models are:
- `peredictor_BERT_handcrafted.py`: uses *BERT* encoder as base text transformer, considers probability labels
- `predictor_BERT_RNN_handcrafted.py`: uses *BERT* encoder as base text transformer, followed by a *biLSTM*, considers probability labels
- `predictor_GPT2_handcrafted.py`: uses *GPT2* decoder as base text transformer, considers probability labels
- `predictor_GPT2_RNN_handcrafted.py`: uses *GPT2* decoder as base text transformer, followed by a *biLSTM*, considers probability labels

All these models can also be traiend from scratch, if run without loading the model and without the finetuning flag.

## 4. Dataset preprocessing
We provide four useful script for managing the datasets:
- `utils\cut_phrases.py`: loads a dataset of complete phrases, cuts them into all the possible prefixes of various lengths, adds the original label relative to the complete sentence and associates the length percentage weight for each prefix.
- `utils\add_weights.py`: useful when a dataset with prefixes is available, and the weight for each prefix needs to be computed. Only works if to each prefix is associated an index that connects it to the original complete phrase.
- `utils\check_dataset.py`: returns some statistics about the dataset, and can be used for binary-labeled datasets of complete phrases.
- `utils\check_prefix_dataset.py`: returns some statistics about the dataset, and can be used for probability-labeled datasets of prefixes.

## 5. Testing
All the trained models can be loaded, using the same `peredictor_....py` it was trained with, adding the path of the trained weights after `--load_model_dir` and adding the `--test` flag. The data used for testing is loaded from the folder linked by `--dataset_path`.