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
    python3 ~/Predictor/predictor.py --dataset_path ~/Predictor/data/ --results_path ~/Predictor/results/
```

All the `.csv` files in the `\~/Predictor/data/` folder will be used for the training and evaluation, while the logs, checkpoints, outputs and errors will be stored in the `\~/Predictor/results/` folder