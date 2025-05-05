# DOCKER IMAGE
From the 'docker' folder, 'requirements.txt' file is to be inserted in a folder called with the name for the docker image (e.g. 'ee559_docker_env'), together with the 'Dockerfile'. Then, with a terminal initiated in the docker_env folder, the image can be created with the command:
    docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1 --build-arg LDAP_GROUPNAME=rcp-runai-course-ee-559_AppGrpU --build-arg LDAP_GID=84650 --build-arg LDAP_USERNAME=<username> --build-arg LDAP_UID=<uid>
The image has to then be pushed to the registry:
    docker push registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1

# LAUNCHING PYTHON SCRIPT
The python script can be launched either in a training job or in an interactive job with the following command:
    python3 ~/Predictor/predictor2.py --dataset_path ~/Predictor/data/ --results_path ~/Predictor/results/

All the .csv files in the '~/Predictor/data/' folder will be used for the training and evaluation, while the logs, checkpoints, outputs and errors will be stored in the '~/Predictor/results/'