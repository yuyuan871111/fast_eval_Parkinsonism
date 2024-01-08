# FastEval Parkinsonism
FastEval Parkinsonism
(https://fastevalp.cmdm.tw/) is a deep learning-driven video-based system, providing
users to capture keypoints, estimate the severity, and summarize in a report.
![cover_image](./imgs/cover.png)


## Local installation for the web with all functions
### Fast Installation at local by docker-compose
```bash
git clone <github-link>
# e.g. git clone git@github.com:CMDM-Lab/fasteval_parkinsonism.git
docker-compose up --build # for building
```

### Start
```bash
docker-compose up # start server or reactivate server
```

### Show
Open your browser and use the `https://localhost:13006`

### Stop
```bash
docker-compose down # stop server
# or use `ctrl+C`
```

### look at the linux environment
```bash
docker compose run web /bin/bash
```

## Hand Predictor API
Read the document for indenpendently usage ([Hand Predictor API](./src/lib/hand_predictor)).

### Quick start with Hand Predictor API merely
Requirement: conda.  
```bash
# clone the repository from github
git clone git@github.com:yuyuan871111/fast_eval_Parkinsonism.git    # by ssh
git clone https://github.com/yuyuan871111/fast_eval_Parkinsonism.git    # by html

# install and activate environment via conda
conda env create -f environment.yml # only do this for the first time
conda activate mediapipe

# change the working directory
cd ./src/lib/hand_predictor  # the path of the main script is as same as hand_predictor's README file

# find help messange
python hand_predictor.py -h

# get the testing file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Jf9178R_U97Osu12cVudJvFQehtDI3U7' -O test.MOV

# main function for hand predictor API
python hand_predictor.py --wkdir_path . --seed 42 --filename test --ext MOV --hand_LR Left --hand_pos 1 --input_root_path . --output_root_path sample_output --mode single
```


### Website development  
Read the document for indenpendently usage ([Website development](./src)).  


### Other informations
1. More information about alternative installation in [TroubleShooting](./TroubleShooting.md).  
2. If you face a problem with message `A server is already running. Check /home/myuser/local/app/tmp/pids/server.pid.`, please look at the linux environment and remove the file `server.pid` (shown as the codes below).  
```bash
docker compose run web /bin/bash                # run the docker compose environment
rm /home/myuser/local/app/tmp/pids/server.pid   # remove the file
```
