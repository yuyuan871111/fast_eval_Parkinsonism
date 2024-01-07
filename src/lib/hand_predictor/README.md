# Hand predictor API
## Quick start
Requirement: Miniconda or Anaconda.   
```bash
# install and activate environment via conda
conda env create -f environment.yml 
conda activate mediapipe

# change the working directory
cd ./src/lib/hand_predictor  # the path of the main script is as same as this README file

# find help messange
python hand_predictor.py -h

# get the testing file
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Jf9178R_U97Osu12cVudJvFQehtDI3U7' -O test.MOV

# main function for hand predictor API
python hand_predictor.py --wkdir_path . --seed 42 --filename test --ext MOV --hand_LR Left --hand_pos 1 --input_root_path . --output_root_path sample_output --mode single
```
