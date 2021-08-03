import sys
import os
sys.path.append(os.getcwd())
import  yaml


# Load global configuration
def load_yamlcfg(config_file):
    """Load global configuration
        Load model parameters and training hyperparameters for training models with different data sets
    """

    with open(config_file, 'r', encoding='utf-8') as f:
        # Read the content of the yaml file
        config  =  yaml . load ( f , Loader = yaml . FullLoader )

    return config