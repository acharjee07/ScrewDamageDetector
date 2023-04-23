import torch
import torch.nn.functional as F
from typing import List
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from augmentations import valid_transforms
from config import Config
import glob
from dataprocessor import ScrewDatasetTest
from backbones import Effnet
from utils import ensemble_models
from utils import visual
import argparse










if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('test_data_path')
    parser.add_argument('weights_path')
    args=parser.parse_args()

    
  

    test_df=test_df=pd.DataFrame(glob.glob(args.test_data_path +'/*'),columns=['image_path'])
    test_data=ScrewDatasetTest(df=test_df,transform=valid_transforms)


    test_loader = DataLoader(test_data, batch_size=4, shuffle=True, pin_memory=True, drop_last=True, num_workers=2)



    weights_path=glob.glob(args.weights_path+'/*')

    models=[Effnet() for i in range(len(weights_path))]
    ens_out=ensemble_models(models,weights_path,test_loader,.5)


    test_results=pd.DataFrame(ens_out,columns=['predicitons','image_path','gt'])
    test_results['label']=test_results['predicitons'].map({j:i for i,j in Config.label_dict.items()})
    test_results=test_results.drop('gt',axis=1)
    print(test_results.head())
    visual(test_results,4)
    