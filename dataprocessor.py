
import glob
import cv2
from config import Config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import visual, visualize_dataset
from torch.utils.data import Dataset



def get_data_splits(test,data_path,label_dict,upsample,fold):
    if test:
        all_image_paths=glob.glob(data_path + '/*')
        df=pd.DataFrame(all_image_paths,columns=['image_path'])
        return df
    else:
        all_image_paths=glob.glob(data_path + '/**/*.png', recursive=True)
        df=pd.DataFrame([[path,path.split('/')[-2]] for path in all_image_paths],columns=['image_path', 'label'])
        df['num_label']=df['label'].map(label_dict)

        print('Training data label value counts \n',df['label'].value_counts())
        
        print('Some samples of training data')
        data_visual(df)
        df['def_label']=[label if label=='good' else  path.split('/')[-1].split('0')[0] for label,path in zip(df['label'],df['image_path'])]
        df['def_label'].value_counts()


        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        df = df.reset_index(drop=True)

        df["fold"] = -1

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['def_label'])):
            df.loc[val_idx, 'fold'] = fold


        val_df=df[df['fold']==fold]
        train_df=df.drop(val_df.index)
        print('training data distributuion',train_df['def_label'].value_counts())
        if upsample:
            df0=train_df[train_df['label']=='not-good']
            train_df=pd.concat([train_df,df0,df0,df0]).reset_index(drop=True)
            print('after upsampling data distribuition', train_df['def_label'].value_counts())

        val_df=val_df.reset_index(drop=True)
        train_df=train_df.reset_index(drop=True)

        return train_df,val_df
        


class ScrewDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['image_path'])
        label = row['label']
        if self.transform is not None:
          image = self.transform(image=image)['image']
        num_label=row['num_label']
        
        return image/255.0, F.one_hot(torch.tensor(num_label),num_classes=2).float()

    def __len__(self):
        return len(self.df)