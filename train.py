
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from config import Config
from torch.utils.data import DataLoader
from dataprocessor import get_data_splits
from dataprocessor import ScrewDataset
from lit_model import ScrewModel
from backbones import Effnet
from augmentations import train_transforms,valid_transforms
from pytorch_lightning.callbacks import ModelCheckpoint



Config.debug=True
Config.folds=1




for fold in range(Config.folds):
    print('-----------------running for fold {}---------------------'.format(fold))
    if Config.debug:
            logger =None
    else:
        logger= TensorBoardLogger("logs/")


    train_df,valid_df=get_data_splits(test=False,data_path='',label_dict=Config.label_dict,upsample=True,fold=fold)
    if Config.debug:
        train_df=train_df[0:10]
        val_df=val_df[0:4]
    train_data=ScrewDataset(df=train_df,transform=train_transforms)
    valid_data=ScrewDataset(df=val_df,transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True, drop_last=True, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=4, shuffle=False, pin_memory=True, drop_last=True, num_workers=2)
    
    
    checkpoint_callback = ModelCheckpoint(monitor='pre0',
                                          dirpath= "/kaggle/working/fold{}".format(fold),
                                          save_top_k=2,
                                          save_last= True,
                                          save_weights_only=True,
                                          filename= '{epoch:02d}-{pre0:.4f}',

                                          verbose= True,
                                          mode='max')
    
    
    trainer = Trainer(
        max_epochs=Config.epochs,
        accelerator="gpu",
        log_every_n_steps=10,
        callbacks=checkpoint_callback,
        logger=logger

    )
    lit_model=ScrewModel(Effnet())
    trainer.fit(lit_model, train_dataloaders = train_loader, val_dataloaders = valid_loader)
    wandb.finish()