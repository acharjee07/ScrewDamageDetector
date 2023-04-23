import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from config import Config
from matrics import get_classification_metrics
from backbones import effnet


class ScrewModel(pl.LightningModule):
    def __init__(self, model,Config):
        super().__init__()
        self.backbone=model
        self.config=Config
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.loss_function = nn.BCELoss()
        self.validation_step_outputs = []

    def forward(self,images):
        logits = self.backbone(images)
        # out=self.softmax(logits)
        return logits
    
    def configure_optimizers(self):
        self.optimizer=torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.backbone.parameters()), 
            lr=self.config.LR,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler=OneCycleLR(self.optimizer, max_lr=self.config.LR, epochs=self.config.epochs, steps_per_epoch=self.config.steps_per_epoch,pct_start=.5)


        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}
        return [self.optimizer], [scheduler]
  

    def training_step(self, batch, batch_idx):
        image, target = batch        
        y_pred = self.backbone(image)
        out=self.softmax(y_pred)
       
        
        loss = self.loss_function(out,target)
        preds_thresholded = (out >= 0.5).int()

        # compute accuracy
        accuracy = (preds_thresholded == target).all(dim=1).float().mean()

        logs={'train_loss':loss,'train_acc':accuracy.item(),'lr':self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True


        )
        return loss        

    def validation_step(self, batch, batch_idx):
        image, target = batch     
        y_pred = self.backbone(image)
        out=self.softmax(y_pred)
        val_loss = self.loss_function(out, target)

        
  
        # logs={'valid_loss':val_loss,'val_acc':val_acc,'val_precision':precision,'val_recall':recall,'val_f1':f1}
        logs={'valid_loss':val_loss}
        self.validation_step_outputs.append({'val_loss':val_loss.detach().cpu(),'logits':out.detach().cpu(),'targets':target.cpu()})
        
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True


        )
        
        return {"val_loss": val_loss, "logits": y_pred, "targets": target}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()

        output_val = torch.cat([x['logits'] for x in self.validation_step_outputs],dim=0).cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in self.validation_step_outputs],dim=0).cpu().detach().numpy()
        out_df=pd.DataFrame([[np.where(np.array(x)>.5)[0][0],np.where(np.array(y)>.5)[0][0] ]for x,y in zip(output_val,target_val)],columns=['output','target'])
        out_df.to_csv('out.csv')
        metrics=np.array(get_classification_metrics(out_df,2))
        
        logs={'acc':metrics[0][0],'pre0':metrics[0][1],'pre1':metrics[1][1],'rec0':metrics[0][2],'rec1':metrics[1][2]}

        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
            
        
        )
        return {'acc0': metrics[0][0],'acc1':metrics[1][0]}
 
lit_model=ScrewModel(effnet,Config)