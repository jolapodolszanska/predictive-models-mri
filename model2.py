import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

class HybridCNN(pl.LightningModule):
    def __init__(self):
        super(HybridCNN, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        resnet_out_features = self.resnet.fc.in_features
        inception_out_features = self.inception.fc.in_features
        
        self.resnet.fc = nn.Identity()
        self.inception.fc = nn.Identity()

        self.fc1 = nn.Linear(resnet_out_features + inception_out_features, 512)
        self.fc2 = nn.Linear(512, 4)
        self.dropout = nn.Dropout(0.5)

        self.val_precision = MulticlassPrecision(num_classes=4, average='macro').to(self.device)
        self.val_recall = MulticlassRecall(num_classes=4, average='macro').to(self.device)
        self.val_f1 = MulticlassF1Score(num_classes=4, average='macro').to(self.device)

        self.validation_preds = []
        self.validation_labels = []
        self.validation_features = []

    def forward(self, x):
        resnet_output = self.resnet(x)
        inception_output = self.inception(x)
        if isinstance(inception_output, tuple):
            inception_output = inception_output[0]
        combined_features = torch.cat((resnet_output, inception_output), dim=1)
        combined_features = self.dropout(combined_features)
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        # Wagi strat
        weights = torch.tensor([1.0, 7.0, 1.0, 2.0]).to(self.device)  # Ustawienie wag dla klas
        loss = F.cross_entropy(outputs, labels, weight=weights)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
    
        # Zbieranie cech (features) w trakcie walidacji
        self.validation_preds.append(preds.cpu())
        self.validation_labels.append(labels.cpu())
        self.validation_features.append(outputs.detach().cpu())  # Dodajemy cechy do listy
        
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)
        f1_score = self.val_f1(preds, labels)
    
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1_score', f1_score, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "labels": labels}

    def on_validation_epoch_end(self):
        if self.validation_preds and self.validation_labels and self.validation_features:
            self.validation_preds = torch.cat(self.validation_preds)
            self.validation_labels = torch.cat(self.validation_labels)
            self.validation_features = torch.cat(self.validation_features)
        
            print(f"Validation epoch end - preds: {self.validation_preds.size()}, labels: {self.validation_labels.size()}, features: {self.validation_features.size()}")
        else:
            print("No features were collected during validation.")
        
        self.validation_preds = []  # Reset after each epoch
        self.validation_labels = []  # Reset after each epoch
        self.validation_features = []  # Reset after each epoch

class ConfusionMatrixCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if len(pl_module.validation_preds) > 0 and len(pl_module.validation_labels) > 0:
            preds = torch.cat(pl_module.validation_preds).numpy()
            labels = torch.cat(pl_module.validation_labels).numpy()
            conf_matrix = confusion_matrix(labels, preds)
            
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')

            # Dodanie oznaczeń do osi
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')
            
            trainer.logger.experiment.add_figure("Confusion Matrix", fig, global_step=trainer.current_epoch)
            plt.close(fig)

class TSNECallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        print("t-SNE callback triggered")
        if len(pl_module.validation_features) > 0 and len(pl_module.validation_labels) > 0:
            features = torch.cat(pl_module.validation_features).numpy()
            labels = torch.cat(pl_module.validation_labels).numpy()
            
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
            plt.figure(figsize=(10, 10))
            for label in np.unique(labels):
                indices = labels == label
                plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {label}')
            plt.legend()
            plt.title('t-SNE projection of the features')

            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            trainer.logger.experiment.add_figure("t-SNE Projection", plt.gcf(), global_step=trainer.current_epoch)
            plt.close()

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32):
        super(CustomDataModule, self).__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),  # Augmentacja
            transforms.RandomRotation(10),  # Augmentacja
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

if __name__ == '__main__':
    dataset_path = 'D:/Badania/embedded/Dataset'
    data_module = CustomDataModule(dataset_path)

    model = HybridCNN()
    
    logger = TensorBoardLogger('tb_logs', name='CNN-corection')
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, verbose=True)
    progress_bar = ProgressBar()
    
    confusion_matrix_callback = ConfusionMatrixCallback()
    tsne_callback = TSNECallback()

    trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1, logger=logger, callbacks=[checkpoint_callback, progress_bar, confusion_matrix_callback, tsne_callback])
    trainer.fit(model, datamodule=data_module)
