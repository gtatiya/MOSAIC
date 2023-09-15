# Author: Gyan Tatiya

import logging
import numpy as np
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
import torchvision.models as models
import lightning.pytorch as pl
from torchmetrics import Accuracy

import wav2clip
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection


class Encoder(nn.Module):
    def __init__(self, input_dim, h_dims, out_dim, h_activ=None, out_activ=None):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        num_layers = len(layer_dims) - 1
        layers = []
        for index in range(num_layers):
            layer = nn.Linear(layer_dims[index], layer_dims[index + 1])

            if h_activ and index < num_layers - 1:
                layers.extend([layer, h_activ])
            elif out_activ and index == num_layers - 1:
                layers.extend([layer, out_activ])
            else:
                layers.append(layer)

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class AdaptivePoolTime1d(nn.Module):
    """Adaptive pooling over time axis

    Pooling module shall be provided to act on last axis of input tensor
    """

    def __init__(self, pooling: nn.Module):
        """Initialize AdaptivePoolTime1d

        Args:
            pooling: 1d pooling module, e.g. torch.nn.AdaptiveMaxPool1d
        """
        super().__init__()
        self.pooling = pooling

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward step

        Args:
            inputs: (batch, frames, embedding)

        Returns:
            (batch, embedding)

        """
        
        inputs = torch.swapaxes(inputs, 1, 2)
        output = self.pooling(inputs)
        output = torch.squeeze(output, dim=-1)

        return output


class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric cross entropy loss as in CLIP paper"""

    def __init__(self, tau: float, *, input_layer: Literal["proj", "emb"] = "proj") -> None:
        super().__init__()
        # As per CLIP paper, temperature is learnable
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
        # Specify what inputs are needed for loss computation. 'emb' - embeddings; 'proj' - projection of embeddings
        self.loss_args = [input_layer]

    def forward(self, **kwargs) -> Tensor:
        """Compute Symmetric Cross Entropy Loss

        Args:
            kwargs: dict of embedding or projection from modalities, (N, P); Assume L2 normalized.
                    with `input_layer=emb`, kwargs is expected to have two keys `emb1`,`emb2`
                    with `input_layer=proj`, kwargs is expected to have two keys `proj1`,`proj2`

        Returns:
            loss
        """

        x1, x2 = kwargs[f"{self.loss_args[0]}1"], kwargs[f"{self.loss_args[0]}2"]

        logits1 = torch.exp(self.logit_scale) * (x1 @ x2.t())
        logits2 = logits1.t()
        labels = torch.arange(len(logits1), device=logits1.device)
        loss1 = F.cross_entropy(input=logits1, target=labels)
        loss2 = F.cross_entropy(input=logits2, target=labels)
        loss = (loss1 + loss2) / 2

        return loss


class CategoryClassifier(pl.LightningModule):
    def __init__(self, dataset_module, output_size, batch_size: int = 32, learning_rate: float = 0.0001):
        super().__init__()

        self.dataset_module = dataset_module
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dataset_module.batch_size = batch_size

        input_channels = self.dataset_module.dataset.num_features
        hidden_layer_sizes = [input_channels//2]
        h_activation_fn, out_activation_fn = nn.ReLU(), None
        self.model = Encoder(input_channels, hidden_layer_sizes, output_size, h_activation_fn, out_activation_fn)

        self.accuracy = Accuracy(task='multiclass', num_classes=output_size)

        self.save_hyperparameters(ignore=['dataset_module'])

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)
    
    def common_step(self, batch, batch_idx):
        
        batch['unify_repr'] = batch['unify_repr'].to(self.device)

        y = []
        for object_name in batch['object_name']:
            category = object_name.split('_')[0]
            y.append(self.dataset_module.dataset.metadata['category_ids'][category])
        y = torch.Tensor(y).to(self.device).type(torch.int64)
        
        logits = self.model(batch['unify_repr'])
        
        # training metrics
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = self.accuracy(y_hat, y)
        proba = F.softmax(logits, dim=1)

        return loss, acc, y, y_hat, proba

    def training_step(self, batch, batch_idx):

        loss, acc, y, y_hat, proba = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        np.random.shuffle(self.dataset_module.train_idx)
    
    def validation_step(self, batch, batch_idx):

        loss, acc, y, y_hat, proba = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):

        loss, acc, y, y_hat, proba = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, prog_bar=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        return self.dataset_module.train_dataloader()


class UnifyRepresentations(pl.LightningModule):
    def __init__(self, dataset_module, modalities_train, batch_size: int = 32, learning_rate: float = 0.0001,
                 use_mha=False):
        super().__init__()

        self.dataset_module = dataset_module
        self.modalities_train = modalities_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_mha = use_mha

        self.dataset_module.batch_size = batch_size

        unify_input_size = 0
        unify_output_size = 512
        for modality in modalities_train:
            if modality == 'audio':
                self.audio_model = wav2clip.get_model()
                self.audio_model.requires_grad_(modalities_train[modality])
                unify_input_size += self.audio_model.transform.sequential[3].out_features
            elif modality == 'text':
                self.text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
                self.text_model.requires_grad_(modalities_train[modality])
                self.text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
            elif modality == 'video':
                self.video_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
                self.video_model.requires_grad_(modalities_train[modality])
                self.video_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
                self.video_aggregation = AdaptivePoolTime1d(nn.AdaptiveAvgPool1d(output_size=1))
                unify_input_size += self.video_model.visual_projection.out_features
            elif modality == 'haptic':
                self.haptic_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Initialize model with the best available weights
                self.haptic_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 channel in input for haptic instead of 3 in image                
                self.haptic_model.fc = nn.Linear(self.haptic_model.fc.in_features, unify_output_size) # modifying output classes
                self.haptic_model.requires_grad_(modalities_train[modality])
                unify_input_size += unify_output_size
            else:
                raise Exception(modality + ' does not exits!')
        
        if use_mha:
            self.mha = MultiheadAttention(unify_output_size, num_heads=8)
        
        if unify_input_size > unify_output_size:
            hidden_layer_sizes = [unify_input_size//2]
            self.unify_model = Encoder(unify_input_size, hidden_layer_sizes, unify_output_size)
        
        self.sce_loss = SymmetricCrossEntropyLoss(tau=0.07)

        self.save_hyperparameters(ignore=['dataset_module'])
    
    def common_step(self, batch, batch_idx):
        
        modalities_x = {}
        for modality in self.modalities_train:
            if modality != 'text':
                batch[modality] = batch[modality].to(self.device)
            x = []
            if modality == 'text':
                x = self.text_tokenizer(batch[modality], padding=True, return_tensors='pt')['input_ids'].to(self.device)
            elif modality == 'video':
                for video in batch[modality]:
                    inputs = self.video_processor(images=video, return_tensors='pt')
                    x.append(inputs['pixel_values'])
                x = torch.Tensor(np.stack(x)).to(self.device)
            elif modality == 'haptic':
                x = torch.unsqueeze(batch[modality], 1)  # create an axis for channel
            elif modality == 'audio':
                x = batch[modality]
            else:
                raise Exception(modality + ' does not exits!')
            modalities_x[modality] = x

        uni_embeds = []
        for modality in sorted(self.modalities_train):
            x = modalities_x[modality]
            if modality == 'video':
                # Process frames independently
                batch_size, num_frames, num_channels, heigh, width = x.size()
                frames = torch.reshape(x, (batch_size * num_frames, num_channels, heigh, width))
                outputs = self.video_model(pixel_values=frames)
                outputs = outputs.image_embeds

                # Reshape to (batch, frames, embedding_size)
                outputs = torch.reshape(outputs, (batch_size, num_frames, -1))

                # Aggregate
                embeds = self.video_aggregation(outputs)
                uni_embeds.append(embeds)
            elif modality == 'text':
                text_embeds = self.text_model(x).text_embeds
            elif modality == 'haptic':
                embeds = self.haptic_model(x)
                uni_embeds.append(embeds)
            elif modality == 'audio':
                embeds = self.audio_model(x)
                uni_embeds.append(embeds)
            else:
                raise Exception(modality + ' does not exits!')
        
        if hasattr(self, 'mha'):
            uni_embeds = torch.stack(uni_embeds, dim=0)  # [sequence=num_modalities, batch, embed_dim]
            # Compute self-attention across modalities
            mha_out, weights = self.mha(uni_embeds, uni_embeds, uni_embeds)
            # residual/skip connection: smoother gradient flow during training and mitigate the risk of vanishing gradients
            mha_out += uni_embeds
            uni_embeds = torch.concat([mha_out[i] for i in range(mha_out.shape[0])], 1)
        else:
            uni_embeds = torch.cat(uni_embeds, dim=-1)

        if hasattr(self, 'unify_model'):
            uni_embeds = self.unify_model(uni_embeds)

        # training metrics
        loss_args = {'proj1': uni_embeds, 'proj2': text_embeds}
        loss = self.sce_loss(**loss_args)

        info = {'object_name': batch['object_name'], 'trial_num': batch['trial_num']}

        return loss, uni_embeds, info

    def training_step(self, batch, batch_idx):

        loss, uni_embeds, info = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        np.random.shuffle(self.dataset_module.train_idx)
    
    def validation_step(self, batch, batch_idx):

        loss, uni_embeds, info = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):

        loss, uni_embeds, info = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def train_dataloader(self):
        return self.dataset_module.train_dataloader()
