"""
Builds the neural network
"""

import logging
from datetime import datetime

import lightning as L
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import BINARY_MODE, DiceLoss,FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from torchviz import make_dot
from torchmetrics import Accuracy, F1Score, JaccardIndex

from .layers.blocks import (ConvNextV2, ConvNextV2Block, ConvolutionalBlock,
                            DecoderUpsampleBlock, EncoderDecoderConnections,
                            PrintLayer, SegmentationHead)

import torch.nn.functional as F
import monai.networks.nets as mnets
import wandb
import torchvision.transforms as T
import numpy as np

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha  # Focal loss alpha (controls balance between classes)
        self.gamma = gamma  # Focal loss gamma (controls focus on hard examples)
        self.smooth = smooth  # Smoothing factor for Dice loss

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors for Dice loss computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Dice Loss
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()
        
        # Combine Dice and Focal Losses
        combined_loss = dice_loss + focal_loss
        return combined_loss

def create_logger(output_log_path: str) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: str
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/stitch_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


class ConvNeXtV2Encoder(torch.nn.Module):
    def __init__(self, model_name, in_channels, out_channels, depth, depths):
        super().__init__()
        self._depth = depth
        self._out_channels = [in_channels] + out_channels
        self._encoder = ConvNextV2(
            depth=depth,
            depths=depths,
            in_channels=in_channels,
            dims=out_channels,
        )

    def forward(self, x):
        return self._encoder.forward_features(x)

# class EncoderPath(nn.Module):
#     """
#     Encoder path of the Unet
#     """

#     def __init__(self, *args, **kwargs) -> None:
#         super(EncoderPath, self).__init__(*args, **kwargs)
#         ### First block
#         self.conv_1 = ConvNextV2Block(
#             in_channels=1,
#             out_channels=16,
#             stride=1,
#             kernel_size=7,
#             padding="same",
#             point_wise_scaling=2,
#         )

#         self.max_1 = nn.MaxPool3d(
#             kernel_size=2
#         )

#         self.conv_2 = ConvNextV2Block(
#             in_channels=16,
#             out_channels=32,
#             stride=1,
#             kernel_size=7,
#             padding="same",
#             point_wise_scaling=2,
#         )
        
#         self.max_2 = nn.MaxPool3d(
#             kernel_size=2
#         )

#         self.conv_3 = ConvNextV2Block(
#             in_channels=32,
#             out_channels=64,
#             stride=1,
#             kernel_size=7,
#             padding="same",
#             point_wise_scaling=2,
#         )

#         self.max_3 = nn.MaxPool3d(
#             kernel_size=2
#         )

#         self.conv_4 = ConvNextV2Block(
#             in_channels=64,
#             out_channels=128,
#             stride=1,
#             kernel_size=3,
#             padding="same",
#             point_wise_scaling=2,
#         )

#         self.skip_connections = ()
#         # self.print_layer = PrintLayer()

#         # Initializing
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.constant_(m.bias, 0)

#     def get_skip_connections(self):
#         return self.skip_connections

#     def forward(self, x):
#         # Input: N, C, D, H, W | output: N, out_channels.conv1, D, H, W
#         # print("Encoder ", x.shape)

#         ########## FIRST BLOCK ##########
#         # Input: N, 1, 128, 128, 128 | output: N, 16, 128, 128, 128
#         skip_1 = self.conv_1(x)
#         # self.print_layer(x, m="Conv 1")
        
#         # Input: N, 16, 128, 128, 128 | output: N, 16, 128, 128, 128
#         # skip_1 = self.conv_2(x)
#         # self.print_layer(skip_1, m="Conv 2 - Skip")

#         # Input: N, 16, 128, 128, 128 | output: N, 16, 64, 64, 64
#         x = self.max_1(skip_1)

#         ########## SECOND BLOCK ##########
#         # Input: N, 16, 64, 64, 64 | output: N, 32, 64, 64, 64
#         skip_2 = self.conv_2(x)
#         # self.print_layer(x, m="Conv 2")
        
#         # # Input: N, 32, 64, 64, 64 | output: N, 32, 64, 64, 64
#         # skip_2 = self.conv_4(x)
#         # self.print_layer(skip_2, m="Conv 4")

#         # Input: N, 32, 32, 32, 32 | output: N, 32, 16, 16, 16
#         x = self.max_2(skip_2)

#         ########## THIRD BLOCK ##########
#         # Input: N, 32, 32, 32, 32 | output: N, 64, 32, 32, 32
#         skip_3 = self.conv_3(x)
#         # self.print_layer(x, m="Conv 3")
#         # Input: N, 64, 32, 32, 32 | output: N, 64, 32, 32, 32
#         # skip_3 = self.conv_6(x)
#         # self.print_layer(skip_3, m="Conv 6")

#         # Input: N, 64, 32, 32, 32 | output: N, 64, 16, 16, 16
#         x = self.max_3(skip_3)

#         ########## FOURTH BLOCK ##########
#         # Input: N, 64, 16, 16, 16 | output: N, 128, 16, 16, 16
#         x = self.conv_4(x)
#         # self.print_layer(x, m="Conv 4")
#         # Input: N, 128, 16, 16, 16 | output: N, 128, 16, 16, 16
#         # skip_4 = self.conv_8(x)
#         # self.print_layer(skip_4, m="Conv 8")

#         # Input: N, 128, 16, 16, 16 | output: N, 128, 8, 8, 8
#         # x = self.max_4(skip_4)

#         ########## FIFTH BLOCK ##########
#         # Input: N, 128, 8, 8, 8 | output: N, 256, 8, 8, 8
#         # x = self.conv_9(x)
#         # self.print_layer(x, m="conv 9")

#         # x = self.conv_10(x)
#         # self.print_layer(x)

#         self.skip_connections = (skip_1, skip_2, skip_3)
#         return x


class EncoderPath(nn.Module):
    """
    Encoder path of the Unet
    """

    def __init__(self, *args, **kwargs) -> None:
        super(EncoderPath, self).__init__(*args, **kwargs)
        self.conv_1 = ConvolutionalBlock(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            strides=1,
            padding="same",
        )

        self.conv_2 = ConvolutionalBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.drop_3 = nn.Dropout(0.2)

        # ConvolutionalBlock(
        #     in_channels=16,
        #     out_channels=32,
        #     kernel_size=3,
        #     strides=1,
        #     padding="same"
        # )

        self.conv_3 = ConvolutionalBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.drop_4 = nn.Dropout(0.2)

        # ConvolutionalBlock(
        #     in_channels=32,
        #     out_channels=32,
        #     kernel_size=3,
        #     strides=2,
        #     padding=1
        # )

        self.conv_4 = ConvolutionalBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            strides=2,
            padding=1,
        )

        self.conv_5 = ConvolutionalBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            strides=1,
            padding="same",
        )

        self.conv_6 = ConvolutionalBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            strides=2,
            padding=1,
        )
        self.drop_6 = nn.Dropout(0.2)

        # ConvolutionalBlock(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     strides=2,
        #     padding=1
        # )

        self.skip_connections = []
        # self.print_layer = PrintLayer()

        # Initializing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def get_skip_connections(self):
        return self.skip_connections

    def forward(self, x):
        # Input: N, C, D, H, W | output: N, out_channels.conv1, D, H, W

        # Input: N, 1, 64, 64, 64 | output: N, 16, 64, 64, 64
        skip_0 = self.conv_1(x)
        # Input: N, 16, 64, 64, 64 | output: N, 16, 32, 32, 32
        x = self.conv_2(skip_0)
        # self.print_layer(x)

        # Input: N, 16, 32, 32, 32 | output: N, 32, 32, 32, 32
        skip_1 = self.conv_3(x)
        # Input: N, 32, 32, 32, 32 | output: N, 32, 16, 16, 16
        x = self.conv_4(skip_1)
        # self.print_layer(x)

        # Input: N, 32, 16, 16, 16 | output: N, 64, 16, 16, 16
        skip_2 = self.conv_5(x)
        # Input: N, 64, 16, 16, 16 | output: N, 64, 8, 8, 8
        x = self.conv_6(skip_2)
        # self.print_layer(x)

        self.skip_connections = (skip_0, skip_1, skip_2)

        return x

# class DecoderBlock(nn.Module):
#     def __init__(self) -> None:
#         super(DecoderBlock, self).__init__()

#         self.conv_trans_1 = DecoderUpsampleBlock(
#             in_channels=128,
#             out_channels=64,
#             norm_rate=1e-4,
#             kernel_size=3,
#             strides=1
#         )

#         self.conv_trans_2 = DecoderUpsampleBlock(
#             in_channels=128,
#             out_channels=32,
#             norm_rate=1e-4,
#             kernel_size=3,
#             strides=1
#         )

        
#         self.conv_trans_3 = DecoderUpsampleBlock(
#             in_channels=64,
#             out_channels=16,
#             norm_rate=1e-4,
#             kernel_size=3,
#             strides=1
#         )

        
#         # self.conv_trans_4 = nn.ConvTranspose3d(
#         #     in_channels=32,
#         #     out_channels=8,
#         #     kernel_size=2,
#         #     stride=2,
#         #     padding=0,
#         # )
#         self.conv_seg_head = ConvolutionalBlock(
#             in_channels=32,
#             out_channels=16,
#             kernel_size=3,
#             strides=1,
#             padding="same",
#         )
        
#         self.segmentation_head = SegmentationHead(
#             in_channels=16,
#             out_channels=1,  # N classes
#             kernel_size=1,
#             strides=1,
#         )
#         # self.print_layer = PrintLayer()

#     def forward(self, x, skips):
#         # print("Decoder input ", x.shape, "Skip cons: ", len(skips))
#         # print([p.shape for p in skips])

#         ######## FIRST BLOCK ########
#         x = self.conv_trans_1(x)
#         # self.print_layer(x, m="conv trans 1")
#         # print(x.shape, skips[-1].shape)
#         x = torch.cat([ x, skips[-1] ], dim=1)
#         # self.print_layer(x, m="Skip - upconv cat 1")
        
#         ######## SECOND BLOCK ########
#         x = self.conv_trans_2(x)
#         x = torch.cat([ x, skips[-2] ], dim=1)
#         # self.print_layer(x, m="conv trans 2")

#         ######## THIRD BLOCK ########
#         x = self.conv_trans_3(x)
#         # self.print_layer(x, m="conv trans 3")

#         x = torch.cat([ x, skips[-3] ], dim=1)
#         # self.print_layer(x, m="Skip - upconv cat 3")
    
#         # End block
#         x = self.conv_seg_head(x)
#         x = self.segmentation_head(x)
#         # self.print_layer(x, m="Seg head")
#         return x

class DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super(DecoderBlock, self).__init__()

        # Input: N, 64, 8, 8, 8 | output: N, 64, 16, 16, 16
        self.up_conv_1 = DecoderUpsampleBlock(
            in_channels=512,
            out_channels=256,
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        # Input: N, 64, 16, 16, 16 | output: N, 32, 16, 16, 16
        # self.conv_1 = ConvolutionalBlock(
        #     in_channels=512,
        #     out_channels=256,
        #     kernel_size=3,
        #     strides=1,
        #     padding="same",
        # )

        # Input: N, 32, 16, 16, 16 | output: N, 32, 32, 32, 32
        self.up_conv_2 = DecoderUpsampleBlock(
            in_channels=512,
            out_channels=128,
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        # Input: N, 32, 32, 32, 32 | output: N, 16, 32, 32, 32
        # self.conv_2 = ConvolutionalBlock(
        #     in_channels=256,
        #     out_channels=128,
        #     kernel_size=3,
        #     strides=1,
        #     padding="same",
        # )

        # Input: N, 16, 32, 32, 32 | output: N, 16, 64, 64, 64
        self.up_conv_3 = DecoderUpsampleBlock(
            in_channels=256,
            out_channels=64,
            norm_rate=1e-4,
            kernel_size=3,
            strides=1,
        )

        # Input: N, 16, 64, 64, 64 | output: N, 8, 64, 64, 64
        self.conv_3 = ConvolutionalBlock(
            in_channels=128,
            out_channels=32,
            kernel_size=3,
            strides=1,
            padding="same",
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=1,  # N classes
            kernel_size=1,
            strides=1,
        )
        # self.print_layer = PrintLayer()

        # Initializing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, skips):
        # Input: N, 64, 8, 8, 8 | output: N, 64, 16, 16, 16
        x = self.up_conv_1(x)
        # Input: N, 64, 16, 16, 16 | output: N, 64, 16, 16, 16

        x = torch.cat([ x, skips[-1] ], dim=1)
        
        # x = torch.add(x, skips[-1])
        # # Input: N, 64, 16, 16, 16 | output: N, 32, 16, 16, 16
        # x = self.conv_1(x)
        # # self.print_layer(x)
        
        # # Input: N, 32, 16, 16, 16 | output: N, 32, 32, 32, 32
        x = self.up_conv_2(x)
        x = torch.cat([ x, skips[-2] ], dim=1)

        # # Input: N, 32, 32, 32, 32 | output: N, 32, 32, 32, 32
        # x = torch.add(x, skips[-2])
        # # Input: N, 32, 32, 32, 32 | output: N, 16, 32, 32, 32
        # x = self.conv_2(x)

        # # self.print_layer(x)

        # # Input: N, 16, 32, 32, 32 | output: N, 16, 64, 64, 64
        x = self.up_conv_3(x)
        # # Input: N, 16, 64, 64, 64 | output: N, 16, 64, 64, 64
        x = torch.cat([ x, skips[-3] ], dim=1)
        x = self.conv_3(x)

        # x = torch.add(x, skips[-3])
        # # Input: N, 16, 64, 64, 64 | output: N, 8, 64, 64, 64
        # # self.print_layer(x)

        # # End block
        x = self.segmentation_head(x)
        # self.print_layer(x)

        return x


def dice_coefficient(y_true, y_pred):
    # Flatten the tensors
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)

    dice = (2.0 * intersection + 1e-5) / (
        union + 1e-5
    )  # Add a small epsilon to avoid division by zero

    return dice


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


class Neuratt(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.encoder_path = EncoderPath()
        # self.decoder_path = DecoderBlock()
        self.model = mnets.SwinUNETR(
            img_size=(1024, 1024),
            in_channels=1,
            out_channels=1,
            feature_size=12,
            spatial_dims=2,
        )

        # mnets.ViTAutoEnc(
        #     in_channels=1,
        #     img_size=(1024, 1024),
        #     patch_size=64,
        #     out_channels=1,
        #     deconv_chns=8,
        #     hidden_size=768,
        #     mlp_dim=1024,
        #     num_layers=4,
        #     num_heads=8,
        #     proj_type='conv',
        #     dropout_rate=0.0,
        #     spatial_dims=2,
        #     qkv_bias=False,
        #     save_attn=False
        # )
        

        self.val_images = []
        self.n_val_images = 3

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        # FocalLoss(
        #     mode="binary",
        #     alpha=0.25,
        #     gamma=2
        # )
        # DiceFocalLoss(alpha=0.25, gamma=2, smooth=1e-6)
        #nn.BCEWithLogitsLoss()
        #nn.functional.binary_cross_entropy()
        # DiceFocalLoss(alpha=0.25, gamma=2, smooth=1e-6)
        # FocalLoss(mode="binary", alpha=0.25, gamma=2)
        #
        #(
        #    nn.BCEWithLogitsLoss()
        #)  # DiceLoss(BINARY_MODE)#, from_logits=True)
        ##nn.functional.binary_cross_entropy()
        # DiceLoss(BINARY_MODE, from_logits=True)

        # Metrics
        # self.accuracy_metric = Accuracy(task="binary")
        self.dice_score_metric = F1Score(task="binary")
        self.jaccard_index_metric = JaccardIndex(task="binary")

        self.save_hyperparameters()

    def forward(self, x):
        """method used for inference input -> output"""

        # encoder_result = self.encoder_path(x)
        # skip_conns = self.encoder_path.get_skip_connections()
        # decoder_result = self.decoder_path(encoder_result, skip_conns)

        decoder_result = self.model(x)
        
        return decoder_result
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        decoder_result = self(x)

        # Loss function
        # loss = nn.functional.binary_cross_entropy(
        #     input=decoder_result, target=y
        # )
        # print("Decoder: ", y.shape, decoder_result.shape)

        prob_mask = decoder_result.sigmoid()
        
        loss = self.loss_fn(prob_mask, y)
        if torch.isnan(loss):
            print("Loss is NaN!")
            np.save("/results/problem_data.npy", x.detach().cpu().numpy())
            np.save("/results/problem_mask.npy", y.detach().cpu().numpy())
            np.save("/results/problem_pred.npy", prob_mask.detach().cpu().numpy())
            np.save("/results/problem_bef_sigmoid.npy", decoder_result.detach().cpu().numpy())
            
            exit()
        # else:
        #     print("Loss function")
        #     np.save("/results/no_problem_data.npy", x.detach().cpu().numpy())
        #     np.save("/results/no_problem_mask.npy", y.detach().cpu().numpy())
        #     np.save("/results/no_problem_pred.npy", prob_mask.detach().cpu().numpy())
        #     np.save("/results/no_problem_bef_sigmoid.npy", decoder_result.detach().cpu().numpy())
            
        #     exit()
        
        pred_mask = (prob_mask > 0.5).float()

        self.log("train/loss", loss.item())

        # Metrics
        # self.accuracy_metric(pred_mask, y)
        self.dice_score_metric(pred_mask, y)
        self.jaccard_index_metric(pred_mask, y)

        # self.log(
        #     name="train/metrics/accuracy",
        #     value=self.accuracy_metric,
        #     on_epoch=True,
        # )

        self.log(
            name="train/metrics/dice_score",
            value=self.dice_score_metric,
            on_epoch=True,
        )

        self.log(
            name="train/metrics/jaccard_index",
            value=self.jaccard_index_metric,
            on_epoch=True,
        )

        self.log_dict(
            {
                "loss": loss.item(),
                "dice_score": self.dice_score_metric,
                "jaccard_score": self.jaccard_index_metric,
                # "accuracy": self.accuracy_metric,
            },
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoder_result = self(x)
        # loss = nn.functional.binary_cross_entropy(
        #     input=decoder_result, target=y
        # )
        # print("Decoder: ", y.shape, decoder_result.shape)
        loss = self.loss_fn(decoder_result, y)
        self.log("val_loss", loss.item())

        prob_mask = decoder_result.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        self.dice_score_metric(pred_mask, y)
        self.jaccard_index_metric(pred_mask, y)

        # self.log(
        #     name="train/metrics/accuracy",
        #     value=self.accuracy_metric,
        #     on_epoch=True,
        # )

        self.log(
            name="validation/metrics/dice_score",
            value=self.dice_score_metric,
            on_epoch=True,
        )

        self.log(
            name="validation/metrics/jaccard_index",
            value=self.jaccard_index_metric,
            on_epoch=True,
        )

        if len(self.val_images) < self.n_val_images:
            
            self.val_images.append(
                {
                    "images": x[-1].detach().cpu().numpy(),
                    "preds": pred_mask[-1].detach().cpu().numpy(),
                    "labels": y[-1].detach().cpu().numpy()
                }
            )
        
        return loss

    def on_validation_epoch_end(self):
        images = []
        for output in self.val_images:
            for img, pred, label in zip(output["images"], output["preds"], output["labels"]):
                # Log input image + true mask + predicted mask
                images.append(
                    wandb.Image(
                        img,
                        masks={
                            "ground_truth": {"mask_data": label, "class_labels": {0: "Background", 1: "Tissue"}},
                            "prediction": {"mask_data": pred, "class_labels": {0: "Background", 1: "Tissue"}}
                        },
                        caption="Predictions"
                    )
                )

        wandb.log({"Predictions": images})
        self.val_images.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        decoder_result = self(x)
        loss = self.loss_fn(decoder_result, y)

        prob_mask = decoder_result.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        dice = self.dice_score_metric(prob_mask, y)
        jacc = self.jaccard_index_metric(pred_mask, y)
        
        metrics = {
            'loss': loss.item(),
            'dice': dice.item(),
            'jacc': jacc.item()
        }
        
        return (x, pred_mask, metrics)

    def predict(self, batch, threshold=0.5, dataloader_idx=0):

        decoder_result = self(batch)

        prob_mask = decoder_result.sigmoid()
        pred_mask = (prob_mask > threshold).float()
        
        return pred_mask, prob_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.000015)
        return optimizer


class Neuratt_Test(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Neuratt_Test, self).__init__(*args, **kwargs)
        self.encoder_path = EncoderPath()
        self.decoder_path = DecoderBlock()

    def forward(self, x):
        x = self.encoder_path(x)
        skip_conns = self.encoder_path.get_skip_connections()
        x = self.decoder_path(x, skip_conns)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(
        p.numel() for p in model.parameters()
    )


def check_model():
    # logger = create_logger("test_network.log")

    print(torch.__version__)
    image_size = 64
    channels = 1
    x = torch.Tensor(1, channels, image_size, image_size, image_size)

    print(f"Input shape: {x.shape}")
    model = Neuratt_Test()

    out = model(x)
    trainable_params, total_params = count_params(model)
    
    print(model)

    trainable_params_encoder, total_params_encoder = count_params(model.encoder_path)
    trainable_params_decoder, total_params_decoder = count_params(model.decoder_path)
    
    print("Trainable params: ", trainable_params, " total params: ", total_params, " output image: ", out.shape)
    print("Trainable encoder params: ", trainable_params_encoder, " total params: ", total_params_encoder, " output image: ", out.shape)
    print("Trainable decoder params: ", trainable_params_decoder, " total params: ", total_params_decoder, " output image: ", out.shape)
    
    # make_dot(out).render("neuratt", format="png")

if __name__ == "__main__":
    check_model()
