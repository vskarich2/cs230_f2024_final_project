import torch
import numpy as np
import pandas as pd
from model import VisionGPT2Model
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gc



class Trainer:
    def __init__(self, model_config, train_config, dls):

        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scaler = GradScaler()
        # More infor on GradScalar: https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5

        self.train_dl, self.val_dl = dls

        total_steps = len(self.train_dl)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )


        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity', 'val_loss', 'val_perplexity']] = None

        self.standardize = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

    def save_model(self):
        self.train_config.model_path.mkdir(exist_ok=True)
        model = self.model.state_dict()
        torch.save(model, self.train_config.model_path / 'model.pt')

    def load_best_model(self):
        model = torch.load(self.train_config.model_path / 'model.pt')
        self.model.load_state_dict(model)

    def train_single_epoch(self, epoch):

        # More info on tqdm here: https://github.com/tqdm/tqdm
        progress = tqdm(self.train_dl, total=len(self.train_dl))

        running_loss = 0.

        for image, input_ids, labels in progress:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)

                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer)

                self.scaler.update()

                self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item()

                progress.set_description(f'train loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        self.metrics.loc[epoch, ['train_loss', 'train_perplexity']] = (train_loss, train_pxp)

    @torch.no_grad()
    def validate_single_epoch(self, epoch):

        progress = tqdm(self.val_dl, total=len(self.val_dl))

        running_loss = 0.

        for image, input_ids, labels in progress:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()

                progress.set_description(f'valid loss: {loss.item():.3f}')


        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        return val_pxp

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):

        best_pxp = 1e9
        best_epoch = -1
        progress = tqdm(range(self.train_config.epochs))

        for epoch in progress:

            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                #print('unfreezing all GPT2 layers')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()

            progress.set_description('training')

            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            progress.set_description('validating')

            pxp = self.validate_one_epoch(epoch)
            self.clean()

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):

        self.model.eval()

        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.standarize(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)

        return caption