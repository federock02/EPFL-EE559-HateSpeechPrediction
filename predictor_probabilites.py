import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import yaml
from tqdm import tqdm
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from checkpoint_utils import save_checkpoint, load_checkpoint

def parse_args():
    # when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    return args.dataset_path, args.results_path

DATADIR, RESULTS_DIR = parse_args()
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
# directory to store model checkpoints to
if not os.path.exists(Path(RESULTS_DIR) / 'checkpoints'):
    os.mkdir(Path(RESULTS_DIR) / 'checkpoints')
# dataset directory
if not os.path.exists(Path(DATADIR) / 'data'):
    raise Exception(f'Dataset not found. Please upload a dataset first. '
                    f'It should be stored in the {Path(DATADIR) / 'data'} directory')

config_file = Path(DATADIR) / 'data' / 'cfg.yaml'
data = Path(DATADIR) / 'data' / 'data_2.csv'

class Predictor:
    def __init__(self, cfg_file, data_file):
        # configuration file
        self._read_config(cfg_file)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # freeze all parameters except the pooler
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.pooler.parameters():
            param.requires_grad = True

        # prediction model
        self.model = self.PredictionModel(
            text_encoder=self.text_encoder,
            hidden_size=self.hidden_size,
            output_dim=1
        )

        # dataset
        self.train_loader, self.val_loader = self._load_dataset(data_file)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # optimizer
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        # criterion
        self.criterion = nn.BCELoss()
        # self.criterion = nn.BCEWithLogitsLoss()

        # metrics
        self.metrics = {
            "f1": self._f1,
            "accuracy": self._acc
        }
    
    def _read_config(self, config_path):
        # opening the config file and extracting the parameters
        with open("cfg.yaml", "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # text
        self.max_len = config["text"].get("max_len", 512)
        self.hidden_size = config["text"].get("hidden_size", 512)

        #training
        self.batch_size = config["training"]["batch_size"]
        self.lr = config["training"]["lr"]
        self.epochs = config["training"]["epochs"]
        self.val_split_ratio = config["training"].get("val_split_ratio", 0.2)
    
    def _tokenize_text(self, text):
        # Tokenize the text
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']
    
    def _collate_fn(self, batch):
        # Collate function to create batches
        input_ids = torch.cat([item[0] for item in batch], dim=0)
        attention_mask = torch.cat([item[1] for item in batch], dim=0)
        # Ensure labels are converted to float and have the correct shape for BCELoss
        labels = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1)
        return input_ids, attention_mask, labels
    
    def _load_dataset(self, data_file):
        df = pd.read_csv(data_file)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        text_data = df["text"]
        labels = df["label"].astype("float32")
        print(text_data[0], labels[0])

        # Perform train/validation split
        # stratify=labels ensures that the proportion of labels is the same in train and val sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            text_data, labels, test_size=self.val_split_ratio, random_state=42, stratify=labels
        )

        train_texts  = train_texts.tolist()
        for text in train_texts:
            cut = np.random.randint(0, len(text.split()))
            text = " ".join(text.split()[:cut]) # take the prefix
        val_texts    = val_texts.tolist()
        for text in val_texts:
            cut = np.random.randint(0, len(text.split()))
            text = " ".join(text.split()[:cut]) # take the prefix
        train_labels = train_labels.tolist()
        val_labels   = val_labels.tolist()

        print(f"Train set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")

        # Instantiate dataset and dataloaders
        train_dataset = self.TextDataset(train_texts, train_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn)

        val_dataset = self.TextDataset(val_texts, val_labels, text_transform=self._tokenize_text, max_len=self.max_len, train=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)

        return train_loader, val_loader
    
    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        for input_ids, attention_mask, labels in tqdm(self.train_loader, desc="Training"):
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, prob, classes = self.model(input_ids, attention_mask)
            # Compute loss
            loss = self.criterion(prob, labels) # using probabilities (BCELoss)
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](classes.cpu(), labels.cpu())
        
        epoch_loss / len(self.train_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.train_loader)

        print('train Loss: {:.4f}, '.format(epoch_loss), ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

        return epoch_loss, epoch_metrics
    
    def _evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = dict(zip(self.metrics.keys(), torch.zeros(len(self.metrics))))
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(self.val_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                # Forward pass
                logits, _, classes = self.model(input_ids, attention_mask)
                # Compute loss
                #loss = self.criterion(classes, labels)
                loss = self.criterion(logits, labels)
                # Accumulate loss
                epoch_loss += loss.item()
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += self.metrics[k](classes.cpu(), labels.cpu())
                
                # Compute accuracy
                # _, predicted = torch.max(classes, 1)
                correct += (classes == labels).sum().item()
                total += labels.size(0)

        epoch_loss /= len(self.val_loader)
        for k in epoch_metrics.keys():
            epoch_metrics[k] /= len(self.val_loader)

        return epoch_loss, epoch_metrics
    
    def _f1(self, preds, target):
        return f1_score(target, preds, average='macro')

    def _acc(self, preds, target):
        return accuracy_score(target, preds)

    def _plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
        fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

        ax[0].plot(train_loss, c='blue', label='train')
        ax[0].plot(test_loss, c='orange', label='validation')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].legend()

        for i in range(len(metrics_names)):
            ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
            ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
            ax[i + 1].set_title(metrics_names[i])
            ax[i + 1].set_xlabel('epoch')
            ax[i + 1].legend()

        plt.savefig(Path(RESULTS_DIR) / "training loss and metrics.jpg")
        # close the figure to free up memory
        plt.close(fig)
    
    def _update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
        for i in range(len(metrics_names)):
            curr_metric_name = metrics_names[i]
            metrics_log[i].append(new_metrics_dict[curr_metric_name])
        return metrics_log
    
    def train_model(self):
        best_accuracy = 0
        train_loss_log,  val_loss_log = [], []
        metrics_names = list(self.metrics.keys())
        train_metrics_log = [[] for i in range(len(self.metrics))]
        val_metrics_log = [[] for i in range(len(self.metrics))]
        store_checkpoint_for_every_epoch = False
        for epoch in range(self.epochs):
            train_loss, train_metrics = self._train_epoch()
            train_loss_log.append(train_loss)
            train_metrics_log = self._update_metrics_log(metrics_names, train_metrics_log, train_metrics)

            val_loss, val_metrics = self._evaluate_epoch()
            val_loss_log.append(val_loss)
            val_metrics_log = self._update_metrics_log(metrics_names, val_metrics_log, val_metrics)
            accuracy = val_metrics["accuracy"]
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            self._plot_training(train_loss_log, val_loss_log, metrics_names, train_metrics_log, val_metrics_log)

            # Save the model if the accuracy is improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Model saved!")
            
            save_checkpoint(self.model, self.optimizer, epoch, loss=train_loss, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth", store_checkpoint_for_every_epoch=store_checkpoint_for_every_epoch)

    def load_model(self, model_path):
        # Load the model state dict
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("Model loaded!")

    def predict(self, text):
        # Tokenize the text
        input_ids, attention_mask = self._tokenize_text(text)

        # Move to device
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            _, prob, class_value = self.model(input_ids, attention_mask)
        
        return prob.cpu().numpy(), class_value.cpu().numpy()

    class PredictionModel(nn.Module):
        def __init__(self, text_encoder, hidden_size, output_dim):
            super().__init__()
            self.text_encoder = text_encoder

            self.classifier = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(hidden_size, output_dim)
            )
            self.classifier.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        def forward(self, input_ids, attention_mask):
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_features = out.pooler_output
            output = self.classifier(text_features)
            prob = torch.sigmoid(output)
            class_value = torch.round(prob)
            return output, prob, class_value
        
    class TextDataset(Dataset):
        def __init__(self, text_data, labels, train, text_transform=None, max_len=512):
            self.text_data = text_data
            self.labels = labels
            self.text_transform = text_transform
            self.max_len = max_len
            self.train = train

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # Tokenize and preprocess text
            text = self.text_data[idx]
            
            length = len(text.split())
            divider = np.random.randint(0, length)
            text = " ".join(text.split()[:divider]) # take the prefix

            if self.text_transform:
                input_ids, attention_mask = self.text_transform(text)

            label = self.labels[idx]
            return input_ids, attention_mask, label
        

if __name__ == "__main__":
    predictor = Predictor(config_file, data)
    predictor.train_model()
    predictor.load_model("best_model.pth")
    
    # Example prediction
    text = "This is a sample text for prediction."
    prob, class_value = predictor.predict(text)
    print(f"Probability: {prob}, Class Value: {class_value}")