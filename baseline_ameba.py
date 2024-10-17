# Imports
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from captum.attr import Saliency
from tqdm import tqdm
import matplotlib
import csv
import pickle
from datetime import datetime
matplotlib.use('Agg')


# PyTorch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,Subset, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# Sklearn Imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_curve,auc, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    balanced_accuracy_score, 
    accuracy_score,
    confusion_matrix, 
    classification_report
)

import scipy.stats as st

# Class: VGG16
class VGG16(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(VGG16, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes

        # Init modules
        # Backbone to extract features
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.vgg16(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        if nr_classes == 2:
            self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=1)
        else:
            self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=nr_classes)

        return
    
    # Method: forward
    def forward(self, inputs):

        # Compute Backbone features
        features = self.vgg16(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # FC1-Layer
        outputs = self.fc1(features)

        return outputs



# Class: DenseNet121
class DenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(DenseNet121, self).__init__()
        
        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes

        # Init modules
        # Backbone to extract features
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.AvgPool2d= nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.batch_norm = nn.BatchNorm2d(self.densenet121[-1].num_features)
        self.dropout = nn.Dropout(p=0.3)    
        
        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.densenet121(_in_features)
        _in_features = self.AvgPool2d(_in_features)
        _in_features = _in_features.size(1) * _in_features.size(2) * _in_features.size(3)
        
        # Create FC1 Layer for classification
        if nr_classes == 2:
            self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=1)
        else:
            self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=nr_classes)
        
        return
    def freezelayers(self):
        for param in self.densenet121.parameters():
            param.requires_grad = False
    def unfreeze_all_layers(self):
        for param in self.densenet121.parameters():
            param.requires_grad = True

    # Method: forward
    def forward(self, inputs):
        features = self.densenet121(inputs)
        features = self.AvgPool2d(features)
        features = self.batch_norm(features)
        features= self.dropout(features)
        
        # Step 3: Reshaping
        features = features.view(features.size(0), -1)

        # FC1-Layer
        outputs = self.fc1(features)

        return outputs

def aggregate_classifications(group):
    classification_columns = ['Cultura para FUNGOS Res', 'Cultura para ACANTHAMOEBA Res', 'Cultura para BACTÉRIA Res']
    for col in classification_columns:
        if 'Positiva' in group[col].values:
            group[col] = 'Positiva'
        else:
            group[col] = 'Negativa'
    return group.iloc[0]


# Class: CorneaUnifespDataset
class CorneaUnifespDataset(Dataset):

    # Method: __init__
    def __init__(
            self, 
            csv_file_path='/nas-ctm01/datasets/private/MEDICAL/UNIFESP/cornea-unifesp-db/data/metadata_proc_flipped.csv', 
            image_directory='/nas-ctm01/datasets/private/MEDICAL/UNIFESP/cornea-unifesp-db/data/images-resized',
            image_directory_flipped='/nas-ctm01/datasets/private/MEDICAL/UNIFESP/cornea-unifesp-db/data/images-resized-flipped',
            #train_size=0.8,
            #val_size=0.1,
            #test_size=0.1,
            seed=42,
            split='train',
            transform=None, 
            classification=None):
        
        #assert int(train_size+val_size+test_size) == 1
        #assert split in ('train', 'validation', 'test')

        # Assign class variables
        self.csv_file_path = csv_file_path
        self.image_directory = image_directory
        self.image_directory_flipped = image_directory_flipped
        #self.seed = seed
        #self.train_size, self.val_size, self.test_size = train_size, val_size, test_size
        

        # Open CSV file
        filtered_patients_info = self.create_database(csv_file_path,image_directory,image_directory_flipped)
        filtered_patients_info['coleta'] = filtered_patients_info[['Cultura para FUNGOS Res', 'Cultura para ACANTHAMOEBA Res', 'Cultura para BACTÉRIA Res']].apply(
            lambda row: 1 if any(val == 'Positiva' for val in row) else 0, axis=1
        )
        filtered_patients_info_coleta = filtered_patients_info[filtered_patients_info['coleta'] == 1]
        filtered_patients_info_coleta = filtered_patients_info_coleta.groupby('file_names').apply(aggregate_classifications).reset_index(drop=True)
        filtered_patients_info_coleta.loc[:, 'Cultura para FUNGOS Res'] = np.where(filtered_patients_info_coleta['Cultura para FUNGOS Res'] == 'Positiva', 1, 0)
        filtered_patients_info_coleta.loc[:, 'Cultura para BACTÉRIA Res'] = np.where(filtered_patients_info_coleta['Cultura para BACTÉRIA Res'] == 'Positiva', 1, 0)
        filtered_patients_info_coleta.loc[:, 'Cultura para ACANTHAMOEBA Res'] = np.where(filtered_patients_info_coleta['Cultura para ACANTHAMOEBA Res'] == 'Positiva', 1, 0)

            
        if classification=='bacteria':
            X = filtered_patients_info_coleta.drop(['Cultura para BACTÉRIA Res'], axis=1)
            y = filtered_patients_info_coleta['Cultura para BACTÉRIA Res'].astype(int)
            filtered_patients_info_coleta['Sexo'] = filtered_patients_info_coleta['Sexo'].map({'Masculino': 0, 'Feminino': 1})
            y_sex = filtered_patients_info_coleta['Sexo']
            bins = [0, 18, 40, 65, 100]
            labels = [0, 1, 2, 3]
            filtered_patients_info_coleta['idade'] = pd.cut(filtered_patients_info_coleta['Idade'], bins=bins, labels=labels, right=False)
            filtered_patients_info_coleta['idade'] = filtered_patients_info_coleta['idade'].astype(int)
            y_age = filtered_patients_info_coleta['idade']
           
        elif classification=='fungi':
            X = filtered_patients_info_coleta.drop(['Cultura para FUNGOS Res'], axis=1)
            y = filtered_patients_info_coleta['Cultura para FUNGOS Res'].astype(int)
            filtered_patients_info_coleta['Sexo'] = filtered_patients_info_coleta['Sexo'].map({'Masculino': 0, 'Feminino': 1})
            y_sex = filtered_patients_info_coleta['Sexo']
            bins = [0, 18, 40, 65, 100]
            labels = [0, 1, 2, 3]
            filtered_patients_info_coleta['idade'] = pd.cut(filtered_patients_info_coleta['Idade'], bins=bins, labels=labels, right=False)
            filtered_patients_info_coleta['idade'] = filtered_patients_info_coleta['idade'].astype(int)
            y_age = filtered_patients_info_coleta['idade']
        elif classification=='ameba':
            X = filtered_patients_info_coleta.drop(['Cultura para ACANTHAMOEBA Res'], axis=1)
            y = filtered_patients_info_coleta['Cultura para ACANTHAMOEBA Res'].astype(int)
            filtered_patients_info_coleta['Sexo'] = filtered_patients_info_coleta['Sexo'].map({'Masculino': 0, 'Feminino': 1})
            y_sex = filtered_patients_info_coleta['Sexo']
            bins = [0, 18, 40, 65, 100]
            labels = [0, 1, 2, 3]
            filtered_patients_info_coleta['idade'] = pd.cut(filtered_patients_info_coleta['Idade'], bins=bins, labels=labels, right=False)
            filtered_patients_info_coleta['idade'] = filtered_patients_info_coleta['idade'].astype(int)
            y_age = filtered_patients_info_coleta['idade']
        """
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size, 
            random_state=self.seed, 
            stratify=y
        )

        # Split into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            train_size=(self.train_size/(self.train_size+self.val_size)), 
            random_state=42,
            stratify=y_train_val
        )

        # print(len(X_train)/len(X), self.train_size)
        # print(len(X_val)/len(X), self.val_size)
        # print(len(X_test)/len(X), self.test_size) 

        # Prepare dataset
        train_image_fnames, val_image_fnames, test_image_fnames, y_train_, y_val_, y_test_ = self.load_and_preprocess_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        # Check the splits
        if split == 'train':
            images = train_image_fnames.copy()
            labels = y_train_.copy()
        elif split == 'validation':
            images = val_image_fnames.copy()
            labels = y_val_.copy()
        else:
            images = test_image_fnames.copy()
            labels = y_test_.copy()
        
        self.images = images
        self.labels = labels
        self.transform = transform
        """
        image_fnames, y_train_,y_age,y_sex= self.load_and_preprocess_data(
           X , y, y_age, y_sex
        )

        images = image_fnames.copy()
        labels = y_train_.copy()
        y_sex=y_sex.copy()
        y_age=y_age.copy()

        self.images = images
        self.labels = labels
        self.sex= y_sex
        self.age=y_age
        self.transform = transform
        return

    # Method: create_database
    def create_database(self, csv_file_path,image_folder_path,image_flipped_folder_path):
        patients_info = pd.read_csv(csv_file_path)
        image_filenames = set(os.listdir(image_folder_path))
        image_filenames_flipped = set(os.listdir(image_flipped_folder_path))
        patients_info = patients_info[patients_info['file_names'].isin(image_filenames)| patients_info['file_names'].isin(image_filenames_flipped)]
        return patients_info

    """
    # Method: load_and_preprocess_data
    def load_and_preprocess_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
    
        # Get file lists
        train_image_fnames = X_train['file_names'].tolist()
        val_image_fnames = X_val['file_names'].tolist()
        test_image_fnames = X_test['file_names'].tolist()

        y_train_ = y_train.values
        y_val_ = y_val.values
        y_test_ = y_test.values
        
        return train_image_fnames, val_image_fnames, test_image_fnames, y_train_, y_val_, y_test_
    """
    def load_and_preprocess_data(self, X, y,y_age,y_sex):
    
        # Get file lists
        train_image_fnames = X['file_names'].tolist()
        y_age=y_age.values
        y_sex=y_sex.values
        y_train_ = y.values
       
        return train_image_fnames, y_train_,y_age, y_sex

    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Get image fname and label
        image_fname = self.images[idx]
        label = self.labels[idx]
        age=self.age[idx]
        sex=self.sex[idx]
        # Load image
        image_fpath_original = os.path.join(self.image_directory, image_fname)
        image_fpath_flipped = os.path.join(self.image_directory_flipped, image_fname)
        if os.path.isfile(image_fpath_original):
            image_fpath = image_fpath_original
        elif os.path.isfile(image_fpath_flipped):
            image_fpath = image_fpath_flipped
        image_pil = Image.open(image_fpath).convert('RGB')     
        
        if self.transform:
            image_tensor = self.transform(image_pil)

        return image_tensor, torch.tensor(label, dtype=torch.long),torch.tensor(age, dtype=torch.long),torch.tensor(sex, dtype=torch.long)



# Function: plot_roc_curve_per_class
def plot_roc_curve_per_class(y_true, y_probs, num_classes,fold):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, color='lightgreen', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('(ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir,'Results_10folds','infections_baseline', f'{fold}')
    plot_name = f'ROC_curve_{model_name}_{analysis}_{epochs}.png'
    os.makedirs(save_dir, exist_ok=True)
    full_plot_path = os.path.join(save_dir, plot_name)
    plt.savefig(full_plot_path)
    plt.close()

def compute_metrics_with_indices(indices, labels_test, preds_test,task_probs_test,subset_name):
    if subset_name =='General':
        task_labels_test = labels_test
        task_preds_test = preds_test
        task_probs_test=task_probs_test
    else:
        task_labels_test = labels_test[indices]
        task_preds_test = preds_test[indices]
        task_probs_test=task_probs_test[indices]
    try:
        accuracy_test = accuracy_score(task_labels_test, task_preds_test)
    except:
        accuracy_test = np.nan

    try:
        auc_test = roc_auc_score(task_labels_test, task_probs_test)
    except:
        auc_test = np.nan

    try:
        precision_test = precision_score(task_labels_test, task_preds_test)
    except:
        precision_test = np.nan

    try:
        recall_test = recall_score(task_labels_test, task_preds_test)
    except:
        recall_test = np.nan

    try:
        f1_test = f1_score(task_labels_test, task_preds_test)
    except:
        f1_test = np.nan

    try:
        BA_test = balanced_accuracy_score(task_labels_test, task_preds_test)
    except:
        BA_test = np.nan

    
    return accuracy_test,auc_test,precision_test,recall_test, f1_test, BA_test

# Function: Load model
def load_model(model_name, nr_classes):
    if model_name == 'vgg16':
        model = VGG16(
            channels=3, 
            height=224, 
            width=224, 
            nr_classes=nr_classes
        )
    elif model_name == 'densenet121':
        model = DenseNet121(
            channels=3, 
            height=224, 
            width=224, 
            nr_classes=nr_classes
        )
    else:
        raise ValueError("Invalid model name")
    return model

def save_model_with_info(model, model_name, analysis,fold):
    
    epochs=args.epochs
    plot_name = f'model_{model_name}_{analysis}_{epochs}_{fold}.pth'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir,'Results_10folds','infections_baseline', f'{fold}')
    os.makedirs(save_dir, exist_ok=True)
    full_plot_path = os.path.join(save_dir, plot_name)
    torch.save(model.state_dict(), full_plot_path)
    return full_plot_path

def load_model_with_info(filepath, channel, height, width):
    checkpoint = torch.load(filepath)
    model = DenseNet121(channels=channel,height=height,width=width, nr_classes=num_classes) 
    model.load_state_dict(checkpoint)
    return model

# Function: Train and Evaluate
def train_and_evaluate(model, train_loader, val_loader,test_loader, weights, optimizer, epochs, model_name, analysis,fold,metrics_per_subset,mean_confusion_matrix):
    
    # Get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(f"Device: {device}")

    # Move model (and other stuff) into device
    model.to(device)

    #Normalize weights
    positive_weight = weights[1]/weights[0]
    weights=torch.tensor([1.0, positive_weight], dtype=torch.float64)
    weights.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(weight=weights[1])
    criterion.to(device)

    # Create lists to track losses
    train_losses = []
    val_losses = []
    best_val_loss= float('inf')
    model.freezelayers()
    freeze = True
   # Go through the number of epochs
    for epoch in range(epochs):
        if freeze == True:
            if epoch+1 >= 10:
                model.unfreeze_all_layers()
                freeze = False
        model.train()
        epoch_train_losses = []
        print("epoch",epoch)

        for images, labels,age,sex in train_loader:
            
            # Get images and labels, and move them into device
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            # Clear gradients
            optimizer.zero_grad()

            # Get logits
            outputs = model(images)
            outputs = outputs.float()

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the error
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
        print("Train loss: ", sum(epoch_train_losses) / len(epoch_train_losses))
        elapsed_time = datetime.now() - start_time
        print(f"Train Time", elapsed_time)
    
        # Validation loop
        model.eval()
        all_labels = list()
        all_probs = list()
        epoch_val_losses = []

        with torch.no_grad():

            # Get data
            for images, labels,age,sex in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.unsqueeze(1).float()

                # Get logits
                outputs = model(images)
                outputs = outputs.float()

                # Compute batch validation loss
                loss = criterion(outputs, labels)
                epoch_val_losses.append(loss.item())

                # Get probabilities
                probs = torch.nn.functional.sigmoid(outputs)

                # Populate lists
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute validation loss
        val_losses.append(sum(epoch_val_losses) / len(epoch_val_losses))
        print("Validation loss: ", sum(epoch_val_losses) / len(epoch_val_losses))
        elapsed_time = datetime.now() - start_time
        print(f"Val Time", elapsed_time)   
        val_loss=sum(epoch_val_losses) / len(epoch_val_losses)

        # Convert lists into NumPy arrays
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.where(all_probs >= 0.5, 1, 0)

        # Calculate metrics
        accuracy_val = accuracy_score(y_true=all_labels, y_pred=all_preds)
        auc_val = roc_auc_score(y_true=all_labels, y_score=all_probs)
        f1_val = f1_score(y_true=all_labels, y_pred=all_preds)
        precision_val = precision_score(y_true=all_labels, y_pred=all_preds)
        recall_val = recall_score(y_true=all_labels, y_pred=all_preds)
        balanced_accuracy_val = balanced_accuracy_score(y_true=all_labels, y_pred=all_preds)
        print(f"Validation Metrics - ACC:{accuracy_val}, AUC: {auc_val}, F1 Score: {f1_val}, Precision: {precision_val}, Recall: {recall_val}, Balanced Accuracy: {balanced_accuracy_val}")
        if val_loss< best_val_loss:
            best_val_loss = val_loss
            best_epochs = epoch
            # Save the current model
            best_model_path= save_model_with_info(model, model_name, analysis,fold)
            print(f'Saved the best model with loss: {best_val_loss:.4f} at epoch {best_epochs}')

    
    # Plot the evolution of losses over epochs1
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_name=f'loss_evolution_{model_name}_{analysis}_{epochs}_{fold}.png'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir,'Results_10folds','infections_baseline', f'{fold}')
    os.makedirs(save_dir, exist_ok=True)
    full_plot_path = os.path.join(save_dir, plot_name)
    plt.savefig(full_plot_path)
    plt.close()
    
    
    all_positive_indices = []
    all_negative_indices = []
    all_positive_images = []
    all_negative_images = []
    best_model = load_model_with_info(best_model_path, 3, 224, 224)
    best_model.to(device)    
    # Test loop
    best_model.eval()
    y_test = list()
    y_proba_test = list()
    saliency = Saliency(best_model)
    num_instances_to_plot = 1
    sex_labels_test = []
    age_labels_test = []
    reverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # Undo normalization
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    ])
    
    with torch.no_grad():
        for images, labels,age,sex in tqdm(test_loader, desc=f"Testing"):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            # Get logits
            outputs = best_model(images)
            images = reverse_transform(images)
            
            images.clamp_(0, 1)
            outputs = outputs.float()
            sex_labels_test.extend(sex.cpu().numpy())
            age_labels_test.extend(age.cpu().numpy())
            # Convert logits into probabilities
            probs = torch.nn.functional.sigmoid(outputs)
            y_test.extend(labels.cpu().numpy())
            y_proba_test.extend(probs.cpu().numpy())
            positive_indices = (labels== 1).nonzero().squeeze(1)
            negative_indices = (labels == 0).nonzero().squeeze(1)
            # Append positive and negative indices and corresponding images to arrays
            all_positive_indices.append(positive_indices.cpu().numpy())
            all_negative_indices.append(negative_indices.cpu().numpy())
            all_positive_images.append(images[positive_indices])
            all_negative_images.append(images[negative_indices])

    # Convert lists into NumPy arrays
    y_test = np.array(y_test)
    y_proba_test = np.array(y_proba_test)
    y_pred_test = np.where(y_proba_test >= 0.5, 1, 0)
    #plot saliency maps
    fig, axs = plt.subplots(2, 2, figsize=(10, 20))
    fig.suptitle(f'Task Infection', fontsize=16)

    try:
        try:
            if len(all_positive_images) > 0 and all_positive_images[0][0].shape != (0, 3, 224, 224):
                for i in range(num_instances_to_plot):
                    try:
                        print(all_positive_images[0][0].shape)
                        selected_pos_img = all_positive_images[0][0][i].cpu().numpy()
                        selected_pos_img = selected_pos_img.transpose((1, 2, 0))
                        selected_pos_img_sal = selected_pos_img.transpose((2, 0, 1))
                        selected_pos_img_sal = np.expand_dims(selected_pos_img_sal, axis=0)
                        selected_pos_img = np.array(selected_pos_img)
                        axs[i, 0].imshow(selected_pos_img)
                        axs[i, 0].set_title(f'Positive Instance {i+1}')
                        axs[i, 0].axis('off')

                        selected_pos_img_sal = torch.tensor(selected_pos_img_sal).to(device)
                        pos_saliency_map = saliency.attribute(selected_pos_img_sal)
                        pos_saliency_map = pos_saliency_map.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
                        axs[i, 1].imshow(pos_saliency_map)
                        axs[i, 1].axis('off')
                    except Exception as e:
                        print(f"Error processing positive instance {i+1}: {e}")
        except Exception as pos_e:
            print(f"Error in processing positive images: {pos_e}")

        try:
            if len(all_negative_images) > 0 and all_negative_images[0][0].shape != (0, 3, 224, 224):
                for i in range(num_instances_to_plot):
                    try:
                        selected_neg_img = all_negative_images[0][0][i].cpu().numpy()
                        selected_neg_img = selected_neg_img.transpose((1, 2, 0))
                        selected_neg_img_sal = selected_neg_img.transpose((2, 0, 1))
                        selected_neg_img_sal = np.expand_dims(selected_neg_img_sal, axis=0)
                        axs[i+num_instances_to_plot, 0].imshow(selected_neg_img)
                        axs[i+num_instances_to_plot, 0].set_title(f'Negative Instance {i+1}')
                        axs[i+num_instances_to_plot, 0].axis('off')

                        selected_neg_img_sal = torch.tensor(selected_neg_img_sal).to(device)
                        neg_saliency_map = saliency.attribute(selected_neg_img_sal)
                        neg_saliency_map = neg_saliency_map.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
                        axs[i+num_instances_to_plot, 1].imshow(neg_saliency_map)
                        axs[i+num_instances_to_plot, 1].axis('off')
                    except Exception as e:
                        print(f"Error processing negative instance {i+1}: {e}")
        except Exception as neg_e:
            print(f"Error in processing negative images: {neg_e}")
    except Exception as main_e:
        print(f"Error in the main plotting loop: {main_e}")

    plt.show()

    try:
        plot_name = f'maps_{model_name}_{analysis}_{epochs}.png'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, 'Results_10folds', 'infections_baseline', f'{fold}')
        os.makedirs(save_dir, exist_ok=True)
        full_plot_path = os.path.join(save_dir, plot_name)
        plt.savefig(full_plot_path)
        plt.close()
    except Exception as save_e:
        print(f"Error saving the plot: {save_e}")
    
    sex_labels_test=np.array(sex_labels_test)
    age_labels_test=np.array(age_labels_test)
    all_index=np.where(sex_labels_test)[0]
    female_sex_index = np.where(sex_labels_test == 1)[0]
    male_sex_index=np.where(sex_labels_test == 0)[0]
    bin_0_18_index=np.where(age_labels_test == 0)[0]
    bin_18_40_index=np.where(age_labels_test == 1)[0]
    bin_40_65_index=np.where(age_labels_test == 2)[0]
    mais_65_index=np.where(age_labels_test == 3)[0]
    # Calculate metrics for the test set
    plot_roc_curve_per_class(y_test,y_proba_test,2,fold)
    accuracy_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    auc_test = roc_auc_score(y_true=y_test, y_score=y_proba_test)
    f1_test = f1_score(y_true=y_test, y_pred=y_pred_test)
    precision_test = precision_score(y_true=y_test, y_pred=y_pred_test)
    recall_test = recall_score(y_true=y_test, y_pred=y_pred_test)
    balanced_accuracy_test = balanced_accuracy_score(y_true=y_test, y_pred=y_pred_test)

    print(f"Test Metrics - ACC:{accuracy_test}, AUC: {auc_test}, F1 Score: {f1_test}, Precision: {precision_test}, Recall: {recall_test}, Balanced Accuracy: {balanced_accuracy_test}")
    subset_indices_list = [all_index,female_sex_index, male_sex_index, bin_0_18_index, bin_18_40_index, bin_40_65_index, mais_65_index]
    subset_names = ['General','Female Sex', 'Male Sex', '0-18 yo', '18-40 yo', '40-65 yo', '+65 yo']
    metrics=['F1','Recall','Precision','BA','ACC','AUC']
    task='Ameba'
    for subset_name, subset_indx in zip(subset_names, subset_indices_list):
        accuracy, auc, precision, recall, f1, BA = compute_metrics_with_indices(subset_indx, y_test, y_pred_test,y_proba_test,subset_name)
        metrics_per_subset[subset_name]['F1']['Ameba'][fold]=f1
        metrics_per_subset[subset_name]['Recall']['Ameba'][fold]=recall
        metrics_per_subset[subset_name]['Precision']['Ameba'][fold]=precision
        metrics_per_subset[subset_name]['BA']['Ameba'][fold]=BA
        metrics_per_subset[subset_name]['ACC']['Ameba'][fold]=accuracy
        metrics_per_subset[subset_name]['AUC']['Ameba'][fold]=auc
        print(f"Metrics for {subset_name} - Task {task}:")
        print(
        f"Accuracy: {accuracy:.4f}, "
        f"AUC: {auc:.4f}, "
        f"F1-Score: {f1:.4f}, "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"BA:{BA:.4f}")
    #confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    mean_confusion_matrix+= conf_matrix
    print(conf_matrix)
    elapsed_time = datetime.now() - start_time
    print(f"Test time", elapsed_time)

    # Generate and print classification report
    class_report = classification_report(y_true=y_test, y_pred=y_pred_test)
    print("Classification Report:")
    print(class_report)
    return metrics_per_subset,mean_confusion_matrix
def unflatten_dict(d, sep='_'):
    unflattened_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        current = unflattened_dict
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = v
    return unflattened_dict
def holm_bonferroni(p_values, alpha=0.05):
    m = len(p_values)
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    sorted_p_values = [p_values[i] for i in sorted_indices]
    
    corrected_p_values = [np.nan] * m
    for i, p in enumerate(sorted_p_values):
        if not np.isnan(p):
            corrected_p_values[sorted_indices[i]] = min(p * (m - i), 1)
        else:
            corrected_p_values[sorted_indices[i]] = np.nan
    
    return corrected_p_values
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    start_time = datetime.now()
    mean_confusion_matrix= np.zeros((2,2))
    subset_names = ['General','Female Sex', 'Male Sex', '0-18 yo', '18-40 yo', '40-65 yo', '+65 yo']
    tasks_list = ['Ameba'] 
    metrics=['F1','Recall','Precision','BA','ACC','AUC']
    metrics_per_subset = {}
    num_folds=3
    for subset_name in subset_names:
        metrics_per_subset[subset_name] = {}  
        for metric in metrics:
            metrics_per_subset[subset_name][metric] = {} 
            for task_name in tasks_list:
                metrics_per_subset[subset_name][metric][task_name] = {}
                for fold in range(1,num_folds+1):
                    metrics_per_subset[subset_name][metric][task_name][fold] = {}
                    

    metrics_for_comp = {}
    for metric in metrics:
        metrics_for_comp[metric] = {} 
        for task_name in tasks_list:
            metrics_for_comp[metric][task_name] = {} 
            for subset_name in subset_names:  
                metrics_for_comp[metric][task_name][subset_name] = {}
                for fold in range(1,num_folds+1):
                    metrics_for_comp[metric][task_name][subset_name][fold] = {}
                    

    confidence_intervals= {}
    for subset_name in subset_names:
        confidence_intervals[subset_name] = {}  
        for metric in metrics:
            confidence_intervals[subset_name][metric] = {} 
            for task_name in tasks_list:
                confidence_intervals[subset_name][metric][task_name] = {}
                
                
    subset_comp = ['sex','age']
    p_values= {}
    for subset_comp in subset_comp:
        p_values[subset_comp] = {}  
        for metric in metrics:
            p_values[subset_comp][metric] = {} 
            for task_name in tasks_list:
                p_values[subset_comp][metric][task_name] = {}    
    # CLI
    parser = argparse.ArgumentParser(description='infection_type.py')
    parser.add_argument('--num_classes', type=int, default=2, required=False, help="Number of classes (to build the networks).")
    parser.add_argument('--epochs', type=int, default=200, required=False, help="Number of training epochs.")
    parser.add_argument('--model_names', nargs='+', type=str, required=False, default=['densenet121'], help="Model(s) to train.")
    #parser.add_argument('--analysis', type=str, required=False, default='infections', help="The type of analysis.")
    args = parser.parse_args()

    # Get transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20), 
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    # Load data
    """
    train_dataset = CorneaUnifespDataset(split="train", transform=transform,classification='bacteria')
    val_dataset = CorneaUnifespDataset(split="validation", transform=transform,classification='bacteria')
    test_dataset = CorneaUnifespDataset(split="test", transform=transform,classification='bacteria')
    
    class_counts = np.bincount(train_dataset.labels)
    print(class_counts)
    class_weights = 1.0 / compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
    sample_weights = class_weights[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16,sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    """
    dataset = CorneaUnifespDataset(transform=transform,classification='ameba')
    # Hyperparameters and training settings
    num_classes = args.num_classes
    epochs = args.epochs
    model_names = args.model_names
    analysis='ameba'
    print(analysis)
    fold_info=[]
    """
    for model_name in model_names:
        model = load_model(model_name, num_classes)
        class_counts= np.unique(train_dataset.labels)
        weights= compute_class_weight(class_weight="balanced",classes=class_counts, y=train_dataset.labels)
        class_weights=torch.from_numpy(weights)
        #print(class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        print("this node works")
        train_and_evaluate(model, train_loader, val_loader, test_loader, class_weights, optimizer, epochs,num_classes,model_name,analysis)
        #print("model runed")"""
    batch_size = 16
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
    flipped_indices = np.array([i for i, img in enumerate(dataset.images) if 'flipped' in img]).astype(int)
    not_flipped_indices = np.array([i for i, img in enumerate(dataset.images) if 'flipped' not in img]).astype(int) 
    img= np.array([dataset.images[i] for i in not_flipped_indices])
    labels= np.array([dataset.labels[i] for i in not_flipped_indices])
    fold = 1
    for train_index, test_index in skf.split(img,labels):

        X_train = np.array(img)[train_index.astype(int)]
        X_train = X_train[..., np.newaxis]
        y_train = np.array(labels)[train_index.astype(int)]
        
        X_test = np.array(img)[test_index.astype(int)]
        X_test = X_test[..., np.newaxis]
        y_test = np.array(labels)[test_index.astype(int)]

        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,train_size= 0.8/0.9,random_state= 42, stratify=y_train)
        flipped_filenames = []
        for filenames_list in X_train:
            fname = filenames_list[0]
            basename, extension = fname.rsplit('.', 1)
            flipped_filename = f"{basename}_flipped.{extension}"
            flipped_filenames.append([flipped_filename])
        X_train = np.vstack((X_train, np.array(flipped_filenames)))
    
        flipped_filenames = []
        for filenames_list in X_val:
            fname = filenames_list[0] 
            basename, extension = fname.rsplit('.', 1)
            flipped_filename = f"{basename}_flipped.{extension}"
            flipped_filenames.append([flipped_filename])
        X_val = np.vstack((X_val, np.array(flipped_filenames)))
        
        flipped_filenames = []
        for filenames_list in X_test:
            fname = filenames_list[0]
            basename, extension = fname.rsplit('.', 1)
            flipped_filename = f"{basename}_flipped.{extension}"
            flipped_filenames.append([flipped_filename])
        X_test = np.vstack((X_test, np.array(flipped_filenames)))

        train_indices = [dataset.images.index(fname) for fname in X_train]
        val_indices = [dataset.images.index(fname) for fname in X_val]
        test_indices = [dataset.images.index(fname) for fname in X_test]

        # Create SubsetRandomSampler instances using the obtained indices
        dataset_train = Subset(dataset=dataset, indices=train_indices)
        dataset_val = Subset(dataset=dataset, indices=val_indices)
        dataset_test = Subset(dataset=dataset, indices=test_indices)
        # Create DataLoaders with the specified samplers
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        print(f"Training fold {fold}/{skf.get_n_splits()}")

        for model_name in model_names:
            model = load_model(model_name, num_classes)
            class_counts, _ = np.unique(y_train, return_counts=True)
            weights = compute_class_weight(class_weight="balanced", classes=class_counts, y=y_train)
            class_weights = torch.from_numpy(weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,weight_decay=1e-8)
            print("this node works")
            elapsed_time = datetime.now() - start_time
            print(f"Time until start of fold {fold}:", elapsed_time)
            metrics_per_subset,mean_confusion_matrix=train_and_evaluate(model, train_loader, val_loader,test_loader, class_weights, optimizer, epochs,model_name, analysis,fold,metrics_per_subset,mean_confusion_matrix)
            for subset_name_Save in subset_names:
                for metric_Save in metrics:
                    for task_name_Save in tasks_list:
                        for fold_Save in range(1,num_folds+1):
                            metrics_for_comp[metric_Save][task_name_Save][subset_name_Save][fold_Save] = metrics_per_subset[subset_name_Save][metric_Save][task_name_Save][fold_Save]
        fold_info.append({
        "fold": fold,
        "train_fnames": X_train,
        "val_fnames": X_val,
        "test_fnames": X_test})
        fold += 1  # Increment fold count
    mean_confusion_matrix /= num_folds
    print('Average Confusion Matrix across folds')
    print('ameba:')
    print(mean_confusion_matrix)
    elapsed_time = datetime.now() - start_time
    print(f"Start of CIs math", elapsed_time)
    #Results
    #Results CIs
    for subset_name, metrics_dict in metrics_per_subset.items():
        for metric, tasks_dict in metrics_dict.items():
            for task_name, folds_dict in tasks_dict.items():
                folds = folds_dict.keys()
                # Calculate mean and standard deviation across all folds
                values = [value for value in folds_dict.values() if isinstance(value, (int, float))]
               
                mean_all_folds = np.mean(values)
                std_dev_all_folds = np.std(values)
                
                # Calculate confidence interval for all folds (assuming 95% confidence level)
                n_folds = len(folds)
                confidence_interval = st.norm.interval(0.95, loc=mean_all_folds, scale=std_dev_all_folds/np.sqrt(n_folds))
                confidence_intervals[subset_name][metric][task_name]=confidence_interval
                print(f"Subset: {subset_name}, Metric: {metric}, Task: {task_name} - Confidence Interval for all folds:", confidence_interval)
                # Example p-value calculation (t-test between sexes for one age bin)
                print("------------------")
                
    subset_comp='sex'
    for metric, tasks_dict in metrics_for_comp.items():
        for task_name, subs_dict in tasks_dict.items():
            for subset_name,folds_dict in subs_dict.items():
                if subset_name not in ['Female Sex', 'Male Sex']:  # Skip other subsets
                    continue
                
                folds = folds_dict.keys()
                # Calculate mean and standard deviation across all folds
                if subset_name == 'Female Sex':
                    fem = [value for value in folds_dict.values() if isinstance(value, (int, float))]
                if subset_name == 'Male Sex':
                    males = [value for value in folds_dict.values() if isinstance(value, (int, float))]    
            t_stat, p_value = st.ttest_ind(fem,males)
            p_values[subset_comp][metric][task_name]=p_value
            if p_value >= 0.05:
                print(f"p-value for t-test between male and female: {p_value} -differences for {metric} and tasks {task_name} in sex are not signifficant")
            else :
                print(f"!p-value for t-test between male and female: {p_value} -differences for {metric} and tasks {task_name} in sex are signifficant!")
    subset_comp='age'
    for metric, tasks_dict in metrics_for_comp.items():
        for task_name, subs_dict in tasks_dict.items():
            for subset_name,folds_dict in subs_dict.items():
                if subset_name not in ['0-18 yo', '18-40 yo', '40-65 yo', '+65 yo']:  # Skip other subsets
                    continue
                
                folds = folds_dict.keys()
                # Calculate mean and standard deviation across all folds
                if subset_name == '0-18 yo':
                    max18 = [value for value in folds_dict.values() if isinstance(value, (int, float))]
                if subset_name == '18-40 yo':
                    max40 = [value for value in folds_dict.values() if isinstance(value, (int, float))]  
                if subset_name == '40-65 yo':
                    max65 = [value for value in folds_dict.values() if isinstance(value, (int, float))]  
                if subset_name == '+65 yo':
                    max100 = [value for value in folds_dict.values() if isinstance(value, (int, float))]     
            t_stat, p_value = st.f_oneway(max18,max40,max65,max100)
            p_values[subset_comp][metric][task_name]=p_value
            if p_value >= 0.05:
                print(f"p-value for t-test between ages: {p_value} -differences for {metric} and tasks {task_name} in age are not signifficant")
            else :
                print(f"!p-value for t-test between ages: {p_value} -differences for {metric} and tasks {task_name} in age are signifficant!")     
    
    flattened_p_values = flatten_dict(p_values)
    p_value_list = [p for p in flattened_p_values.values()]
    corrected_p_values = holm_bonferroni(p_value_list)
    corrected_flattened_p_values = {key: corrected_p_values[i] for i, key in enumerate(flattened_p_values.keys())}
    corrected_p_values_dict = unflatten_dict(corrected_flattened_p_values)
    for category, metrics_dict in corrected_p_values_dict.items():
        for metric, tasks_dict in metrics_dict.items():
            for task_name, p_value in tasks_dict.items():
                if p_value < 0.05:
                    print(f"!p-value for {metric} and task {task_name} in {category}: {p_value} - differences are significant!")
                else:
                    print(f"p-value for {metric} and task {task_name} in {category}: {p_value} - differences are not significant")  
   
    #save pickles
    # Define the folder path to save the pickle files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir,'Results_10folds','infections_baseline')
    os.makedirs(folder_path, exist_ok=True)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pickle_files = {
        "metrics_per_subset.pkl": metrics_per_subset,
        "confidence_intervals.pkl": confidence_intervals,
        "pvalues.pkl": p_values,
        "p_values_corrected.pkl":corrected_p_values_dict
    }
    for file_name, data in pickle_files.items():
        # Combine the folder path and file name to get the full file path
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

   
    #save folds
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir,'Results_10folds','infections_baseline')
    os.makedirs(save_dir, exist_ok=True)
    plot_name=f'folds_{model_name}_{analysis}_{epochs}.csv'
    full_plot_path = os.path.join(save_dir, plot_name)
    with open(full_plot_path, mode='w', newline='') as csv_file:
        fieldnames = ['fold', 'train_fnames', 'val_fnames', 'test_fnames']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_info)
    print("code is done, yey")
    
