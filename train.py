import torch
import numpy as np
from Model.ViViT_FE import ViViT_FE
from Model.st_gcn import GGCN
from Model.ViViT_Pose import ViViT_Pose
from configuration import build_config
from dataloader import TinyVirat, VIDEO_LENGTH, TUBELET_TIME, NUM_CLIPS, POSE_POINT_COUNT, POSE_POINT_SIZE

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import re
import os

exp='2'

#Make exp dir
exp_path = os.path.join('exps', f'exp_{exp}')
os.makedirs(exp_path, exist_ok=True)

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#Data parameters
max_pose_objects = 4
tubelet_dim=(3,TUBELET_TIME,4,4) #(ch,tt,th,tw)
pose_tubelet_dim=(POSE_POINT_SIZE,TUBELET_TIME)
num_classes=26
img_res = 128
vid_dim=(img_res,img_res,VIDEO_LENGTH) #one sample dimension - (H,W,T)
pose_vid_dim=(POSE_POINT_COUNT,max_pose_objects,VIDEO_LENGTH) #one sample dimension - (H,W,T)


# Training Parameters
print("Creating params....")
params = {'batch_size':2,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 250
gradient_accumulations = 1
inf_threshold = 0.7

#Data Generators
cfg = build_config('TinyVirat')

train_dataset = TinyVirat(cfg, 'train', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res, max_pose_objects=max_pose_objects)
training_generator = DataLoader(train_dataset,**params)

val_dataset = TinyVirat(cfg, 'val', 1.0, num_frames=tubelet_dim[1], skip_frames=2, input_size=img_res, max_pose_objects=max_pose_objects)
validation_generator = DataLoader(val_dataset, **params)

#Define model
print("Initiating Model...")
checkpoint_path_base = os.path.join(exp_path, 'checkpoints')
os.makedirs(checkpoint_path_base, exist_ok=True)
top_checkpoints = sorted([f.name for f in os.scandir(checkpoint_path_base)], key=lambda x: int(re.match(r'epoch_([0-9]+)', x)[1]), reverse=True)
checkpoint_path = os.path.join(checkpoint_path_base, top_checkpoints[0]) if len(top_checkpoints) > 0 else None
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    print(f'Checkpoint from Epoch {epoch} Loaded')
else:
    checkpoint = None

spat_op='cls' #or GAP

#model=ViViT_FE(vid_dim=vid_dim,num_classes=num_classes,tubelet_dim=tubelet_dim,spat_op=spat_op)
model=ViViT_Pose(vid_dim=pose_vid_dim,num_classes=num_classes,tubelet_dim=pose_tubelet_dim,spat_op=spat_op)
if checkpoint: model.load_state_dict(checkpoint['model_state_dict'])

model=model.to(device)

#Define loss and optimizer
lr=0.01
wt_decay=5e-4
criterion=torch.nn.BCEWithLogitsLoss() #CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=wt_decay)
if checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

'''
#ASAM
rho=0.5
eta=0.01
minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
'''

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
if checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0
score = checkpoint['score'] if checkpoint else []

def compute_accuracy(pred,target,inf_th):
    pred = pred
    target = target.cpu().data.numpy()
    #Pass pred through sigmoid
    pred = torch.sigmoid(pred)
    pred = pred.cpu().data.numpy()
    #Use inference throughold to get one hot encoded labels
    pred = pred > inf_th

    #Compute equal labels
    return accuracy_score(pred,target)

#TRAINING AND VALIDATING
if __name__ == '__main__':
    for epoch in range(start_epoch, max_epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        for batch_idx, (inputs, pose_inputs, targets) in enumerate(tqdm(training_generator)):
            inputs = inputs.to(device)
            #print("Targets shape : ",targets.shape)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Ascent Step
            predictions = model(pose_inputs.float()); #targets = torch.tensor(targets,dtype=torch.long); predictions = torch.tensor(predictions,dtype=torch.long)

            batch_loss = criterion(predictions, targets)

            # compute gradients of this batch.
            (batch_loss / gradient_accumulations).backward()
            # so each parameter holds its gradient value now,
            # and when we run `loss.backward()` again in next batch iteration,
            # then the previous gradient computed and the current one will be added.
            # this is the default behaviour of gradients in pytorch.

            if (batch_idx + 1) % gradient_accumulations == 0:
                optimizer.step()
                model.zero_grad()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                accuracy +=  compute_accuracy(predictions,targets,inf_threshold)
            cnt += len(targets) #number of samples
            scheduler.step()

        loss /= cnt;
        accuracy /= (batch_idx+1)
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        scheduler.step()

        #Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for batch_idx, (inputs, pose_inputs, targets) in enumerate(validation_generator):
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(pose_inputs.float())
                loss += criterion(predictions, targets).sum().item()
                accuracy += compute_accuracy(predictions,targets,inf_threshold)
                cnt += len(targets)
            loss /= cnt
            accuracy /= (batch_idx+1)

        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")

        score.append((loss, accuracy))
        checkpoint_path = os.path.join(checkpoint_path_base, f'epoch_{epoch}.pt')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'score': score,
        }, checkpoint_path)
