from data import MNIST
from model import Classifier, Inference
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def test(inference, dataloader):
    with torch.no_grad():
        acc = []
        for images, labels in dataloader:
            pred = inference(images)
            batch_acc = torch.mean((pred == labels).to(torch.float32))
            acc.append(batch_acc)
        print(torch.mean(torch.tensor(acc)))

root = r'C:\Users\elleh\OneDrive\MachineLearningOperation\dtu_mlops\data\corruptmnist'
tasks = ['train', 'test']
datasets = {task: MNIST(root, task) for task in tasks}
dataloaders = {task: DataLoader(datasets[task], batch_size=16, num_workers=0) for task in tasks}
classifier = Classifier()
inference = Inference(classifier)
optimizer = torch.optim.Adam(classifier.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
criterion = torch.nn.CrossEntropyLoss()

losses = []
for i in tqdm(range(5)):
    for images, labels in dataloaders['train']:
        optimizer.zero_grad()
        pred = classifier(images)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

test(inference, dataloaders['test'])
