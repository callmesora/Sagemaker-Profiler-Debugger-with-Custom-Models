#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from torchvision import datasets, models, transforms


from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

import argparse
import os
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys



import os
import sys
import logging
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader,criterion,device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # based on https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-debugger/pytorch_model_debugging/scripts/pytorch_mnist.py
    
    model.eval()
    # ===================================================#
    # 3. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    loss_fn = criterion
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer,device,epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    # =================================================#
    # 2. Set the SMDebug hook for the training phase. #
    # =================================================#
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.set_mode(smd.modes.TRAIN)
        
    loss_fn = criterion
    
    if hook:
        hook.register_loss(criterion)
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0: # Log interval
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # According to https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    net = models.resnet50(pretrained=True)
    #net = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in net.parameters():
        param.requires_grad = False
        
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 133)

    return net


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
                  for x in ['train', 'valid','test']}
    
    logger.info(f"Train_loader size: {len(dataloaders['train'])}, Valid_loader size: {len(dataloaders['valid'])}, test_loader size: {len(dataloaders['test'])}")
    return dataloaders



def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.to(device)
    
    # Initialize Args
    data = args.data # data folder
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_dir = args.model_dir 
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    dataloaders = create_data_loaders(data,batch_size) 
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    
    
    
    for epoch in range(1, 2 + 1):
        # ===========================================================#
        # 5. Pass the SMDebug hook to the train and test functions. #
        # ===========================================================#
        train(model, train_loader, loss_criterion, optimizer,device,epoch)
        test(model, test_loader, loss_criterion,device)
    
    
    
    '''
    TODO: Test the model to see its accuracy
    '''
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(model_dir, "model.pth")
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    
    args=parser.parse_args()
    
    main(args)
