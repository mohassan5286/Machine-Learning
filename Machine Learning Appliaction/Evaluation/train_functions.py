# Imports here

# import torch
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

# define the transformation for the data
def data_transformation():
    """Define the transformation for training data, validation data and testing data
    
    INPUT: 
    None.

    OUTPUT:
    train_transforms: the transformation of the training data.
    valid_transforms: the transformation of the validation data.
    test_transforms : the transformation of the testing data.       
    """
    # the transformation of the training data
    train_transforms = transforms.Compose([
                       transforms.RandomResizedCrop(224, scale = (.8, 1.0)),
                       transforms.RandomRotation(30),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ])

    # the transformation of the validation data
    valid_transforms = transforms.Compose([
                       transforms.Resize((256, 256)),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ])

    # the transformation of the testing data
    test_transforms  = transforms.Compose([
                       transforms.Resize((256, 256)),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ])    

    return train_transforms, valid_transforms, test_transforms

# load the datasets
def load_datasets(data_dir, train_transforms, valid_transforms, test_transforms):

    """Load the dataset for training data, validation data and testing data 
    
    INPUT: 
    data_dir        : the directory of the folder that contains the data.
    train_transforms: the transformation of the training data.
    valid_transforms: the transformation of the validation data.
    test_transforms : the transformation of the testing data.

    OUTPUT:
    train_datasets: the training dataset.
    valid_datasets: the validation dataset.
    test_datasets : the testing datasets.       
    """
    # define the directory of the training data, validation data and testing data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    # load the datasets of training data, validation data and testing data   
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets  = datasets.ImageFolder(test_dir, transform = test_transforms)

    return train_datasets, valid_datasets, test_datasets

def load_dataloaders(train_datasets, valid_datasets, test_datasets):
    """Load the dataloaders for training data, validation data and testing data 
    
    INPUT: 
    train_datasets: the training datasets.
    valid_datasets: the validation datasets.
    test_datasets : the testing datasets.

    OUTPUT:
    trainloaders: the loaders of training data. 
    validloaders: the loaders of validation data.
    testloaders : the loaders of testing data.
    """
    
    # Load the dataloaders for training data, validation data and testing data 
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    testloaders  = torch.utils.data.DataLoader(test_datasets, batch_size = 64)

    return trainloaders, validloaders, testloaders 

# create the model
def create_model(model_name, hidden_units, learning_rate, using_gpu):
    """Create the model
    
    INPUT: 
    model_name    : the name of the pretrained model you want to use.
    hidden_units  : the number of hidden units in the hidden layer you want to use. 
    learning_rate : the learning rate you want to use.
    using_gpu     : determine if you want to use GPU or CPU.

    OUTPUT:
    model    : the model you want to train it. 
    criterion: used to calculate the loss.
    optimizer: used to update the the parameters.
    """

    # import the pretrained model
    model = eval("models." + model_name + "(pretrained = True)")

    # frezze parameters of the model
    for parameters in model.parameters():
        parameters.requires_grad = False

    # access to the final layer name and access to its number of inputs

    # access to the name and features of all layers and store them in a list     
    layers = [(name, features) for name, features in model.named_children()]
    # access to the name of the final layer
    last_layer_name = layers[-1][0]
    # access to the number of input features of the last layer 
    try:
        # if the last layer consists of sequential linear layers
        input_units = layers[-1][1][0].in_features
    except:
        # if the last layer consists of one linear layer
        input_units = layers[-1][1].in_features
        
    # create a new classifer
    classifier = nn.Sequential(
                 nn.Linear(input_units, hidden_units),
                 nn.ReLU(),
                 nn.Dropout(p = .2),
                 nn.Linear(hidden_units, 102),
                 nn.LogSoftmax(dim = 1)
                 )

    # assign the new classifier
    setattr(model, last_layer_name, classifier)

    # create the criterion
    criterion = nn.NLLLoss()

    # create the optimizer
    last_layer = getattr(model, last_layer_name)
    optimizer  = optim.Adam(last_layer.parameters(), lr = learning_rate)

    # use GPU or CPU
    device = torch.device("cuda" if using_gpu else "cpu")
    model.to(device)

    return model, criterion, optimizer

# train the model and test it in the validation data
def train_model(using_gpu, model, criterion, optimizer, trainloaders, validloaders, epochs):
    """Train the model and test it in the validation data
    
    INPUT: 
    using_gpu   : determine if you want to use GPU or CPU.
    model       : the model which you will train. 
    criterion   : used to calculate the loss.
    optimizer   : used to update the the parameters.
    trainloaders: the loaders of training data. 
    validloaders: the loaders of validation data.
    epochs      : the number of epochs to train the model.

    OUTPUT:
    None.
    """

    # use GPU or CPU
    device = torch.device("cuda" if using_gpu else "cpu")

    #train the model and test it in the validation set
    for epoch in range(epochs):

        ## train the model
        model.train()

        training_loss = 0

        for images, labels in trainloaders:

            images, labels = images.to(device), labels.to(device)

            outputs        = model(images)
            loss           = criterion(outputs, labels)
            training_loss  += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test the model in validation set
        with torch.no_grad():

            model.eval()

            validation_loss = 0
            accuracy        = 0

            for images, labels in validloaders:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss    = criterion(outputs, labels)
                ps      = torch.exp(outputs)

                top_p, top_class   =  ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)

                validation_loss += loss.item()
                accuracy        += torch.mean(equals.type(torch.FloatTensor))

            # calculate the average training loss and average validation loss the average accuracy on validation data
            average_training_loss    = training_loss    / len(trainloaders)
            average_validation_loss  = validation_loss / len(validloaders)
            average_accuracy         = accuracy       / len(validloaders) * 100

            print(f"epoch: {epoch + 1} / {epochs} .. training_loss: {average_training_loss:.3f} .. validation loss: {average_validation_loss:.3f} .. accuracy: {average_accuracy:.3f}%")

# test the model
def test_model(using_gpu, model, criterion, testloaders):
    """Test the model in testing set
    
    INPUT: 
    using_gpu  : determine if you want to use GPU or CPU.
    model      : the model which you will test. 
    criterion  : used to calculate the loss.
    testloaders: the loaders of testing data.
    
    OUTPUT:
    None.
    """

    # use GPU or CPU
    device = torch.device("cuda" if using_gpu else "cpu")

    # test model
    with torch.no_grad():

        model.eval()

        running_loss = 0
        accuracy     = 0

        for images, labels in testloaders:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            ps      = torch.exp(outputs)

            top_p, top_class   =  ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)

            running_loss += loss.item()
            accuracy     += torch.mean(equals.type(torch.FloatTensor))

        # calculate the average running loss and the average accuracy
        average_running_loss = running_loss / len(testloaders)
        average_accuracy     = accuracy    / len(testloaders) * 100

        print(f"accuracy: {average_accuracy:.3f}% .. running loss: {average_running_loss:.3f}")

# save checkpoint
def save_checkpoint(save_dir, epochs, model_name, input_units, hidden_units, image_datasets, optimizer, model):
    """Save the model 
    
    INPUT: 
    save_dir      : the directory you want to save the model in.
    epochs        : the number of epochs to train the model.
    model_name    : the name of the pretrained model you used.
    inputs_units  : the number of inputs units to the last layer in the model.
    hidden_units  : the number of the hidden units in the hidden layer.
    image_datasets: the dataset of the training data or validation data or testing data .
    optimizer     : the optimizer you used to update the model parameters.
    model         : the model you trained.

    OUTPUT:
    None.
    """

    checkpoint = {
                  "epochs": epochs,
                  "model_name": model_name,
                  "input_size": input_units,
                  "output_size": 102,
                  "hidden_layers": [hidden_units],
                  "class_to_idx": image_datasets.class_to_idx,
                  "optimizer_state_dict":optimizer.state_dict(),
                  "state_dict": model.state_dict()
                 }

    torch.save(checkpoint, save_dir + "checkpoint.pth")