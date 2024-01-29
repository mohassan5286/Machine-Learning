# Imports here

# import torch
import torch
from torch import nn, optim
from torchvision import transforms, models

# import numpy
import numpy as np

# import Image
from PIL import Image

# import json
import json

# import matplotlib
import matplotlib.pyplot as plt

# load the checkpoint
def load_checkpoint(checkpoint_path, using_gpu):
    """Load the checkpoint 

    INPUT:
    checkpoint_path: the path of the checkpoint.pth file. 
    using_gpu      : determine if you want to use GPU or CPU. 

    OUTPUT:
    model       : the checkpoint model.
    criterion   : used to calculate the loss.
    optimizer   : used to update the parameters.
    class_to_idx: used to turn the class of the images to index. 
    """
    
    # choose the current device(GPU or CPU) 
    device = torch.device("cuda" if using_gpu else "cpu")
    
    # load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = device)
 
    # import the pretrained model we use
    model = eval("models." + checkpoint["model_name"] + "(pretrained = True)").to(device)

    # create the architecture of our classifier 
    module = []

    module.append(nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"][0]))
    module.append(nn.ReLU())
    module.append(nn.Dropout(p = .2))

    for index in range((len(checkpoint["hidden_layers"])) - 1):
        module.append(nn.Linear(checkpoint["hidden_layers"][index], checkpoint["hidden_layers"][index + 1]))
        module.append(nn.ReLU())
        module.append(nn.Dropout(p = .2))

    module.append(nn.Linear(checkpoint["hidden_layers"][-1], checkpoint["output_size"]))
    module.append(nn.LogSoftmax(dim = 1))

    classifier = nn.Sequential(*module)

    # reach to the name and features of all layers
    layers = [(name, features) for name, features in model.named_children()]

    # access to the last layer name
    last_layer_name  = layers[-1][0]

    # assign the classifier to our model
    setattr(model, last_layer_name, classifier)

    # load the parameters to the model
    model.load_state_dict(checkpoint["state_dict"])

    # create criterion
    criterion = nn.NLLLoss()

    # create the architecture of optimizer and load its parameters
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    class_to_idx = checkpoint["class_to_idx"]

    return model, criterion, optimizer, class_to_idx

# apply processes to the image
def process_image(image):
    """Apply processes to an image to be ready as an input to the model 

    INPUT:
    image: the image which we will apply the processes in

    OUTPUT:
    np_image: an image as numpy array after applying the transformation
    """
    # define the resize and crop transformations for the image
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224)  
                ])
    
    # apply resize and crop transformations to the image
    image = transform(image) 

    # convert the pil_image to numpy array & convert from 0 ~ 255 integers to 0 ~ 1 floats
    np_image = np.array(image) / 255.0

    # define the mean and standard deviation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # normalize the image
    np_image = (np_image - mean) / std

    # make the color channel to be the first dimension because PyTorch tensor expect the color channel to be the first dimension
    np_image = np_image.transpose((2,0,1))
    
    return np_image

# show the image
def imshow(image, ax=None, title=None):
    """Show the image

    INPUT:
    image: the image we use.
    ax   : the axis of the plot which should contain the image.
    title: the title of the plot which contains the image.

    OUTPUT:
    ax: the axis of the plot after plot the image.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.set_title(title) 
    ax.imshow(image)

    return ax

# predict the classes of the image
def predict(path_to_image, checkpoint_path, topk, using_gpu):
    """Predict the classes of the images and its probabilities

    INPUT:
    path_to_image  : the path of the image.
    checkpoint_path: the path of the checkpoint.pth file.
    topk           : the top k classes you want to predict.
    using_gpu      : determine if you want to use GPU or CPU.

    OUTPUT:
    probs  : the probabilities of the classes.
    classes: the indices of the classes. 
    """
    # open the image
    image = Image.open(path_to_image)

    # use GPU or CPU
    device = torch.device("cuda" if using_gpu else "cpu")
    
    # apply a process_image function to the image to make the image be ready as input to the model
    image = process_image(image)

    # convert the image from numpy array to tensor
    image = torch.tensor(image, dtype=torch.float32).to(device)

    # add a dimension of batch to be like the input images 
    image = image.unsqueeze(0)
    
    # load the checkpoint
    model, criterion, optimizer, class_to_idx = load_checkpoint(checkpoint_path, using_gpu)

    # calculate the top k probabilities and their indices
    model.eval()

    with torch.no_grad():
        ps = torch.exp(model(image))
        top_p, top_class = ps.topk(topk, dim = 1)

    # convert the probabilites to a list and  
    probs = top_p.tolist()[0]
    probs = [round(prob, 8) for prob in probs]

    # convert from the classes to indices and store it in a list 
    classes = [key for key, value in class_to_idx.items() if value in top_class]
    
    return probs, classes

# show a plot for the image with top classes and its probabilities
def result(path_to_image, checkpoint_path, topk, category_names, using_gpu):
    """show a plot for the image with top classes and its probabilities

    INPUT:
    path_to_image  : the path of the image.
    checkpoint_path: the path of the checkpoint.pth file.
    topk           : the top k classies you want to show.
    category_name  : the mapping from indicies to the name of the class. 
    using_gpu      : determine if you want to use GPU or CPU.

    OUTPUT:
    None.
    """
    # create the dictionary of categories to name
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # calculate the top k probabilities and their classes
    probs, classes = predict(path_to_image, checkpoint_path, topk, using_gpu) 

    # convert from classes indicies to names
    classes_names = []
    for index in range(topk):
        classes_names.append(cat_to_name.get(classes[index])) 

    # open the image
    image = Image.open(path_to_image)

    # plot the image
    plt.subplot(2, 1, 1)
    ax = plt.gca()
    x = imshow(torch.from_numpy(process_image(image)), ax, title = classes_names[0])
    ax.axis("off")

    # plot the horizontal bar chart for top k probabilities
    plt.subplot(2, 1, 2)
    ax = plt.gca()
    ax.barh(classes_names, probs)
    plt.tight_layout()
    plt.show()