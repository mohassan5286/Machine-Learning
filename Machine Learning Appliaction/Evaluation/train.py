
# Imports here

# import argparse
import argparse

# import train_functions
import train_functions

# create the parser
parser = argparse.ArgumentParser(description = "pass the arguments for training the model")

# create data_dir argument which is the directory of the folder which contains data
parser.add_argument("data_dir", help = "the directory of the data", type = str)

# create save_dir argument which is the directory where you want to save the model checkpoint
# the default value is "" (save the checkpoint in the current directory) 
parser.add_argument("--save_dir", help = "the directory where you want to save the model", default = "", type = str)

# create arch argument which is the name of pretrained model
# the default value is vgg13
parser.add_argument("--arch", help = "the name of pretrained model you want to use", default = "vgg13", type = str)

# create learning_rate argument which is the learning rate which we will use to train the model
# the default value is .001
parser.add_argument("--learning_rate", help = "the learning rate of your model", default = 0.001,  type = float)

# create hidden_units argument which is the number of hidden_units in the hidden layer
# the default value is 512
parser.add_argument("--hidden_units", help = "the number of hidden units", default = 512 , type = int)

# create epochs argument which is the number of epochs of training the model
# the default value is 20
parser.add_argument("--epochs", help = "the number of epochs to train the model", default = 20, type = int)

# create the gpu argument which is used to indicate whether you want to use GPU or CPU
parser.add_argument("--gpu", help = "", action = "store_true")

# create args which contain all the arguments
args = parser.parse_args()

# assign all the arguments
data_dir      = args.data_dir
save_dir      = args.save_dir
model_name    = args.arch
learning_rate = args.learning_rate
hidden_units  = args.hidden_units
epochs        = args.epochs 
using_gpu     = args.gpu

# define the transformation
train_transforms, valid_transforms, test_transforms = train_functions.data_transformation()

# load the dataset of training data, validation data and testing data
train_datasets, valid_datasets, test_datasets = train_functions.load_datasets(data_dir, train_transforms, valid_transforms, test_transforms)

# load the dataloaders of training data, validation data and testing data
trainloaders, validloaders, testloaders = train_functions.load_dataloaders(train_datasets, valid_datasets, test_datasets)

# create model and criterion which is used to calculate the loss and optimizer which is used to update the parameters
model, criterion, optimizer = train_functions.create_model(model_name, hidden_units, learning_rate, using_gpu)

# train the model
train_functions.train_model(using_gpu, model, criterion, optimizer, trainloaders, validloaders, epochs)

# test the model
train_functions.test_model(using_gpu, model, criterion, testloaders)

# save checkpoint
layers = [(name, features) for name, features in model.named_children()]
# the last layer consists of sequential linear layers
input_units = layers[-1][1][0].in_features
train_functions.save_checkpoint(save_dir, epochs, model_name, input_units, hidden_units, train_datasets, optimizer, model)