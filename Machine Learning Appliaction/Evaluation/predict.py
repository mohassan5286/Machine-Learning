# Imports here

# import argparse
import argparse

# import test_functions
import predict_functions

# create the parser
parser = argparse.ArgumentParser("pass the parameter to predict the top k classes of the image")

# create path_to_image argument which is the path of the image
parser.add_argument("path_to_image", help = "the path of the image", type = str)

# create checkpoint argument which is the path of the checkpoint.pth file
parser.add_argument("checkpoint", help = "the path of the checkpoint.pth file", type = str)

# create topk argument which is the top k classes you want to show
# the default value is 3
parser.add_argument("--topk", help = "the top k most likely classes", default = 3, type = int)

# create category_name argument which is the the path of json file that map categories to real name
# the default is cat_to_name.json (the file is called cat_to_name.json & exist in the current directory)
parser.add_argument("--category_names", help = "the path of json file that map categories to real name", default = "cat_to_name.json",  type = str)

# create the gpu argument which is used to indicate whether we want to use GPU or CPU
parser.add_argument("--gpu", help = "", action = "store_true")

# create args which contain all the arguments
args = parser.parse_args()

# assign all the arguments
path_to_image  = args.path_to_image 
checkpoint     = args.checkpoint
topk           = args.topk
category_names = args.category_names
using_gpu      = args.gpu

# predict the top k classes of the image
predict_functions.result(path_to_image, checkpoint, topk, category_names, using_gpu)

