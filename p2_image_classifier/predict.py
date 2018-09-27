import numpy as np

# Pytorch
import torch
from torch import nn
from torchvision import models

# Miscellanious
from PIL import Image
import json
import argparse

def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath)
    model = getattr(models, arch)(pretrained=True)
    
    if 'densenet' in arch:
        num_features = model.classifier.in_features
    elif 'resnet' in arch:
        num_features = model.fc.in_features
    elif 'vgg' in arch:
        # We want to add new layers to classifier, so get the last output size
        num_features = model.classifier[-1].out_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define new classifier
    classifier = nn.Sequential(
        nn.Linear(num_features, checkpoint['clf_output_size']),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(checkpoint['clf_output_size'], 102),
        nn.LogSoftmax(dim=1)
    )
    
    if 'resnet' in arch:
        model.fc = classifier
    elif 'vgg' in arch:
        # Unfreeze classifier because we just add new layers to it
        for param in model.classifier.parameters():
            param.requires_grad = True
        model.classifier = nn.Sequential(model.classifier, classifier)
    else:
        model.classifer = classifier
        
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Pre-process image
    image.thumbnail((256, 256))
    # CenterCrop: (left, upper, right, lower)-tuple
    w, h = image.size
    image = image.crop((w//2 - 224//2, h//2 - 224//2, w//2 + 224//2, h//2 + 224//2))
    image = np.array(image)
    
    # Normalize image
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image / 255 - means) / std
    
    return image.T


def predict(image_path, model, idx_to_class, use_gpu, topk=5):
    
    image = Image.open(image_path)
    image = torch.from_numpy(process_image(image))
    device = torch.device("cpu")
    if use_gpu:
        device = torch.device("cuda:0")
        image = image.unsqueeze(0).type(torch.cuda.FloatTensor)
    image.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        exp_output = torch.exp(output)
        probs, max_classes = exp_output.data.topk(5)
        if use_gpu:
            max_classes = max_classes.cpu()
        max_idx = [idx_to_class[i] for i in max_classes.numpy()[0]]
        
    return probs, max_idx

def main():
    parser = argparse.ArgumentParser(description='Predicting classes of flowers')
    parser.add_argument('input', action='store', help='Path of input image')
    parser.add_argument('checkpoint', action='store', help='Path of checkpoint')
    parser.add_argument('--model', action='store', dest='model_name', help='Name of the architecture', default='densenet121')
    parser.add_argument('--top_k', action='store', dest='top_k', help='K most-likely classes', default=5)
    parser.add_argument('--gpu', action='store', dest='gpu', help='Use gpu or not', default=True)
    parser.add_argument('--category_names', action='store', dest='category_names', help='Path to JSON file contain names of classes')
    args = parser.parse_args()
    
    # Collect inputs
    image_path = args.input
    if image_path == None:
        return "No input image"
    checkpoint_path = args.checkpoint
    model_name = args.model_name
    top_k = args.top_k
    gpu = args.gpu
    
    # Build model
    model, checkpoint = load_checkpoint(checkpoint_path, model_name)
    if gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda:0")
    else:
        use_gpu = False
        device = torch.device("cpu")
  
    model = model.to(device)
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {value:key for key, value in class_to_idx.items()}
    probs, max_classes = predict(image_path, model, idx_to_class, device, topk=top_k)
    
    classes_json = args.category_names
    if (classes_json != None):
        with open(classes_json, 'r') as f:
            cat_to_name = json.load(f)
            max_classes = [cat_to_name[i] for i in max_classes]
    print(f'Top {top_k} predictions:')
    for i in range(top_k):
        print(f'   {i + 1}: {max_classes[i]}, with probability of {probs[0][i]}')
        
        
if __name__ == "__main__":
    main()