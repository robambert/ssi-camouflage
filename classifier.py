import torchvision.models as models
from torch import unsqueeze, sort
from torch.nn.functional import softmax
from torchvision.transforms import transforms
import numpy as np
from PIL import Image

MNASNET = models.mnasnet1_0(pretrained=True)
MNASNET.eval()
TRANSFORM = transforms.Compose([transforms.Resize(256), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

with open('imagenet_classes.txt') as f:
  CLASSES = [line.strip() for line in f.readlines()]

def evaluate_image(path:str, nb_results:int):
    img = Image.open(path)
    batch = unsqueeze(TRANSFORM(img), 0)
    out = MNASNET(batch)
    percentage = softmax(out, dim=1)[0] * 100
    _, indices = sort(out, descending=True)
    result = [(CLASSES[idx], percentage[idx].item()) for idx in indices[0][:nb_results]]
    for elt in result:
        print("%s : %f" % (elt[0], elt[1]))

evaluate_image('images/mbappe.jpeg', 5)

