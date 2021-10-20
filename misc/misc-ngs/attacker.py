import torch
import IPython
from PIL import Image
import torch.nn as nn
import shutil
import random
from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2

ALPHA = 2/255
EPS = 8/255
STEPS = 10
LOSS = nn.CrossEntropyLoss()
THRESHOLD = 0.075

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])
model = mobilenet_v2()
model.classifier[1] = torch.nn.Linear(
    in_features=model.classifier[1].in_features, out_features=16)
model.load_state_dict(torch.load('model.pt'))
model.eval()
model.to(device)

for idx in range(2550):
    if random.random() > THRESHOLD:
        continue
    image = transform(Image.open(
        f'images/{idx:04}.png')).to(device).repeat(1, 3, 1, 1)

    label = model(image).argmax(axis=1)
    ori_image = image.clone().detach()
    for iter in range(STEPS):
        image.requires_grad = True
        output = model(image)
        if output.argmax(axis=1) != label:
            break
        cost = LOSS(output, label)
        grad = torch.autograd.grad(cost, image,
                                   retain_graph=False,
                                   create_graph=False)[0]
        adv_image = image + ALPHA*sum(grad[0]).repeat(1, 3, 1, 1).sign()
        a = torch.clamp(ori_image - EPS, min=0)
        b = (adv_image >= a).float()*adv_image + (adv_image < a).float()*a
        c = (b > ori_image+EPS).float()*(ori_image+EPS) + \
            (b <= ori_image + EPS).float()*b
        image = torch.clamp(c, max=1).detach()

    transforms.ToPILImage()(image[0][0]).save(f'tmp.png')
    image = transform(Image.open('tmp.png')).to(device).repeat(1, 3, 1, 1)
    new_label = model(image).argmax(axis=1)
    if new_label != label:
        print(
            f"{idx:04} successed after {iter} iters. {label.item()}->{new_label.item()}")
        shutil.copy("tmp.png", f"images/{idx:04}.png")
