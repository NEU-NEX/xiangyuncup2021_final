import torch
from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2
from PIL import Image
import random
from tqdm import trange
import ecc

ALLOWED = 0.1
N = 255
K = 223

data, targets = torch.load('dataset.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
model = mobilenet_v2()
model.classifier[1] = torch.nn.Linear(
    in_features=model.classifier[1].in_features, out_features=16)
model.load_state_dict(torch.load('model.pt'))
model.eval()
model.to(device)


def sample_image(cls):
    allow = random.random() < ALLOWED
    while 1:
        im = Image.fromarray(random.choice(
            data[targets == cls]).numpy(), mode='L')
        correct = model(transform(im).repeat(1, 3, 1, 1)
                        ).argmax(axis=1).item() == cls
        if correct or allow:
            return im


cnt = 0
with open('flag.zip', 'rb') as f:
    content = f.read()
    for i in trange(0, len(content), K):
        block = content[i:i+K].ljust(K, b'\0')
        encoded = ecc.rs_encode_msg(block, N-K)
        for b in encoded:
            upper = b >> 4
            lower = b & 0xf
            upper_im = sample_image(upper).save(f'images/{cnt:04}.png')
            cnt += 1
            lower_im = sample_image(lower).save(f'images/{cnt:04}.png')
            cnt += 1
