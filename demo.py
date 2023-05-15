from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(image_size, device):
    raw_image = Image.open('demo.jpg').convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def demo_1():
    # Image Captioning
    from models.blip import blip_decoder

    device = 'cuda:0'
    image_size = 384
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])


def demo_2():
    # itc itm
    from models.blip_itm import blip_itm

    image_size = 384
    image = load_demo_image(image_size=image_size, device=device)

    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'

    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device='cpu')

    caption = 'a woman sitting on the beach with a dog'

    print('text: %s' % caption)

    itm_output = model(image, caption, match_head='itm')
    itm_score = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
    print('The image and text is matched with a probability of %.4f' % itm_score)

    itc_score = model(image, caption, match_head='itc')
    print('The image feature and text feature has a cosine similarity of %.4f' % itc_score)


# pip install fairscale
if __name__ == '__main__':
    num = 1
    eval(f'demo_{num}()')
