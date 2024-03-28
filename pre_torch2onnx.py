import pathlib
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

class PreprocessModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = model
        # self.model.eval()
        self.scale = 1/3.
        self.nw, self.nh = 640, 426
        self.padding_right = 640 - self.nw 
        self.padding_bottom = 640 - self.nh
        self.resize = transforms.Resize((self.nh, self.nw))
        self.pad = nn.ZeroPad2d((0, self.padding_right, 0, self.padding_bottom))

    def forward(self, x):
        # BGR to RGB conversion
        x = x[:, :, :, [2, 1, 0]]
        # Normalize
        x = x / 255.0
        # Permute dimensions BHWC to BCHW
        x = x.permute(0, 3, 1, 2)
        # Resize
        x = self.resize(x)
        # Pad
        x = self.pad(x)
        return x

def torch2onnx():
    device = torch.device('cpu')
    model = PreprocessModule()
    for k, m in model.named_modules():
        m.export = True
    dummy_input = torch.zeros(1, 1280, 1920, 3).to(device)
    model.to(device).eval()


def infer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, help='input image path for inference')
    opt = parser.parse_args()
    model = PreprocessModule()
    model.eval()

    out_root = 'images/infer' 
    image_list = sorted(pathlib.Path(opt.infer_path).glob('*.jpg'))

    # select random 100 images
    infer_list = np.random.choice(image_list, 100)
    pathlib.Path(out_root).mkdir(exist_ok=True)
    for image_path in infer_list:
        img = cv2.imread(str(image_path))
        inp = img.astype('float32') 
        inp = torch.from_numpy(np.expand_dims(inp, 0))

        out = model(inp)

        result_img = out.permute(0, 2, 3, 1).cpu().numpy()
        result_img.shape = (640, 640, 3)
        result_img *= 255.
        result_img = result_img[:, :, ::-1]
        image_name = pathlib.Path(image_path).stem
        cv2.imwrite(f'{out_root}/{image_name}.jpg', result_img)
        print(f'{out_root}/{image_name}.jpg')

if __name__ == '__main__':
    infer()