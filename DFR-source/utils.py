import numpy as np
import os
import cv2
from skimage.io import imread, imsave, imshow
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image, ImageDraw
import time
import cv2
import numpy as np
import torch
import math
import random
from torchvision import transforms


def normalize(x):
    """ Normalize x to [0, 1]
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def visulization(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(saving_path, "gt_{}".format(img_name)), img)

    # binary score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)
    img = img[:, :, ::-1]  # bgr to rgb
    mask_img = img.copy()
    img[..., 1] = np.where(score == 255, 255, img[..., 1])
    
    mask_img[..., 1] = np.where(mask == 255, 255, mask_img[..., 1])

    # save
    imsave(os.path.join(saving_path, "{}".format(img_name)), img)
    imsave(os.path.join(saving_path, "maskgt_{}".format(img_name)), mask_img)

def visulization_score(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    superimposed_img = img.copy()
    cv2.imwrite(os.path.join(saving_path, "origin_{}".format(img_name)), img)

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=-1)
    img = img[:, :, ::-1]  # bgr to rgb
    # normalized score {0, 255}

    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)



    heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET)  # 将score转换成热力图
    superimposed_img_remap = heatmap * 0.7 + superimposed_img * 0.8     # 将热力图叠加到原图像
    # cv2.imwrite('cam.jpg', superimposed_img)  # 将图像保存

    mask_heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # 将score转换成热力图
    superimposed_mask_img = mask_heatmap * 0.7 + superimposed_img * 0.8     # 将热力图叠加到原图像



    # save
    cv2.imwrite(os.path.join(saving_path, "{}".format(img_name)), superimposed_img_remap)
    cv2.imwrite(os.path.join(saving_path, "mask_""{}".format(img_name)), superimposed_mask_img)

    imsave(os.path.join(saving_path, "gt_{}".format(img_name)), img)


def spec_sensi_acc_iou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    iou_score = np.sum(intersection) / np.sum(union)

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    return specificity, sensitivity, accuracy, iou_score, auc_score


def spec_sensi_acc_riou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)      # recall for negtive
    # specificity = np.sum(gt_p * pred_p) / np.sum(pred_p)    # precision
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)      # recall for positive
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    # iou_score = np.sum(intersection) / np.sum(union)
    iou_score = np.sum(intersection) / np.sum(mask)    # relative iou

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return specificity, sensitivity, accuracy, iou_score, auc_score, [fpr, tpr, thresholds]


def auc_roc(mask, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    auc_score = roc_auc_score(mask.ravel(), score.ravel())
    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return auc_score, [fpr, tpr, thresholds]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""

    def __init__(self, colorJitter=0.2, transform=None):
        self.transform = transform

        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness=0.55,
                                                      contrast=0.55,
                                                      saturation=0.55,
                                                      hue=0.5)

    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img


def add_defect(input):
    input2 = input.copy()
    w, h = input2.size
    draw = ImageDraw.Draw(input2)
    img = Image.new('RGB', (w, h), (0, 0, 0))
    defect_shape = ['circle', 'ellipse','square','line']
    only_defect_draw = ImageDraw.Draw(img)
    target_color = (255, 255, 255)
    # np.random.seed(int(time.time()))

    for i in range(20):
        # size_range = (0.002, 0.03)
        size_range = (0.05, 0.09)
        shape = np.random.choice(defect_shape)
        size_ratio = np.random.uniform(size_range[0], size_range[1])
        # np.random.seed(int(np.random.random()))
        x = int(np.random.random() * w)
        y = int(np.random.random() * h)
        size = int(size_ratio * min(w, h))
        # 二值图
        color_select = (0, 255)
        pos = np.random.randint(0, 2, 1)
        # color = (color_select[pos[0]])
        # 灰度图
        color = tuple(np.random.randint(20, 50, 3))

        if shape == 'circle':
            draw.ellipse([x, y, x + size, y + size], fill=color)
            only_defect_draw.ellipse([x, y, x + size, y + size], fill=target_color)
        elif shape == 'square':
            draw.rectangle([x, y, x + size, y + size], fill=color)
            only_defect_draw.ellipse([x, y, x + size, y + size], fill=target_color)

        elif shape == 'ellipse':
            size1 = size * np.random.uniform(0.7, 1.2)
            size2 = size * np.random.uniform(0.9, 1.4)
            draw.ellipse([x, y, x + size1, y + size2], fill=color)
            only_defect_draw.ellipse([x, y, x + size1, y + size2], fill=target_color)

        elif shape == 'line':
            while True:
                x1 = int(np.random.random() * w)
                y1 = int(np.random.random() * h)
                width = int(np.random.randint(1, 14))
                if x1 == x or y1 == y:
                    continue
                draw.line([x, y, x1, y1], fill=color, width=width)
                only_defect_draw.line([x, y, x1, y1], fill=target_color, width=width)
                break
    return input2, img
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """

    def __init__(self, area_ratio=[0.01, 0.03], aspect_ratio=0.005, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img,mask_image):
        # TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        target_color = (255, 255, 255)
        color = tuple(np.random.randint(1, 150, 3))

        size_range = (0.06, 0.15)
        size_ratio = np.random.uniform(size_range[0], size_range[1])
        img_draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask_image) 
        x = int(np.random.random() * w)
        y = int(np.random.random() * h)
        size = int(size_ratio * min(w, h))

        size1 = size * np.random.uniform(0.3, 1.2)
        size2 = size * np.random.uniform(0.9, 1.7)
        # img_draw.ellipse([x, y, x + size1, y + size2], fill=color)
        # mask_draw.ellipse([x, y, x + size1, y + size2], fill=target_color)


        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))

        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]

        patch = img.crop(box)
        random_color = np.random.randint(20, 100)
        image_255 = Image.new('RGB', (w, h), (255, 255, 255))
        image_125 = Image.new('RGB', (w, h), (random_color, random_color, random_color))
        mask_patch = image_255.crop(box)
        # patch = image_125.crop(box)


        if self.colorJitter:
            patch = self.colorJitter(patch)

        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))

        resize_h = cut_w
        resize_w = cut_h
        # resize_ratio = 0.8

        # resize_h = int(cut_h*resize_ratio)
        # resize_w = int(cut_w*resize_ratio)
        patch = patch.resize((resize_w,resize_h))
        mask_patch = mask_patch.resize((resize_w,resize_h))

        insert_box = [to_location_w, to_location_h, to_location_w + resize_w, to_location_h + resize_h]
        augmented = img.copy()
        augmented_mask_image = mask_image.copy()


        augmented.paste(patch, insert_box)
        # augmented.paste(patch, insert_box)
        augmented_mask_image.paste(mask_patch,insert_box)

        return super().__call__(augmented, augmented_mask_image)
    
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w,h = img.size
        #w = img.size

        mask = np.ones((h, w,3), np.float32)
        target_mask=np.zeros((h, w,3), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2,:] = 0.
            target_mask[y1: y2, x1: x2,:] = 255
        mask = torch.from_numpy(mask)
        # np.array(Image.open("lena.jpg"))
        mask = mask.expand_as(torch.from_numpy(np.array(img)))
        img = img * mask.numpy()
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        target_mask = Image.fromarray(target_mask.astype('uint8')).convert('RGB')

        
        return img,target_mask

class RandomErasing(object):
    def __init__(self, EPSILON = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        
        if random.uniform(0, 1) > self.EPSILON:
            return img
        
        img = torch.from_numpy(np.array(img))

        print("img size: ",img.size())
        for attempt in range(2):
            print("attempt: ",attempt)
            area = img.size()[0] * img.size()[1]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            print("h: ",h)
            print("w: ",w)

            if w < img.size()[1] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[0] - h)
                y1 = random.randint(0, img.size()[1] - w)
                if img.size()[2] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[x1:x1+h, y1:y1+w,0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w,1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w,2] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                    print("img")
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                # return img
            
        img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')

        return img

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cut_defect_paste(img,mask):
    colorJitter = transforms.ColorJitter(brightness=0.98,
                                        contrast=0.98,
                                        saturation=0.98,
                                        hue=0.5)
    img_h = img.size[0]
    img_w = img.size[1]
    augmented = img.copy()
    # target_mask = Image.new('RGB', (img_w, img_h), (0, 0, 0))
    aspect_ratio=0.005
    defect_path = '/root/project/mypy/DFR/DFR-source/defect'
    for defect_img_path in os.listdir(defect_path):
        defect_img = Image.open(os.path.join(defect_path,defect_img_path))
        h = defect_img.size[0]
        w = defect_img.size[1]
        resize_h = 50
        resize_w = 50
        defect_img = defect_img.resize((resize_w, resize_h))
        patch = defect_img
        # area_ratio=[0.1, 0.15]
        # ratio_area = random.uniform(area_ratio[0], area_ratio[1]) * w * h

        # log_ratio = torch.log(torch.tensor((aspect_ratio, 1 / aspect_ratio)))
        # aspect = torch.exp(
        #     torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        # ).item()
        

        # cut_w = int(round(math.sqrt(ratio_area * aspect)))
        # cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # # one might also want to sample from other images. currently we only sample from the image itself
        # from_location_h = int(random.uniform(0, h - cut_h))
        # from_location_w = int(random.uniform(0, w - cut_w))

        # box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        
        # patch = defect_img.crop(box)
        patch = colorJitter(patch)
        
        random_color = np.random.randint(20, 100)
        image_255 = Image.new('RGB', (resize_h, resize_w), (255, 255, 255))
        # mask_patch = image_255.crop(box)
        # patch = image_125.crop(box)
        mask_patch = image_255

      

        to_location_h = int(random.uniform(0, img_h - 50))
        to_location_w = int(random.uniform(0, img_w - 50))

        # resize_h = cut_w
        # resize_w = cut_h
        # resize_ratio = 0.8

        # resize_h = int(cut_h*resize_ratio)
        # resize_w = int(cut_w*resize_ratio)
        # patch = patch.resize((resize_w,resize_h))
        # mask_patch = mask_patch.resize((resize_w,resize_h))

        insert_box = [to_location_w, to_location_h, to_location_w + resize_w, to_location_h + resize_h]
        # target_mask = target_mask.copy()

        # target_mask = Image.fromarray(target_mask.astype('uint8')).convert('RGB')

        augmented.paste(patch, insert_box)
        # augmented.paste(patch, insert_box)
        mask.paste(mask_patch,insert_box)
        
    return augmented,mask

def dot_255(img):
  
    img.resize((256, 256))
    w = img.size[1]
    h = img.size[0]
    img = np.array(img)
    size = 10
    img[h//2:h//2+size,w//2:w//2+size]=(255,255,255)
    # img[h//2+size,w//2+size]=(255,255,255)
    # img[h//2+size,w//2+size]=(255,255,255)
    # img[h//2+size,w//2+size]=(255,255,255)

    img = Image.fromarray(img.astype('uint8'))
    return img

if __name__ == "__main__":
    path = "/root/allproject/mypy/DFR/DFR-source/checkpoints/screw/Dual network of train good test all/Results/gt_pred_score_map/gt_good-000.png"
    img = Image.open(path)
    mask = Image.new('RGB', (img.size[1], img.size[0]), (0, 0, 0))
    # img = dot_255(img)
    img_anormal = img
    img_anormal,mask = add_defect(img_anormal)
    # img_anormal,mask = add_defect(img_anormal)
    # cut_out = Cutout(10,20)
    # img,mask = cut_out(img)
    # img,mask = cut_defect_paste(img,mask)

    # randoma = RandomErasing()
    # img = randoma(img)
    cutpaste = CutPasteNormal()
    # mask = Image.new('RGB', (img.width, img.height), (0, 0, 0))
    # img_anormal,mask = cutpaste(img_anormal,mask)
    # img_anormal,mask = cutpaste(img_anormal,mask)

    # img_anormal,mask = cutpaste(img_anormal,mask)
    # img_anormal,mask = cutpaste(img_anormal,mask)
    # img_anormal,mask = cutpaste(img_anormal,mask)

    # img,mask = cutpaste(img,mask)

    # img,mask = cutpaste(img,mask)

    # img,mask = cutpaste(img,mask)

    # img,mask = cutpaste(img,mask)

    img_anormal.save('img10.png')    
    mask.save('mask.png')