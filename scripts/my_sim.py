import argparse
import os, json
from typing import Tuple

import torch
import torch.distributions as dists
from torchmetrics.classification import MulticlassCalibrationError

import sys
sys.path.append('..') # add bayesvlm to path
sys.path.append('.') # add bayesvlm to path

from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.hessians import load_hessians, compute_covariances, optimize_prior_precision
from bayesvlm.precompute import precompute_image_features, precompute_text_features, my_make_predictions
from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.utils import get_model_type_and_size, get_image_size, get_transform, load_model


# importing collate function used by other datasets
from bayesvlm.data.common import default_collate_fn

SUPPORTED_DATASETS = ["flowers102", "food101", "stanfordcars", "eurosat", "cifar100", "dtd"]

    
def evaluate_prediction(prediction: torch.Tensor, label: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l1')
    one_hot_pred = prediction.argmax(1)
    acc = (one_hot_pred == label).float().cpu().numpy()
    nlpd = -dists.Categorical(prediction).log_prob(label).cpu().numpy()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece

def main(
    dataset: str,
    hessian_dir: str,
    model_str: str = "clip-base",
    pseudo_data_count: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
):
    if model_str not in MODEL_NAME_MAP:
        raise ValueError(f"Invalid model name: {model_str}, must be one of {MODEL_NAME_MAP.keys()}")
    
    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    image_encoder, text_encoder, vlm = load_model(model_str, device)

    A_img, B_img = load_hessians(hessian_dir, tag='img', return_info=False)
    A_txt, B_txt = load_hessians(hessian_dir, tag='txt', return_info=False)


    print(f"A and B-> A_img: {A_img.shape} and B_img: {B_img.shape}")
    # [1024, 1024] and [768, 768]

    info = {
        'n_img': pseudo_data_count,
        'n_txt': pseudo_data_count,
    }


    print("[0] creating custome data loaders...")
    import requests
    from PIL import Image
    from io import BytesIO
    from torch.utils.data import Dataset

    class ImageURLDataset(Dataset):
        def __init__(self, image_urls, image_ids, class_ids, image_captions, transform=None):
            self.image_urls = image_urls
            self.transform = transform
            self.image_ids = image_ids
            self.class_ids = class_ids
            self.image_captions = image_captions

        def __len__(self):
            return len(self.image_urls)

        def __getitem__(self, idx):
            response = requests.get(self.image_urls[idx])
            img = Image.open(BytesIO(response.content)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return {
                "image": torch.Tensor(img),
                "text": self.image_captions[idx],
                "image_id": self.image_ids[idx],
                "class_id": self.class_ids[idx]
            }

    class TextDataset(Dataset):
        def __init__(self, image_captions, transform=None):
            self.image_captions = image_captions
            self.transform = transform

        def __len__(self):
            return len(self.image_captions)

        def __getitem__(self, idx):
            return self.image_captions[idx]

    images_links = [
        "https://farm4.staticflickr.com/3070/3059356111_1fd852b68a_z.jpg",
        "http://farm4.staticflickr.com/3037/2645140983_8f719ce12e_z.jpg",
        "http://farm7.staticflickr.com/6155/6179447413_d60cf99f28_z.jpg"
    ]
    image_ids = [0, 1, 2]
    
    class_ids = image_ids

    images_captions = [
        " a photo of cat",
        " a photo of motor cycle",
        " a photo of skateboard",
    ]

    # creating custom dataloader for these images
    batch_size = 1
    num_workers = 0
    dataset = ImageURLDataset(images_links, image_ids, class_ids, images_captions, transform=transform)
    my_image_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn= default_collate_fn
    )

    text_dataset = TextDataset(images_captions, transform=transform)
    my_text_loader = torch.utils.data.DataLoader(
        dataset= text_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    

    print("[1] Optimizing prior precision...")
    # if prior_precision_info.json file exists then just load the values
    # else optimize the prior precision
    if os.path.exists(os.path.join(hessian_dir, 'prior_precision_analytic.json')):
        with open(os.path.join(hessian_dir, 'prior_precision_analytic.json')) as f:
            print("computed hi mil gya!")
            temp = json.load(f)
            info['lambda_img'] = temp['lambda_img']
            info['lambda_txt'] = temp['lambda_txt']
    else:
        info['lambda_img'] = optimize_prior_precision(
            image_encoder.vision_projection,
            A=A_img,
            B=B_img,
            lmbda_init=300,
            n=info['n_img'],
            lr=1e-2,
            num_steps=1000,
            device=device,
            verbose=False,
        ).item()

        info['lambda_txt'] = optimize_prior_precision(
            text_encoder.text_projection,
            A=A_txt,
            B=B_txt,
            lmbda_init=300,
            n=info['n_txt'],
            lr=1e-2,
            num_steps=1000,
            device=device,
            verbose=False,
        ).item()
    print("\tn_img:", info['n_img'])
    print("\tn_txt:", info['n_txt'])
    print("\tlambda_img:", info['lambda_img'])
    print("\tlambda_txt:", info['lambda_txt'])

    cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, info)
    vlm.set_covariances(cov_img, cov_txt)

    print("[2] Precomputing features...")
    # this is where the work on data started for first time
    with torch.no_grad():
        image_outputs_test, image_class_ids_test, image_ids_test = precompute_image_features(
            image_encoder=image_encoder,
            loader= my_image_loader,
        )

        label_outputs = precompute_text_features(
            text_encoder=text_encoder,
            class_prompts= images_captions,
            batch_size=batch_size,
        )
        
        print("[VERBOSE]: Both image and label outputs generated successfully!")

        print("[3] Making predictions...")
        prob_logits_test = my_make_predictions(
            clip=vlm,
            image_outputs=image_outputs_test,
            text_outputs=label_outputs,
            batch_size=batch_size,
            device=device,
        )

        print(f"Means: {prob_logits_test.mean}")
        print(f"Vars: {prob_logits_test.var}")

    # probit approximation
    kappa = 1 / torch.sqrt(1. + torch.pi / 8 * prob_logits_test.var)
    pred = torch.softmax(kappa * prob_logits_test.mean, dim=-1)

    print(f"pred: {pred}")

    print("[4] Evaluate model ...")
    acc, nlpd, ece = evaluate_prediction(pred, image_class_ids_test, num_classes=len(images_captions))

    print(f"Zero shot CLIP on {dataset}")
    print(f"ACC: {acc.mean()}, {acc.std()}")
    print(f"NLPD: {nlpd.mean()}, {nlpd.std()}")
    print(f"ECE: {ece}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='food101', help="The dataset to use")
    parser.add_argument("--hessian_dir", type=str, default='hessians/hessian_CLIP-ViT-L-14-laion2B-s32B-b82K', help="The directory containing the hessian files")
    parser.add_argument("--model", type=str, default="clip-large")
    parser.add_argument("--pseudo_data_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,
        model_str=args.model,
        pseudo_data_count=args.pseudo_data_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )