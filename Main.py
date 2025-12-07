import json
import argparse
import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
import clip
import os
import logging
import matplotlib.pyplot as plt
import json

model_names = ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
parser = argparse.ArgumentParser(description='InMaP for ImageNet')
parser.add_argument('--data_path', default='/path/to/dataset', type=str,
                    help='dataset path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-L/14@336px',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: RN50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--iters_proxy', default=2000, type=int, metavar='N',
                    help='number of total iterations for learning vision proxy')
parser.add_argument('--iters_sinkhorn', default=20, type=int, metavar='N',
                    help='number of total iterations for optimizing Sinkhorn distance')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=10, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--tau_t', default=0.01, type=float)
parser.add_argument('--tau_i', default=0.04, type=float)
parser.add_argument('--alpha', default=0.6, type=float)
parser.add_argument('--gamma', default=0.0, type=float)
parser.add_argument('--k', default=8, type=int)
parser.add_argument('--type', default='cub', type=str,
                    help='dataset path')

def main():
    ###############
    logging.basicConfig(filename='New_image_classification_v27_medical_real.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')
    args = parser.parse_args()
    logging.info(f"Running with args: {args}")

    file_path = "./feature_50.json"
    # file_path = "./ImageNet_Real_Real.json"
    # Initialize a variable to hold the content of the file
    imagenet_templates = None

    # Try to open and read the file
    import json

    try:
        with open(file_path, "r") as file:
            imagenet_templates = json.load(file)
            # import pdb;pdb.set_trace()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file {file_path} could not be decoded.")
    if args.type == "eyes":
        imagenet_templates = imagenet_templates["Eyes_Features"]
    if args.type == "Lung":
        imagenet_templates = imagenet_templates["Lung_Features"]
    if args.type == "brain":
        imagenet_templates = imagenet_templates["Brain_Tumor_Features"]
    if args.type == "imagenet":
        imagenet_templates = imagenet_templates["ImageNet_Features"]
    if args.type == "cub":
        imagenet_templates = imagenet_templates["cub_Features"]
    if args.type == "foods":
        imagenet_templates = imagenet_templates["foods_Features"]
    if args.type == "pets":
        imagenet_templates = imagenet_templates["pets_Features"]    
    if args.type == "place":
        imagenet_templates = imagenet_templates["Place_Features"]
    if args.type == "cataract":
        imagenet_templates = imagenet_templates["Cataract_Features"]
    if args.type == "Cell":
        imagenet_templates = imagenet_templates["Cell_Features"]
    ###############
    
    imagenet_single_template = [
        'a photo of a {}.',
    ]

    imagenet_7_templates = [
        'itap of a {}.',
        'a origami {}.',
        'a bad photo of the {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]

    args = parser.parse_args()
    print(args)

    print('load pre-trained model')
    model, preprocess = clip.load(args.arch)
    model = model.cuda()
    model.eval()

    print('load data')
    valdir = os.path.join(args.data_path, 'val')
    val_set = datasets.ImageFolder(valdir, transform=preprocess)
    loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    class_to_idx = val_set.class_to_idx
    imagenet_classes = list(val_set.class_to_idx.keys())
    print("Class to index mapping:", class_to_idx)
    import json
    class_to_idx = val_set.class_to_idx
    with open('class_to_index.json', 'w') as json_file:
        json.dump(class_to_idx, json_file, indent=4)
    print("Class to index mapping saved to class_to_index.json")

    with torch.no_grad():
        image_feat = []
        image_label = []
        for i, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            image_feat.append(F.normalize(image_features, dim=1))
            image_label.append(target)
    image_feat = torch.cat(image_feat, dim=0)
    image_label = torch.cat(image_label, dim=0)
    n = len(image_label)
    image_feat = image_feat.float() # can keep fp16 for efficiency on GPU

    print('obtain text proxy')

    # text_classifier = zeroshot_classifier(clip, model, imagenet_classes, imagenet_single_template)
    k = args.k
    text_classifier, k_zeroshot_weights= zeroshot_classifier(clip, model, imagenet_classes, imagenet_templates, image_feat , k, imagenet_7_templates)
    text_classifier = text_classifier.float()
    logits_list = []

    # Iterate through each set of weights
    # for i in range(k_zeroshot_weights.size(0)):
    #     # Extract current weight component and ensure it is float type
    #     current_weights = k_zeroshot_weights[i].float()
    #     # Perform matrix multiplication to get logits
    #     logits_t = image_feat @ current_weights
    #     # Add calculated logits_t to the list
    #     logits_list.append(logits_t)

    # # Find the logits_t with maximum similarity from the list
    # # import pdb;pdb.set_trace()
    # logits_t = torch.max(torch.stack(logits_list), dim=0)[0]

    logits_t = image_feat @ text_classifier
    acc1, acc5 = accuracy(logits_t, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with text proxy: {top1:.2f}")
    logging.info(f"accuracy with text proxy: {top1:.2f}")

    print('obtain vision proxy without Sinkhorn distance')
    plabel = F.softmax(logits_t / args.tau_t, dim=1)
    image_classifier = image_opt(image_feat, text_classifier, plabel, args.lr, args.iters_proxy, args.tau_i, args.alpha)
    logits_i = image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with image proxy: {top1:.2f}")
    logging.info(f"accuracy with image proxy: {top1:.2f}")

    print('obtain refined labels by Sinkhorn distance')
    import time
    start_time = time.time()

    plabel = StableGreenkhorn(logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)
    # plabel = greenkhorn(logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)
    # plabel = sinkhorn(logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Function executed in {elapsed_time:.6f} seconds")

    print('obtain vision proxy with Sinkhorn distance')
    image_classifier = image_opt(image_feat, text_classifier, plabel, args.lr, args.iters_proxy, args.tau_i, args.alpha)
    logits_i = image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with image proxy + sinkhorn: {top1:.2f}")
    logging.info(f"accuracy with image proxy + sinkhorn: {top1:.2f}")


# def zeroshot_classifier(clip, model, classnames, templates):
#     with torch.no_grad():
#         zeroshot_weights = []
#         for classname in classnames:
#             texts = [template.format(classname) for template in templates]
#             # import pdb;pdb.set_trace()
#             texts = clip.tokenize(texts).cuda()
#             class_embeddings = model.encode_text(texts)
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             zeroshot_weights.append(class_embedding)
#         zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
#     return zeroshot_weights


import torch
import torch.nn.functional as F
import clip  # Ensure this is imported at the top of your script


def zeroshot_classifier(clip, model, classnames, templates, image_features, k, imagenet_7_templates):
    with torch.no_grad():
        zeroshot_weights = []
        k_zeroshot_weights = []
        image_feat = F.normalize(image_features, dim=-1)  # Ensure image features are normalized

        for classname in classnames:
            # texts = [feature["feature"] for feature in templates[classname]]
            # # import pdb;pdb.set_trace()
            # texts = [f"{classname} {text}" for text in texts]
            # texts = [template.format(classname) for template in imagenet_7_templates]
            # texts = [f"{classname}, which has {template['feature']}" for template in templates[classname]]

            texts = [f"a photo of a {classname}, which has {template['feature']}" for template in templates[classname]]

            # texts = [f"{classname} {template['feature']}" for template in templates[classname]]    #Medical
            # texts = [f"{classname}, which has {template['feature']}" for template in templates[classname]]
            # texts = [f"a photo of a {classname}."]
            # import pdb;pdb.set_trace()
            texts = clip.tokenize(texts).cuda()  # Assuming tokenize fits your model
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)

            # Compute cosine similarity
            # cos_sim = torch.matmul(image_feat, class_embeddings.T)
            cos_sim = torch.matmul(image_feat, class_embeddings.to(torch.float32).T)
            # Compute the average similarity between all images and each text feature
            mean_cos_sim = cos_sim.mean(dim=0)

            # Select the top k most similar text features based on average similarity
            # import pdb;pdb.set_trace()
            k = min(k, mean_cos_sim.size(0))
            top_k_values, top_k_indices = torch.topk(mean_cos_sim, k)
            
            # Get the embeddings of the top k most similar text features using the selected indices
            top_k_embeddings = class_embeddings[top_k_indices]
            
            # Average these k embeddings to get an embedding representing the category
            class_embedding = top_k_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=0)
            k_zeroshot_weights.append(top_k_embeddings)
            zeroshot_weights.append(class_embedding)
            # import pdb;pdb.set_trace()
        # Stack weights of all categories into a tensor
        # import pdb;pdb.set_trace()
        k_zeroshot_weights = torch.stack(k_zeroshot_weights, dim=2).cuda()
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        
    return zeroshot_weights, k_zeroshot_weights



def image_opt(feat, init_classifier, plabel, lr=10, iter=2000, tau_i=0.04, alpha=0.6):
    ins, dim = feat.shape
    val, idx = torch.max(plabel, dim=1)
    mask = val > alpha
    plabel[mask, :] = 0
    plabel[mask, idx[mask]] = 1
    base = feat.T @ plabel
    classifier = init_classifier.clone()
    pre_norm = float('inf')
    for i in range(0, iter):
        prob = F.softmax(feat @ classifier / tau_i, dim=1)
        grad = feat.T @ prob - base
        temp = torch.norm(grad)
        if temp > pre_norm:
            lr /= 2.
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad
        classifier = F.normalize(classifier, dim=0)
    return classifier

import torch
import torch.nn.functional as F

##### Version origin greenkhorn

def greenkhorn(M, tau_t=0.01, gamma=0, iter=50):
    row, col = M.shape
    P = F.softmax(M / tau_t, dim=1)
    if gamma > 0:
        q = torch.sum(P, dim=0) ** gamma
        q /= torch.sum(q)
    else:
        q = torch.ones(col, device=M.device) / col

    for _ in range(iter):
        # Update columns
        P /= torch.sum(P, dim=0, keepdim=True)
        P *= q
        
        # Update rows
        P /= torch.sum(P, dim=1, keepdim=True)
        P *= row

    return P

def optimized_greenkhorn_process(M, tau_t=0.01, gamma=0, iter_greenkhorn=50):
    # Using Greenkhorn for optimization
    P = greenkhorn(M, tau_t, gamma, iter_greenkhorn)
    return P

###### Version origin sinkhorn

def sinkhorn(M, tau_t=0.01, gamma=0, iter=20):
    row, col = M.shape
    P = F.softmax(M / tau_t, dim=1)
    P /= row
    if gamma > 0:
        q = torch.sum(P, dim=0, keepdim=True)
        q = q**gamma
        q /= torch.sum(q)
    for it in range(0, iter):
        # total weight per column must be 1/col or q_j
        P /= torch.sum(P, dim=0, keepdim=True)
        if gamma > 0:
            P *= q
        else:
            P /= col
        # total weight per row must be 1/row
        P /= torch.sum(P, dim=1, keepdim=True)
        P /= row
    P *= row  # keep each row sum to 1 as the pseudo label
    return P



###### Version stable Greenkhorn

def StableGreenkhorn(M, tau_t=0.01, gamma=0, iter=20):
    row, col = M.shape
    log_P = M / tau_t
    # Apply more stable softmax to each row to initialize log_P
    log_P -= torch.logsumexp(log_P, dim=1, keepdim=True)

    if gamma > 0:
        q = torch.exp(torch.logsumexp(log_P, dim=0, keepdim=True))
        q = q**gamma
        q /= q.sum()
    else:
        q = torch.ones(col, dtype=log_P.dtype, device=log_P.device) / col

    r = torch.ones(row, dtype=log_P.dtype, device=log_P.device) / row

    for _ in range(iter):
        # Compute violation of rows and columns. Greenkhorn algorithm is an improvement on Sinkhorn algorithm. It uses a greedy strategy, updating only the row or column furthest from its target marginal distribution in each iteration. This strategy reduces the number of iterations required for convergence, thereby improving computational efficiency.
        col_violation = torch.abs(torch.sum(torch.exp(log_P), dim=0) - q)
        row_violation = torch.abs(torch.sum(torch.exp(log_P), dim=1) - r)

        # Select the row or column with the largest violation for update
        if torch.max(col_violation) > torch.max(row_violation):
            # Update columns
            j = torch.argmax(col_violation)
            log_P[:, j] -= torch.logsumexp(log_P[:, j], dim=0, keepdim=True)
            log_P[:, j] += torch.log(q[j])
        else:
            # Update rows
            i = torch.argmax(row_violation)
            log_P[i, :] -= torch.logsumexp(log_P[i, :], dim=0, keepdim=True)
            log_P[i, :] += torch.log(r[i])

    return torch.exp(log_P)

###### Version Multiscale Sinkhorn
# import torch
# import torch.nn.functional as F

# def downsample_matrix(M, factor):
#     """Reduce matrix resolution by average pooling"""
#     size = (M.size(0) // factor, M.size(1) // factor)
#     pool = torch.nn.AvgPool2d(kernel_size=factor, stride=factor, count_include_pad=False)
#     M_downsampled = pool(M.unsqueeze(0).unsqueeze(0)).squeeze(0)
#     return M_downsampled.squeeze(0)

# def upsample_matrix(P, original_shape):
#     """Increase matrix resolution by interpolation"""
#     return F.interpolate(P.unsqueeze(0).unsqueeze(0), size=original_shape, mode='nearest').squeeze(0)

# def coarse_to_fine_sinkhorn(M, tau_t=0.01, gamma=0, iter=20, downscale_factor=4):
#     # Step 1: Process at a coarser resolution
#     coarse_M = downsample_matrix(M, downscale_factor)
#     P = sinkhorn(coarse_M, tau_t, gamma, iter)

#     # Step 2: Upsample the solution to the original resolution
#     fine_M = M
#     P = upsample_matrix(P, fine_M.shape)
    
#     # Step 3: Refine the solution at the original resolution
#     P = sinkhorn(fine_M, tau_t, gamma, iter)
#     return P

# def sinkhorn(M, tau_t=0.01, gamma=0, iter=20):
#     row, col = M.shape
#     P = F.softmax(M / tau_t, dim=1)
#     P /= row

#     if gamma > 0:
#         q = torch.sum(P, dim=0, keepdim=True) ** gamma
#         q /= torch.sum(q)

#     for _ in range(iter):
#         P /= torch.sum(P, dim=0, keepdim=True)
#         if gamma > 0:
#             P *= q
#         else:
#             P /= col
#         P /= torch.sum(P, dim=1, keepdim=True)
#         P *= row

#     return P

###### Version Greenkhorn+Sinkhorn
# import torch
# import torch.nn.functional as F

# def greenkhorn(M, tau_t=0.01, gamma=0, iter=20):
#     row, col = M.shape
#     P = F.softmax(M / tau_t, dim=1)
    
#     if gamma > 0:
#         q = torch.sum(P, dim=0) ** gamma
#         q /= torch.sum(q)
#     else:
#         q = torch.ones(col, device=M.device) / col

#     for _ in range(iter):
#         # Update columns
#         P /= torch.sum(P, dim=0, keepdim=True)
#         P *= q
        
#         # Update rows
#         P /= torch.sum(P, dim=1, keepdim=True)
#         P *= row

#     return P, q

# def sinkhorn(M, tau_t=0.01, gamma=0, iter=20, initial_P=None, q=None):
#     row, col = M.shape
#     if initial_P is None:
#         P = F.softmax(M / tau_t, dim=1)
#     else:
#         P = initial_P

#     for _ in range(iter):
#         # Normalize columns using q if provided, otherwise use uniform distribution
#         P /= torch.sum(P, dim=0, keepdim=True)
#         if q is not None and gamma > 0:
#             P *= q
#         else:
#             P /= col
        
#         # Normalize rows
#         P /= torch.sum(P, dim=1, keepdim=True)
#         P *= row

#     return P

# def optimized_sinkhorn_process(M, tau_t=0.01, gamma=0, iter_greenkhorn=10, iter_sinkhorn=10):
#     # First use Greenkhorn for a coarse optimization
#     P, q = greenkhorn(M, tau_t, gamma, iter_greenkhorn)
    
#     # Then refine the result with Sinkhorn, passing the intermediate matrix and q
#     P = sinkhorn(M, tau_t, gamma, iter_sinkhorn, initial_P=P, q=q)
#     return P









def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == '__main__':
    main()
