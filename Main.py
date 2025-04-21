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
import numpy as np
import time


model_names = ['ViT-L/14', 'ViT-L/14@336px']
parser = argparse.ArgumentParser(description='KPL')
parser.add_argument('--data_path', default='/path/CUB/CUB_200_2011', type=str,
                    help='dataset path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-L/14',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: RN50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--iters_ot', default=20, type=int, metavar='N',
                    help='number of total iterations')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=10, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--tau_t', default=0.01, type=float)
parser.add_argument('--type', default='', type=str,
                    help='dataset type')

def main():
    start_time1 = time.time()
    logging.basicConfig(filename='New_image_classification_v35_time.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')
    args = parser.parse_args()
    logging.info(f"Running with args: {args}")

    file_path = "/path/feature_50.json"
    imagenet_templates = None
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
    else:
        imagenet_templates = imagenet_templates

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

    end_time6 = time.time()
    elapsed_time6 = end_time6 - start_time1
    logging.info(f"part6 in {elapsed_time6:.6f} seconds")  

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
    start_time2 = time.time()
    k = args.k
    text_classifier, k_zeroshot_weights= zeroshot_classifier(clip, model, imagenet_classes, imagenet_templates, image_feat , k)
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    logging.info(f"KE module Function executed in {elapsed_time2:.6f} seconds")  
    text_classifier = text_classifier.float()
    logits_list = []
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    logging.info(f"part1 in {elapsed_time1:.6f} seconds")  

    logits_t = image_feat @ text_classifier
    acc1, acc5 = accuracy(logits_t, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with text proxy: {top1:.2f}")
    logging.info(f"accuracy with text proxy: {top1:.2f}")

    print('obtain vision proxy without Sinkhorn distance')
    plabel = F.softmax(logits_t / args.tau_t, dim=1)
    image_classifier = image_opt(image_feat, text_classifier, plabel, args.lr, 2000, args.tau_i, args.alpha)
    logits_i = image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with image proxy: {top1:.2f}")
    logging.info(f"accuracy with image proxy: {top1:.2f}")
    # import pdb;pdb.set_trace()
# Convert tensors to numpy arrays if they are not already
    image_feat_numpy = logits_t.cpu().numpy()

    print('obtain refined labels by Sinkhorn distance')
    
    start_time3 = time.time()
    
    plabel = StableGreenkhorn(logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)
    # plabel = sinkhorn(logits_t, args.tau_t, args.gamma, args.iters_sinkhorn)


    print('obtain vision proxy with Sinkhorn distance')
    image_classifier = image_opt(image_feat, text_classifier, plabel, args.lr, args.iters_proxy, args.tau_i, args.alpha)
    logits_i = image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, image_label, topk=(1, 2))
    top1 = (acc1 / n) * 100
    print(f"accuracy with image proxy + sinkhorn: {top1:.2f}")
    logging.info(f"accuracy with image proxy + sinkhorn: {top1:.2f}")
    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
    logging.info(f"Function executed in {elapsed_time3:.6f} seconds")

    image_classifier_sinkhorn_numpy = logits_i.cpu().numpy()
    # labels = image_label.cpu().numpy()
    # colors = ['#D45361', '#F2A584', '#91C5DA',  '#B1182D', '#FDDAC4']
    # # colors = ['#D45361', '#F2A584', '#91C5DA',  '#B1182D', '#FDDAC4']
    # # colors = ['#FFFFFF', '#c4dfa2', '#FFFFFF', '#FFFFFF', '#f9e9ab']
    # class_names = ["mild nonproliferative retinopathy", "moderate nonproliferative retinopathy", "no apparent retinopathy",  "proliferative retinopathy", "severe nonproliferative retinopathy"]
    # from sklearn.decomposition import PCA
    # from sklearn.preprocessing import StandardScaler
    #
    # scaler = StandardScaler()
    # image_feat_original_scaled = scaler.fit_transform(image_feat_numpy)
    # image_feat_sinkhorn_scaled = scaler.fit_transform(image_classifier_sinkhorn_numpy)
    # pca = PCA(n_components=2)
    # pca_results_original = pca.fit_transform(image_feat_original_scaled)
    # pca_results_sinkhorn = pca.fit_transform(image_feat_sinkhorn_scaled)
    #
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # for i, color in enumerate(colors):
    #     mask = core_labels_original == i
    #     plt.scatter(core_pca_results_original[mask, 0], core_pca_results_original[mask, 1], c=color, label=class_names[i])
    # plt.title('PCA of Original Features (90% core points)')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # for i, color in enumerate(colors):
    #     mask = core_labels_sinkhorn == i
    #     plt.scatter(core_pca_results_sinkhorn[mask, 0], core_pca_results_sinkhorn[mask, 1], c=color, label=class_names[i])
    # plt.title('PCA of Sinkhorn Optimized Features (90% core points)')
    # plt.legend()
    # svg_filename = '/path/PCA_comparison_0503.svg'
    # plt.savefig(svg_filename, format='svg')
    # plt.show()
def zeroshot_classifier(clip, model, classnames, templates, image_features, k):
    with torch.no_grad():
        zeroshot_weights = []
        k_zeroshot_weights = []
        time_consumptions = []
        image_feat = F.normalize(image_features, dim=-1)

        for classname in classnames:
            texts = [f"{classname}, which has {template['feature']}" for template in templates[classname]]
            # texts = [f"{classname} {template['feature']}" for template in templates[classname]]    #Medical
            # texts = [f"a photo of a {classname}."]
            texts_copy = texts
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            start_time = time.time()
            cos_sim = torch.matmul(image_feat, class_embeddings.to(torch.float32).T)
            mean_cos_sim = cos_sim.mean(dim=0)
            k = min(k, mean_cos_sim.size(0))
            top_k_values, top_k_indices = torch.topk(mean_cos_sim, k)

            top_k_embeddings = class_embeddings[top_k_indices]
            class_embedding = top_k_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=0)
            end_time = time.time()
            time_consumed = end_time - start_time
            time_consumptions.append(time_consumed)
            print(f"Iteration with class {classname} took {time_consumed:.4f} seconds")
            k_zeroshot_weights.append(top_k_embeddings)
            zeroshot_weights.append(class_embedding)

        k_zeroshot_weights = torch.stack(k_zeroshot_weights, dim=2).cuda()
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        total_time_consumed = sum(time_consumptions)
        print(f"Total time consumed for the loop is {total_time_consumed:.4f} seconds")
        
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

def sinkhorn(M, tau_t=0.01, iter=20):
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

def StableGreenkhorn(M, tau_t=0.01, iter=20):
    row, col = M.shape
    log_P = M / tau_t
    log_P -= torch.logsumexp(log_P, dim=1, keepdim=True)
    q = torch.ones(col, dtype=log_P.dtype, device=log_P.device) / col
    r = torch.ones(row, dtype=log_P.dtype, device=log_P.device) / row

    for _ in range(iter):
        col_violation = torch.abs(torch.sum(torch.exp(log_P), dim=0) - q)
        row_violation = torch.abs(torch.sum(torch.exp(log_P), dim=1) - r)

        if torch.max(col_violation) > torch.max(row_violation):
            j = torch.argmax(col_violation)
            log_P[:, j] -= torch.logsumexp(log_P[:, j], dim=0, keepdim=True)
            log_P[:, j] += torch.log(q[j])
        else:
            i = torch.argmax(row_violation)
            log_P[i, :] -= torch.logsumexp(log_P[i, :], dim=0, keepdim=True)
            log_P[i, :] += torch.log(r[i])

    return torch.exp(log_P)

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


if __name__ == '__main__':
    main()
