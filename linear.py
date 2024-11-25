from utils.logger import create_logger

import time
import torch
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from evaluation import evaluate_multiclass
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from torch.utils.data import TensorDataset
import json
import pandas as pd

from utils.common import get_train_paths, load_config
from utils.config import parse_data_str
from utils.data.dataset import BaseData
from utils.evaluation import metric_ood, compute_oscr
from utils.feature_extraction import extract_features
from models.logreg import LogRegModule


def parse_args():
    parser = argparse.ArgumentParser(description='Logistic Regression experiment')
    parser.add_argument('-c','--config_name', type=str, help='model configuration file')
    parser.add_argument('-d','--data', type=str, dest="dataset_str",)
    parser.add_argument('--device', type=str, help='cuda:n or cpu')
    parser.add_argument('-o','--output_dir', type=str, default='output')
    parser.add_argument('--max_train_iters', type=int)
    parser.add_argument('-b','--blocks', nargs='+', type=int, help='blocks to extract features from')
    parser.add_argument('--backbone', type=str, )
    parser.add_argument('--c_min', type=int, help='range of C values to sweep')
    parser.add_argument('--c_max', type=int, help='range of C values to sweep')
    parser.add_argument('--c_step', type=int, help='range of C values to sweep')
    parser.add_argument('-c', '--C', type=float, help='C value for linear classifier')


    parser.set_defaults(
        config_name='base',
        dataset_str='GenImage:split=split1',
        device='cuda:0',
        output_dir='output/linear-probe/',
        train_backbone=False,
        debug=False,
        max_train_iters=1000,
        blocks=None,
        backbone='vit_base_patch16_clip_224.openai'
    )

    args = parser.parse_args()
    return args


def sweep_C_values(
        train_features,
        train_labels,
        test_data_loader,
        out_data_loader,
        max_train_iters,
        results_path="",
        metric="AUC",
        c_min=-6,
        c_max=5,
        c_step=45,
):
    best_stats = None
    best_C = None

    C_POWER_RANGE = np.arange(c_min, c_max, c_step)
    ALL_C = 10 ** C_POWER_RANGE
    all_stats = {}

    for C in ALL_C:
        C = C.item()
        logger.info(f"Training with C={C}")
        model = LogRegModule(C, max_iter=max_train_iters, device=opt.device)
        model.fit(train_features, train_labels)
        stats = evaluate_model(
            model=model,
            test_data_loader=test_data_loader,
            out_data_loader=out_data_loader,
        )
        all_stats[C] = stats
        if best_stats is None or stats[metric] > best_stats[metric]:
            best_stats = stats
            best_C = C
    # Specify the filename

    # Append JSON data to file
    with open(results_path, 'a') as f:
        f.write(json.dumps(all_stats))
        f.write('\n')
    return best_stats, best_C


def predict_set(
        model,
        data_loader,
        run_type="test",
):
    all_preds, all_targets = [], []
    for data, targets in data_loader:
        with torch.no_grad():
            outputs = model(data, targets)
            all_preds.append(outputs["preds"])
            all_targets.append(outputs["target"])
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    if 'open-set' in run_type:
        return all_targets.numpy(), all_preds.numpy()
    else:
        results = evaluate_multiclass(all_targets, all_preds.argmax(dim=1))
        CM = confusion_matrix(all_targets, all_preds.argmax(dim=1))
        perf = round(results['accuracy'], 4) * 100
        logger.info('%s results: %s' % (run_type, str(results)))
        logger.info('%s confusion matrix: %s' % (run_type, str(CM)))
        return all_targets.numpy(), all_preds.numpy(), perf


def evaluate_model(
        model,
        test_data_loader,
        out_data_loader,
        config
):
    unknown_classes = config.unknown_classes['all']
    in_targets, in_preds, closed_results = predict_set(model, test_data_loader, run_type='val')
    #
    out_targets, out_preds = predict_set(model, out_data_loader, run_type='open-set')
    x1, x2 = np.max(in_preds, axis=1), np.max(out_preds, axis=1)
    out_results = metric_ood(x1, x2)
    oscr_score = compute_oscr(in_preds, out_preds, in_targets)
    logger.info('OSCR: %.4f' % oscr_score)
    out_result_details = {}
    for i, label_u in enumerate(out_targets):
        pred_u = out_preds[out_targets==label_u]
        x1, x2 = np.max(in_preds, axis=1), np.max(pred_u, axis=1)
        pred = np.argmax(pred_u, axis=1)
        pred_labels = list(set(pred))
        pred_nums = [np.sum(pred==p) for p in pred_labels]
        result = metric_ood(x1, x2, verbose=False)['Bas']
        logger.info("{}\t \t mostly pred class: {}\t \t average score: {}\t AUROC (%): {:.2f}".format(unknown_classes[i], 
                                                                                    config.known_classes[pred_labels[np.argmax(pred_nums)]],
                                                                                    np.mean(x2), result['AUROC']))
        out_result_details[str(i)] = {'unknown_class':'\t'+ unknown_classes[i],
                                        'pred_class': '\t'+ config.known_classes[pred_labels[np.argmax(pred_nums)]],
                                        'average_score':'\t'+ str(round(np.mean(x2),4)), 
                                        'AUROC':'\t'+ str(round(result['AUROC'],2))}
    
    return {'AUC': out_results['Bas']['AUROC'], 'OSCR': oscr_score * 100, 'accuracy': closed_results, 'out_results_details': out_result_details}
    

if __name__ == '__main__':
    opt = parse_args()

    # Load configuration
    config = load_config('configs.{}'.format(opt.config_name))
    run_dir = os.path.join(opt.output_dir, f'{opt.backbone}')
    os.makedirs(run_dir, exist_ok=True)
    logger = create_logger(run_dir, log_name='train.log')
    logger.info(f'logging to {run_dir}')
    # Arguments
    max_train_iters = opt.max_train_iters
    c = opt.C

    # Create model
    model = timm.create_model(opt.backbone, pretrained=True, num_classes=0, global_pool='')
    model = model.to(opt.device)
    model.eval()
    model_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**model_cfg, is_training=False)
    if 'input_size' in model_cfg:
        input_size = model_cfg['input_size']
    else:
        input_size = (3, 224, 224)
    logger.debug(f"Input size: {input_size}")
    _, sample_out = model.forward_intermediates(torch.randn(1, *input_size).to(opt.device))
    logger.debug(f"Model output shape: {sample_out[0].shape}")
    num_blocks = len(sample_out)
    logger.debug(f"Model has {num_blocks} blocks")
    if opt.blocks is None:
        blocks = list(range(num_blocks))
    else:
        assert all([block < num_blocks for block in opt.blocks])
        blocks = opt.blocks

    # Load data configuration
    data_list = parse_data_str(opt.dataset_str)
    train_data_path, val_data_path = get_train_paths(data_list)
    test_data_path, out_data_paths = data_list['test_data_path'], data_list['out_data_paths']
    # Set configuration variables for the dataset
    config.known_classes = data_list['known_classes']
    config.unknown_classes = data_list['unknown_classes']
    config.class_num = len(config.known_classes)

    batch_size = config.batch_size
    logger.debug('config.class_num', config.class_num)
    Data = BaseData(train_data_path, val_data_path,
                    test_data_path, out_data_paths,
                    opt, config, transform)
    
    tokens = opt.dataset_str.split(":")
    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value
    filename = os.path.join(run_dir, f'{name}_{kwargs["split"]}.json')
    if os.path.exists(filename):
        os.remove(filename)

    start = time.time()

    for block in blocks:
        # features
        train_features, train_labels = extract_features(Data.train_loader, model, block, device=opt.device)
        val_features, val_labels = extract_features(Data.val_loader, model, block, device=opt.device)
        out_features, out_labels = extract_features(Data.out_loaders['all'], model, block, device=opt.device)

        logger.debug(f"Features shape: {train_features.shape}")

        train_data_loader = torch.utils.data.DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        val_data_loader = torch.utils.data.DataLoader(
            TensorDataset(val_features, val_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        out_data_loader = torch.utils.data.DataLoader(
            TensorDataset(out_features, out_labels),
            batch_size=batch_size,
            drop_last=False,
        )

        if len(train_labels.shape) > 1:
            num_classes = train_labels.shape[1]
        else:
            num_classes = train_labels.max() + 1

        if c is None:
            best_stats, best_C = sweep_C_values(
                train_features,
                train_labels,
                val_data_loader,
                out_data_loader,
                max_train_iters,
                results_path=filename,
                c_min=opt.c_min,
                c_max=opt.c_max,
                c_step=opt.c_step,
            )
            c = best_C
        out_features, out_labels = extract_features(Data.out_loaders['test_all'], model, block, device=opt.device)
        test_features, test_labels = extract_features(Data.test_loader, model, block, device=opt.device)
        out_data_loader = torch.utils.data.DataLoader(
            TensorDataset(out_features, out_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        test_data_loader = torch.utils.data.DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        model = LogRegModule(c, max_iter=max_train_iters, device=opt.device)
        model.fit(train_features, train_labels)
        stats = evaluate_model(
            model=model,
            test_data_loader=test_data_loader,
            out_data_loader=out_data_loader,
            config=config 
        )
        # save detailed OSR results
        df = pd.DataFrame(stats['out_result_details'])    
        data = df.values
        data = list(map(list,zip(*data)))
        data = pd.DataFrame(data)
        data.to_csv(os.path.join(run_dir, 'block-{}_{}_result_details.csv'.format(block)), header = 0)
        logger.info(f"Block {block} results: {stats}")


