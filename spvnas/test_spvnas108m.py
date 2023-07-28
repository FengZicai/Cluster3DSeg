import argparse
import sys
import os
import numpy as np

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data

from torchpack import distributed as dist
from torchpack.callbacks import Callbacks, SaverRestore
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tqdm import tqdm
import json

from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='spvnas/configs/semantic_kitti/spvnas/default.yaml', help='config file')
    parser.add_argument('--run-dir', default='.../You/downloaded/model file/spvnas108m', help='run directory')
    parser.add_argument('--name', type=str, help='model name')
    args, opts = parser.parse_known_args()
    opts = ['--run-dir','.../You/downloaded/model file/spvnas108m','--name','spvnas']

    output_path = ""

    configs.load(args.config, recursive=True)
    configs.update(opts)

    gpu = 0
    pytorch_device = torch.device('cuda:' + str(gpu))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset()
    dataflow = {}

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset['test'],
        num_replicas=dist.size(),
        rank=dist.rank(),
        shuffle=False)

    dataflow['test'] = torch.utils.data.DataLoader(
        dataset['test'],
        batch_size=1,
        sampler=sampler,
        num_workers=configs.workers_per_gpu,
        pin_memory=True,
        collate_fn=dataset['test'].collate_fn)

    from core.models.semantic_kitti import SPVNAS
    net_config = json.load(open('spvnas/configs/semantic_kitti/spvnas/net.config'))

    model = SPVNAS(
        net_config['num_classes'],
        macro_depth_constraint=1,
        pres=net_config['pres'],
        vres=net_config['vres']).to(torch.device('cuda:0'))

    model.manual_select(net_config)
    model = model.determinize()

    my_model_dict = model.state_dict()
    pre_weight = torch.load(".../You/downloaded/model file/spvnas108m/checkpoints/step-87015.pt", map_location='cuda:'+ str(gpu))['model']
    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if 'module' in k:
            k = k[7:] # remove module.
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        else:
            nomatch_size += 1
    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))
    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)
    model = model.to(pytorch_device)

    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=configs.train.seed)
    callbacks = Callbacks([
        MeanIoU(configs.data.num_classes, configs.data.ignore_label)
    ])
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']

    trainer.before_train()
    trainer.before_epoch()

    model.eval()

    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key:
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']
        targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        outputs = model(inputs)


        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        for idx in range(invs.C[:, -1].max() + 1):
            cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)
        output_dict = {'outputs': outputs, 'targets': targets}
        trainer.after_step(output_dict)

        predict_labels = outputs.type(torch.uint8)
        predict_labels = predict_labels.cpu().detach().numpy()
        test_pred_label = train2SemKITTI(predict_labels)
        test_pred_label = test_pred_label[np.newaxis,np.newaxis, :]
        _,dir2 = feed_dict['file_name'][0].split('/sequences/',1)
        new_save_dir = './out/spvnas108m/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
        if not os.path.exists(os.path.dirname(new_save_dir)):
            try:
                os.makedirs(os.path.dirname(new_save_dir))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        test_pred_label = test_pred_label.astype(np.uint32)
        test_pred_label.tofile(new_save_dir)

    trainer.after_epoch()

    print('Predicted test labels are saved. Need to be shifted to original label format before submitting to the Competition website.')
    print('Remapping script can be found in semantic-kitti-api.')

if __name__ == '__main__':
    main()
