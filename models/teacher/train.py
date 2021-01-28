import os
import torch
import time
import json
import torch.nn.functional as F
from tqdm import tqdm
from pandas import DataFrame

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.modules import DataParallel

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy, get_dataset
from global_utils import Averager, Mean_confidence_interval, get_outputs_c_h

def save_model(model, args, tag):
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data,args.backbone,tag]) + '.pt'))
    if args.multi_gpu:
        model = model.module
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

def save_checkpoint(args, model, train_log, optimizer, all_task_accout, tag):
    checkpoint_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, tag]) + '_checkpoint.pt.tar'))
    if args.multi_gpu:
        model = model.module
    state = {
        'args': args,
        'model': model.state_dict(),
        'train_log': train_log,
        'val_acc': train_log['max_acc'],
        'optimizer': optimizer.state_dict(),
        'all_task_accout': all_task_accout
    }
    with open(checkpoint_path, 'wb') as f:
        torch.save(state, f)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser('Knowledge Distillation for Model-Agnostic Meta-Learning')
    parser.add_argument('--model-name', type=str, default='teacher', help='Name of the model.')

    parser.add_argument('--data-folder', type=str, default='../../dataset',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--train-data', type=str, default= 'cub', choices=['cub', 'miniimagenet',
        'omniglot'], help='Name of the dataset.')
    parser.add_argument('--test-data', type=str, default= 'cub', choices=['cub', 'miniimagenet',
        'omniglot'], help='Name of the dataset.')
    parser.add_argument('--num-shots', type=int, default= 1, 
        choices=[1, 3, 5, 7, 9],
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default= 5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--test-shots', type=int, default= 15,
        help='Number of examples per class (k in "k-shot", default: 15).')
    parser.add_argument('--backbone', type=str, default='conv4', 
        choices=['conv4','conv6','conv8','resnet10','resnet18'], 
        help='The type of model backbone.')
    
    parser.add_argument('--batch-tasks', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    # # follows Closer_Look, 40000 for 5 shots and 60000 for 1 shot
    parser.add_argument('--train-tasks', type=int, default= 60000, 
        help='Number of tasks the model is trained over (default: 60000).')
    parser.add_argument('--val-tasks', type=int, default=600,
        help='Number of tasks the model network is validated over (default: 600). ')
    parser.add_argument('--test-tasks', type=int, default=800,
        help='Number of tasks the model network is tested over (default: 800). The final results will be the average of these batches.')
    parser.add_argument('--validation-tasks', type=int, default=1000,
        help='Number of tasks for each validation (default: 1000).')
 
    # follows DPGN, decay lr by 0.1 each 15000 tasks 
    parser.add_argument('--lr', type=float, default=0.001,
        help='Initial learning rate (default: 0.001).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15000, 30000, 45000, 60000], 
        help='Decrease learning rate at these number of tasks.')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='Learning rate decreasing ratio (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
   
    parser.add_argument('--augment', action='store_true', 
        help='Augment the training dataset (default: True).')
    parser.add_argument('--pretrain', action='store_true',
        help='If backobone network is pretrained.')
    parser.add_argument('--backbone-path', type=str, default=None,
        help='Path to the pretrained backbone.')
    
    parser.add_argument('--multi-gpu', action='store_true',
        help='True if use multiple GPUs. Else, use single GPU.')
    parser.add_argument('--num-workers', type=int, default=4,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', type=bool, default=True,
        help='Use CUDA if available.')
   
    parser.add_argument('--resume', action='store_true', 
        help='Continue from baseline trained model with largest epoch.')
    parser.add_argument('--resume-folder', type=str, default=None,
        help='Path to the folder the resume is saved to.')

    args = parser.parse_args()

    # make folder and tensorboard writer to save model and results
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.record_folder = './{}_{}_{}_{}-ways_{}-shots_{}'.format(args.train_data, args.test_data, args.backbone, str(args.num_ways), str(args.num_shots), cur_time)
    os.makedirs(args.record_folder, exist_ok=True)

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    elif args.use_cuda:
        raise RuntimeError('You are using GPU mode, but GPUs are not available!')
    
    # construct model and optimizer
    args.image_len = 28 if args.train_data == 'omniglot' else 84
    args.out_channels, _ = get_outputs_c_h(args.backbone, args.image_len)
    
    model = ConvolutionalNeuralNetwork(args.backbone, args.out_channels, args.num_ways)

    if args.use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpus = torch.cuda.device_count()
        if args.multi_gpu:
            model = DataParallel(model)

        model = model.cuda()        
       
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    # training from the checkpoint
    if args.resume and args.resume_folder is not None:
        # load checkpoint
        checkpoint_path = os.path.join(args.resume_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, 'max_acc']) + '_checkpoint.pt.tar'))    # tag='max_acc' can be changed
        state = torch.load(checkpoint_path)
        if args.multi_gpu:
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
    
        train_log = state['train_log']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        # all_task_count表示在当前epoch前一共训练多少任务
        all_task_count = state['all_task_count']

        print('all_task_count: {}, initial_lr: {}'.format(str(all_task_count), str(initial_lr)))

    # training from scratch
    else:
        train_log = {}
        train_log['args'] = vars(args)
        train_log['train_loss'] = []
        train_log['train_acc'] = []
        train_log['val_loss'] = []
        train_log['val_acc'] = []
        train_log['max_acc'] = 0.0
        train_log['max_acc_i_task'] = 0
        initial_lr = args.lr
        all_task_count = 0
 
        if args.pretrain and args.backbone_path is not None:
            backbone_state = torch.load(args.backbone_path)
            if args.multi_gpu:
                model.module.encoder.load_state_dict(backbone_state)
            else:
                model.encoder.load_state_dict(backbone_state)

    # save the args into .json file
    with open(os.path.join(args.record_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    """get train datasets"""
    train_dataset = get_dataset(args, dataset_name=args.train_data, phase='train') 
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers) 
    """get validation datasets"""
    val_dataset = val_dataset = get_dataset(args, dataset_name=args.test_data, phase='val')
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers)
    """get test datasets"""
    test_dataset = get_dataset(args, dataset_name=args.test_data, phase='test') 
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_tasks,
        shuffle=True, num_workers=args.num_workers) 
   
    # training
    with tqdm(train_dataloader, total=int(args.train_tasks/args.batch_tasks), initial=int(all_task_count/args.batch_tasks)) as pbar:
        for train_batch_i, train_batch in enumerate(pbar):
            
            if train_batch_i >= args.train_tasks/args.batch_tasks:
                break
            
            model.train()
            # chech if lr should decrease as in schedule
            if (train_batch_i * args.batch_tasks) in args.schedule:
                initial_lr *=args.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr

            all_task_count +=args.batch_tasks

            support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in train_batch['train']] if args.use_cuda else [_ for _ in train_batch['train']]
            query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in train_batch['test']] if args.use_cuda else [_ for _ in train_batch['test']]
           
            train_loss = torch.tensor(0., device=support_inputs.device)
            train_acc = torch.tensor(0., device=support_inputs.device)

            for _ , (support_input, support_target, query_input,
                    query_target) in enumerate(zip(support_inputs, support_targets,
                    query_inputs, query_targets)):
                #meta inner loop
                support_logit = model(support_input)           
                train_inner_loss = F.cross_entropy(support_logit, support_target)

                model.zero_grad()
                params = gradient_update_parameters(model, train_inner_loss,
                    step_size=args.step_size, first_order=args.first_order)

                #meta outer loop
                query_logit = model(query_input, params=params)
                train_loss += F.cross_entropy(query_logit, query_target)    
                
                with torch.no_grad():
                    train_acc += get_accuracy(query_logit, query_target)
                    
            #得到每个batch中平均的acc和loss
            train_loss.div_(args.batch_tasks)   
            train_acc.div_(args.batch_tasks)
                         
            train_loss.backward()
            optimizer.step()

            pbar.set_postfix(train_acc='{0:.4f}'.format(train_acc.item()))

            # validation
            if all_task_count % args.validation_tasks == 0:
                val_loss_avg = Averager()
                val_acc_avg = Mean_confidence_interval()

                with tqdm(val_dataloader, total=int(args.val_tasks/args.batch_tasks)) as pbar:
                    for val_batch_i, val_batch in enumerate(pbar, 1):

                        if val_batch_i > (args.val_tasks / args.batch_tasks):
                            break
                        
                        support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in val_batch['train']] if args.use_cuda else [_ for _ in val_batch['train']]
                        query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in val_batch['test']] if args.use_cuda else [_ for _ in val_batch['test']]

                        # 每个task建模, 对于
                        val_loss = torch.tensor(0., device=support_inputs.device)
                        val_acc = torch.tensor(0., device=support_inputs.device)
                   
                        for _ , (support_input, support_target, query_input,
                            query_target) in enumerate(zip(support_inputs, support_targets,
                            query_inputs, query_targets)):
                            # meta inner loop
                            model.train()
                            support_logit = model(support_input)
                            val_inner_loss = F.cross_entropy(support_logit, support_target)

                            model.zero_grad()
                            params = gradient_update_parameters(model, val_inner_loss,
                                step_size=args.step_size, first_order=args.first_order)

                            # meta outer loop
                            with torch.no_grad():
                                model.eval()
                                query_logit = model(query_input, params=params)
                                val_loss += F.cross_entropy(query_logit, query_target)    
                                val_acc += get_accuracy(query_logit, query_target)

                        val_loss.div_(args.batch_tasks)   
                        val_acc.div_(args.batch_tasks)

                        pbar.set_postfix(val_acc='{0:.4f}'.format(val_acc.item()))

                        val_loss_avg.add(val_loss.item())
                        val_acc_avg.add(val_acc.item())
            
                # record
                val_acc_mean = val_acc_avg.item()

                print('all_task_count: {}, val_acc_mean: {}'.format(str(all_task_count), str(val_acc_mean)))
                if val_acc_mean > train_log['max_acc']:
                    train_log['max_acc'] = val_acc_mean
                    train_log['max_acc_i_task'] = all_task_count
                    save_model(model, args, tag='max_acc')

                train_log['train_loss'].append(train_loss.item())
                train_log['train_acc'].append(train_acc.item())
                train_log['val_loss'].append(val_loss_avg.item())
                train_log['val_acc'].append(val_acc_mean)

                save_checkpoint(args, model, train_log, optimizer, all_task_count, tag='max_acc')
                del val_loss_avg, val_acc_avg   

    # testing
    test_loss_avg = Averager()
    test_acc_avg = Mean_confidence_interval()
    
    model_path = os.path.join(args.record_folder, ('_'.join([args.model_name, args.train_data, args.test_data, args.backbone, 'max_acc']) + '.pt'))
    state = torch.load(model_path)
    if args.multi_gpu:
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    
    with tqdm(test_dataloader, total=int(args.test_tasks/args.batch_tasks)) as pbar:
        for test_batch_i, test_batch in enumerate(pbar, 1): 

            if test_batch_i > (args.test_tasks / args.batch_tasks):
                break
            
            support_inputs, support_targets = [_.cuda(non_blocking=True) for _ in test_batch['train']] if args.use_cuda else [_ for _ in test_batch['train']]
            query_inputs, query_targets = [_.cuda(non_blocking=True) for _ in test_batch['test']] if args.use_cuda else [_ for _ in test_batch['test']]

            test_loss = torch.tensor(0., device=support_inputs.device)
            test_acc = torch.tensor(0., device=support_inputs.device)

            for _ , (support_input, support_target, query_input,
                    query_target) in enumerate(zip(support_inputs, support_targets,
                    query_inputs, query_targets)):
                # meta inner loop
                model.train()
                support_logit = model(support_input)
                test_inner_loss = F.cross_entropy(support_logit, support_target)

                model.zero_grad()
                params = gradient_update_parameters(model, test_inner_loss,
                    step_size=args.step_size, first_order=args.first_order)

                # meta outer loop
                with torch.no_grad():
                    model.eval()
                    query_logit = model(query_input, params=params)
                    test_loss += F.cross_entropy(query_logit, query_target)            
                    test_acc += get_accuracy(query_logit, query_target)

            test_loss.div_(args.batch_tasks)   
            test_acc.div_(args.batch_tasks)
            pbar.set_postfix(text_acc='{0:.4f}'.format(test_acc.item()))

            test_loss_avg.add(test_loss.item())
            test_acc_avg.add(test_acc.item())

    # record
    index_values = [
        'test_acc',
        'best_i_task',    # the best_i_task of the highest val_acc
        'best_train_acc',    # the train_acc according to the best_i_task of the highest val_acc
        'best_train_loss',    # the train_loss according to the best_i_task of the highest val_acc
        'best_val_acc',
        'best_val_loss'
    ]
    best_index = int(train_log['max_acc_i_task'] / args.validation_tasks) - 1
    test_record = {}
    test_record_data = [
        test_acc_avg.item(return_str=True),
        str(train_log['max_acc_i_task']),
        str(train_log['train_acc'][best_index]),
        str(train_log['train_loss'][best_index]),
        str(train_log['max_acc']),
        str(train_log['val_loss'][best_index]),
    ]
    test_record[args.record_folder] = test_record_data
    test_record_file = os.path.join(args.record_folder, 'record_{}_{}_{}shot.csv'.format(args.train_data, args.test_data, args.num_shots))
    DataFrame(test_record, index=index_values).to_csv(test_record_file)







                      

    


    
            
            
