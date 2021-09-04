import argparse
from datetime import datetime

def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--exp_id',
        default='',
        type=str,
        help='Unique id for this experiment')
    
    parser.add_argument(
        '--image_set',
        default='/Users/HemingY/Documents/brain_classification/data/split/fold_0',
        type=str,
        help='Directory path of image data')
    
    parser.add_argument(
        '--network',
        default='ResNet',
        type=str,
        help='Network structure. Options: resnet')
    
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help="Number of classes")
    
    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help="Number of workers")

    parser.add_argument(
        "--pos_weight", 
        type=float, 
        default=1,
        help='Weight of the positive class (for binary classification only)')
    
    
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    
    
    parser.add_argument(
        '--batch_size', default=2, type=int, help='Batch Size')
    
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    
    parser.add_argument(
        '--pretrain_path',
        default='pretrain/resnet_50.pth',
        type=str,
        help=
        'Path for pretrained model.'
    )
    
    
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    
    parser.add_argument(
        '--pretrained', action='store_true', help='If true, use a pretrained model.')

    parser.set_defaults(no_cuda=False)
    
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument(
        '--augmentation',
        default=0,
        type=int,
        help='data augmentation type')
    
    parser.add_argument(
        '--loss',
        default='cross_entropy',
        type=str,
        help='loss type')


    args = parser.parse_args()
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%H-%M-%S")
    args.save_folder = "./models/{}".format(args.exp_id)
    args.tensorboard_dir = "./logs_{}_{}".format(dt_string, args.exp_id)
    return args
