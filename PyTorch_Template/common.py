import os
from utils import ensure_dirs
import argparse
import yaml
import shutil


def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # define default hyperparameters
        self.default_config = DotDict({
            'output_dir': "your-output-dir",
            'data_root': "your-data-root",
            'exp_name': os.getcwd().split('/')[-1],
            'gpu_ids': None,
            'batch_size': 64,
            'num_workers': 8,
            'nr_epochs': 1000,
            'lr': 1e-3,
            'lr_step_size': 400,
            'ckpt': 'latest',
            'save_frequency': 100,
            'val_frequency': 10,
            'vis_frequency': 10
        })

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # experiment paths
        self.exp_dir = os.path.join(args.__dict__['output_dir'], args.__dict__['exp_name'])
        if phase == "train" and args.__dict__['cont'] is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        # load previously saved hyperparameters
        saved_config = {}
        if phase == "train" and os.path.exists(os.path.join(self.exp_dir, 'config.yaml')):
            with open(os.path.join(self.exp_dir, 'config.yaml'), 'r') as f:
                docs = yaml.load_all(f, Loader=yaml.FullLoader)
                for doc in docs:
                    for k, v in doc.items():
                        saved_config[k] = v

        # finalize hyperparameters
        merge_dict(saved_config, self.default_config)
        merge_dict(args.__dict__, saved_config)

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # create soft link to experiment log directory
        if not os.path.exists('train_log'):
            os.symlink(self.exp_dir, 'train_log')

        # save this configuration
        if self.is_train:
            with open('train_log/config.yaml', 'w') as f:
                yaml.dump(args.__dict__, f)

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training or testing configuration
        self._add_training_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--output_dir', type=str, default=self.default_config.output_dir, help="path to project folder where models and logs will be saved")
        group.add_argument('--data_root', type=str, default=self.default_config.data_root, help="path to source data folder")
        group.add_argument('--exp_name', type=str, default=self.default_config.exp_name, help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=None, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        group.add_argument('--batch_size', type=int, default=None, help="batch size")
        group.add_argument('--num_workers', type=int, default=None, help="number of workers for data loading")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        # group.add_argument("--z_dim", type=int, default=128)
        pass

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=None, help="total number of epochs to train")
        group.add_argument('--lr', type=float, default=None, help="initial learning rate")
        group.add_argument('--lr_step_size', type=int, default=None, help="step size for learning rate decay")
        group.add_argument('--continue', dest='cont',  action='store_true', default=False, help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default=None, required=False, help="desired checkpoint to restore")
        group.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        group.add_argument('--save_frequency', type=int, default=None, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=None, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=None, help="visualize output every x iterations")


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                if user[k] is None:
                    user[k] = v
                else:
                    user[k] = merge_dict(user[k], v)
    return user


if __name__ == "__main__":
    config = get_config('train')
