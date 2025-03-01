import os
import torch
from model import SOR_Mamba,iTransformer,iInformer,iReformer,\
    iFlowformer, iFlashformer, Transformer, Informer, Reformer, Flowformer, \
    Flashformer, Autoformer, S_Mamba

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SOR_Mamba':SOR_Mamba,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'Autoformer': Autoformer,
            'S_Mamba': S_Mamba,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
