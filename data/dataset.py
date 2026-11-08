r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader
from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from common import utils


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath,):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath

        cls.transform=transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        transforms=cls.transform
        use_original_imgsize = split == 'val'
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transforms, split=split, shot=shot, use_original_imgsize=use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker,
                                worker_init_fn=utils.worker_init_fn if nworker > 0 else None)

        return dataloader
