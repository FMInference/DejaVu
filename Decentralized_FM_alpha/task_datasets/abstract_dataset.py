"""This is stolen from Megatron to make a fair comparison."""
"""GLUE dataset."""
from abc import ABC
from abc import abstractmethod
from torch.utils.data import Dataset
from .data_utils import build_sample
from .data_utils import build_tokens_types_paddings_from_text


class GLUEAbstractDataset(ABC, Dataset):
    """GLUE base dataset class."""

    def __init__(self, task_name, dataset_name, datapaths,
                 tokenizer, max_seq_length, sample_as_tuple=False):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sample_as_tuple = sample_as_tuple
        print(' > building {} dataset for {}:'.format(self.task_name, self.dataset_name))
        # Process the files.
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        ids, types, paddings = build_tokens_types_paddings_from_text(
            raw_sample['text_a'], raw_sample['text_b'],
            self.tokenizer, self.max_seq_length)
        sample = build_sample(ids, types, paddings,
                              raw_sample['label'], raw_sample['uid'], self.sample_as_tuple)
        return sample

    @abstractmethod
    def process_samples_from_single_path(self, datapath):
        """Abstract method that takes a single path / filename and
        returns a list of dataset samples, each sample being a dict of
            {'text_a': string, 'text_b': string, 'label': int, 'uid': int}
        """
        pass