"""QQP dataset."""
import torch
from .data_utils import clean_text
from .abstract_dataset import GLUEAbstractDataset


LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length, data_as_tuple=False,
                 test_label=0):
        self.test_label = test_label
        super().__init__('QQP', name, datapaths,
                         tokenizer, max_seq_length, data_as_tuple)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split('\t')
                if first:
                    first = False
                    if len(row) == 3:
                        is_test = True
                        print('   reading {}, {}, and {} columns and '
                                     'setting labels to {}'.format(
                                         row[0].strip(), row[1].strip(),
                                         row[2].strip(), self.test_label))
                    else:
                        assert len(row) == 6
                        print('    reading {}, {}, {}, and {} columns'
                                     ' ...'.format(
                                         row[0].strip(), row[3].strip(),
                                         row[4].strip(), row[5].strip()))
                    continue

                if is_test:
                    assert len(row) == 3, 'expected length 3: {}'.format(row)
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 6:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[3].strip())
                        text_b = clean_text(row[4].strip())
                        label = int(row[5].strip())
                    else:
                        print('***WARNING*** index error, ' 'skipping: {}'.format(row))
                        continue
                    if len(text_a) == 0:
                        print('***WARNING*** zero length a, ' 'skipping: {}'.format(row))
                        continue
                    if len(text_b) == 0:
                        print('***WARNING*** zero length b, ' 'skipping: {}'.format(row))
                        continue
                assert label in LABELS
                assert uid >= 0
                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    print('  > processed {} so far ...'.format(total))

        print(' >> processed {} samples.'.format(len(samples)))
        return samples


def get_glue_qqp_train_data_loader(args, tokenizer, num_workers=0):
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader


def get_glue_qqp_test_data_loader(args, tokenizer, num_workers=0):
    test_dataset = QQPDataset('testing', args.test_data, tokenizer, args.seq_length)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=test_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return test_data_loader
