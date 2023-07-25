from torch import nn
from .gpt_modules import GPTEmbedding, GPTTransformerLayer
from .task_modules import SeqClassification, Seq2SeqClassification


class GPTStageBase(nn.Module):
    def __init__(self, args, vocab_size, num_classes):
        super(GPTStageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self.task = args.task
        self._vocab_size = vocab_size
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
        self._num_classes = num_classes
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = args.num_layers

    def _create_first_layer(self):
        return GPTEmbedding(self._vocab_size, self._embedding_dim, self._seq_length)

    def _create_last_layer(self):
        if self.task == 'SeqClassification':
            return SeqClassification(self._embedding_dim, self._num_classes)
        elif self.task == 'Seq2SeqClassification':
            return Seq2SeqClassification(self._vocab_size, self._embedding_dim)

    def _create_transformer_layer(self):
        return GPTTransformerLayer(self._embedding_dim, self._num_heads, self._feedforward_dim,
                                   use_checkpoint=True)


class GPTStageFirst(GPTStageBase):
    def __init__(self, args, vocab_size, num_classes, device):
        super(GPTStageFirst, self).__init__(args, vocab_size, num_classes)
        self.device = device
        module_list = [self._create_first_layer()]
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device))
        return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, args, vocab_size, num_classes, device):
        super(GPTStageMiddle, self).__init__(args, vocab_size, num_classes)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        self.model = nn.Sequential(*module_list).to(device)

    def forward(self, x):
        out = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
        return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, args, vocab_size, num_classes, device):
        super(GPTStageLast, self).__init__(args, vocab_size, num_classes)
        self.device = device
        module_list = []
        for _ in range(self._num_layers):
            module_list.append(self._create_transformer_layer())
        # module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list).to(device)
        self.task_layer = self._create_last_layer().to(device)
        # print(self.parameters())

    def forward(self, x, target=None):
        if self.task == 'SeqClassification':
            x = self.model(x.to(self.device)) if self._to_cpu else self.model(x)
            out = self.task_layer(x)
            return out.cpu() if self._to_cpu else out
        elif self.task == 'Seq2SeqClassification':
            assert target is not None
            x = self.model(x)
            return self.task_layer(x, target)
