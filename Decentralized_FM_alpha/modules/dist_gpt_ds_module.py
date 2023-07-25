import torch
from deepspeed.pipe import LayerSpec, PipelineModule
from .gpt_modules import GPTTransformerLayer, GPTEmbedding
from .task_modules import SeqClassification


class GlueSeqClassificationPipeModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True):
        super(GlueSeqClassificationPipeModel, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads, args.embedding_dim*4,
                                                   use_checkpoint=use_checkpoint))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = SeqClassification(args.embedding_dim, num_classes)

    def forward(self, inputs):
        print("Pipe model take input:", inputs)
        input_ids=inputs[0]
        input_emb = self.embedding(input_ids, None)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)

    def to_layers_for_deepspeed_pipeline(self):
        layers = [
            self.embedding,
            *self.transformers,
            self.classifier
        ]
        return layers


class GlueSeqClassificationPipeModelSpec(PipelineModule):
    def __init__(self, args, vocab_size, num_classes, use_checkpoint=True, **kwargs):
        self.use_checkpoint = use_checkpoint

        specs = [
            LayerSpec(GPTEmbedding, vocab_size, args.embedding_dim, args.seq_length)
        ]

        for _ in range(args.num_layers):
            specs.append(LayerSpec(
                GPTTransformerLayer, args.embedding_dim, args.num_heads, args.embedding_dim*4, use_checkpoint))

        specs.append(LayerSpec(SeqClassification, args.embedding_dim, num_classes))
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)
        super().__init__(layers=specs, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)

