from modules.hf_gpt2_module import *


def test_hugging_face_inference():
    config = GPTConfig.from_pretrained('../pretrained_debug_models/gpt2/')
    layer_embeddings = GPTEmbeddings(config).eval()
    layer0 = GPTBlock(config).eval()
    layer1 = GPTBlock(config).eval()
    layer_lm = GPTLMHead(config).eval()
    print(layer0)
    print(layer1)

    # input_x = torch.tensor([[12332, 20475, 43926]])
    input_x = torch.randint(0, 10000, (4, 128))
    x = input_x
    past_list = [None, None]
    generate_seq_len = 10
    y = []
    for i in range(generate_seq_len):
        print("--------------------")
        print('x.shape', x.shape)
        emb = layer_embeddings(x, past_list[0])
        print("emb.shape:", emb.shape)

        #emb, present = layer0(emb, past_list[0])
        #past_list[0] = present
        #print("emb.shape:", emb.shape, "present.shape:", present[0].shape, present[1].shape)

        emb, past_list[0] = layer0(emb, past_list[0])
        print("emb.shape:", emb.shape, "present.shape:", past_list[0][0].shape, past_list[0][1].shape)

        #emb, present = layer1(emb, past_list[1])
        #past_list[1] = present
        #print("emb.shape:", emb.shape, "present.shape:", present[0].shape, present[1].shape)

        emb, past_list[1] = layer0(emb, past_list[1])
        print("emb.shape:", emb.shape, "present.shape:", past_list[1][0].shape, past_list[1][1].shape)

        z = layer_lm(emb)
        print("z.shape:", z.shape)
        x = z.argmax(-1)[:, -1:]
        y.append(x)
        print("y:", len(y))
    y = torch.cat([input_x] + y, 1)


def main():
    test_hugging_face_inference()


if __name__ == '__main__':
    main()
