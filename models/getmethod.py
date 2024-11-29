import exvit.MViT as MViT


def getMethod(name):
    if name=='MViT':
        return MViT(
        )
            patch_size = args.Patches,
            num_patches = band_MultiModal,
            num_classes = num_classes,
            dim = 64,
            depth = 6,
            heads = 4,
            mlp_dim = 32,
            dropout = 0.1,
            emb_dropout = 0.1,
            mode = args.Mode
        )