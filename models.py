

def set_model():
    from unet import UNet
    model = UNet(
        n_channels = 3,
        n_classes = 13
    )
    model = model.cuda()
    return model