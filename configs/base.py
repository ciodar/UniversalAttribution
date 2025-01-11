class Config(object):
    # random seed
    seed = 0

    # feature extraction
    backbone = 'vit_base_patch16_clip_224.laion2b'

    # dataset
    input_data = 'img'
    transform = 'default'
    batch_size = 8
    train_batch_size = 16
    eval_batch_size = 64
    num_workers = 8
    resize_size = (224, 224)
    samples_per_class = 4000

    # perturbations
    train_perturbations = None
    perturbations = None
    perturbation_intensity = -1
    resize_range = (128, 1024)
    blur_range = (3, 15)
    jpeg_range = (10, 75)
    crop_range = (.05, .2)


