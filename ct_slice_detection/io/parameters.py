from ct_slice_detection.core.input_parser import InputParser


def parse_inputs():

    parser = InputParser()

    parser.add_parameter('restart', bool, False,
                         section='train', help='restart training by deleting all associated files')
    parser.add_parameter('num_epochs', int, default=5, section='train', help='number of epochs for training')
    parser.add_parameter('batch_size', int, default=3,
                         section='train', help='number of distinct images to samples from in a given batch')
    parser.add_parameter('img_batch_size', int, default=3,
                         section='train', help='number of samples obtained from a single image in a given batch')
    parser.add_parameter('input_shape', int, default=None, nargs=3,
                         section='train', help='image input shape')
    parser.add_parameter('model_input_shape', int, default=[None,None, 1], nargs=3,
                         section='train', help='model input shape')
    parser.add_parameter('model_name', str, default='UNet', section='train', help='name of model')
    parser.add_parameter('dataset_path', str, default=None, section='train', help='location of dataset .npz')
    parser.add_parameter('n_splits', int, default=3, section='train', help='number of splits for cross validation')
    parser.add_parameter('random_state', int, default=42, section='train', help='random seed')
    parser.add_parameter('ds_factor', int, default=2, section='train', help='output downsampling factor')
    parser.add_parameter('input_spacing', int, default=1, section='train', help='spacing of input image')
    parser.add_parameter('num_val', int, default=20, section='train',
                         help='number of validation samples during training')
    parser.add_parameter('do_crossval', bool, default=False, section='train', help='do cross validation')
    parser.add_parameter('flatten_output', bool, default=False,
                         section='train', help='1D output if true; otherwise, the output is 2D')
    parser.add_parameter('use_cache', bool, default=True, section='train', help='cache input image pre-processing')
    parser.add_parameter('cache_path', str, default=None,
                         section='train',
                         help='path to store the pre-processed images. If None, then model_path is used')
    parser.add_parameter('mode', str, default='heatmap',
                         section='train', help='labelmap as heatmap or regression', choices=['heatmap', 'reg'])
    parser.add_parameter('image_type', str, default='frontal', section='train', choices=['frontal', 'sagittal'])
    parser.add_parameter('cuda_devices', str, '0,1')
    parser.add_parameter('model_path', str, default='/tmp/slice_detection_1/')
    parser.add_parameter('sigma', float, default=3)
    parser.add_parameter('sampling_rate', float, default=0.5, help='rate to sample from crops that contain the slice')
    parser.add_parameter('do_augment', bool, default=True, section='train', help='enable augmentation')
    parser.add_parameter('preview_generator_output', bool, default=False,
                         section='train', help='preview generator output')
    parser.add_parameter('preview_training_output', bool, default=False,
                         section='train', help='preview intermediate training output')
    parser.add_parameter('regression_dual_output', bool, default=False,
                         section='train', help='enable dual output for regression')
    parser.add_parameter('do_checkpoint', bool, default=False,
                         section='train', help='enable model checkpoint saving')
    parser.add_parameter('test_mode', str, default='test', section='train')
    args = parser.parse()

    return args
