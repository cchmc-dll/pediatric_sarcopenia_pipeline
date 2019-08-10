from .detection import  *

available_models = {'UNet2D' : UNetFull,

                    'VGG16Reg' : VGG16Reg,
                    'VGG16Dual' : VGG16RegDual,
                    'UNet1D' : CNNLine
         }

def Models(name):

    if name in available_models.keys():
        return available_models[name]
    else:
        print('invalid model name from {}'.format(available_models.keys()))
        exit(0)
        return None

def get_available_models():
    return available_models.keys()
