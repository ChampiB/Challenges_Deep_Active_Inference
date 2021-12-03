import torch


#
# Singleton to access the type of device to use, i.e. GPU or CPU.
#
class Device:

    instance = None

    @staticmethod
    def get():
        """
        Getter.
        :return: the device on which computation should be performed.
        """
        if Device.instance is None:
            Device.instance = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return Device.instance
