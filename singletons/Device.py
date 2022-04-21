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

    @staticmethod
    def send(models):
        """
        Send the models to the device, i.e. gpu if available or cpu otherwise.
        :param models: the list of model to send to the device.
        :return: nothinh
        """
        for model in models:
            model.to(Device.get())
