import logging


#
# Singleton to access the logger.
#
class Logger:

    instance = None

    @staticmethod
    def get(name="Logger"):
        """
        Getter.
        :param name: the name of the logger.
        :return: the logger to use for logging information.
        """
        if Logger.instance is None:
            Logger.instance = logging.getLogger(name)
        return Logger.instance
