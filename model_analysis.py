from analysis.AnalysisGUI import AnalysisGUI
from omegaconf import OmegaConf
import hydra
import sys


@hydra.main(config_path="config", config_name="analysis")
def main(config):
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Launch the graphical user interface used for the model analysis.
    gui = AnalysisGUI(config)
    gui.loop()

    # Exit the program.
    sys.exit()


if __name__ == '__main__':
    main()
