from analysis.AnalysisGUI import AnalysisGUI
from omegaconf import OmegaConf
import hydra


@hydra.main(config_path="config", config_name="analysis")
def main(config):
    # Make hydra able to load tuples.
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

    # Create the graphical user interface used for the model analysis.
    gui = AnalysisGUI(config)

    # Keep the gui open for the user.
    gui.loop()


if __name__ == '__main__':
    main()
