import multiprocessing as mp
import os



from GravNN.Networks.Configs import *
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def main():
    # dataframe where the network configuration info will be saved
    df_file = "Data/Dataframes/example_training.data"

    # a default set of hyperparameters / configuration details for PINN
    config = get_default_eros_config()

    # hyperparameters which overwrite defaults
    hparams = PINN_III()
    hparams.update(ReduceLrOnPlateauConfig())
    hparams.update(
        {
            "obj_file": [Eros().obj_8k],
            "N_dist": [5000],
            "N_train": [4500],
            "N_val": [500],
            "batch_size": [4096],
            "PINN_constraint_fcn": ["pinn_a"],
            # 'trainable_tanh' : [True],
            # 'tanh_k' : [1.0],
            # 'tanh_r' : [1.0],
        },
    )

    threads = 1
    args = configure_run_args(config, hparams)
    result_list = []
    #with mp.Pool(threads) as pool:

    results = run(args)
    configs = results.get()
    save_training(df_file, configs)


def run(config):
    # Tensorflow dependent functions must be defined inside of
    # run function for thread-safe behavior.
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
    from GravNN.Networks.utils import configure_tensorflow, populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    print(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)
    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    #===========================================
    import matplotlib.pyplot as plt
    import pandas as pd
    from GravNN.Analysis.PlanesExperiment import PlanesExperiment
    from GravNN.CelestialBodies.Asteroids import Eros
    from GravNN.GravityModels.Polyhedral import Polyhedral
    from GravNN.Networks.Model import load_config_and_model
    from GravNN.Support.Grid import Grid
    from GravNN.Trajectories import DHGridDist
    from GravNN.Visualization.MapBase import MapBase
    from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer

    planet = config["planet"][0]
    radius_bounds = [-planet.radius * 3, planet.radius * 3]
    max_percent = 25
    planes_exp = PlanesExperiment(model, config, radius_bounds, 30)
    planes_exp.run()
    vis = PlanesVisualizer(planes_exp)
    vis.plot(percent_max=max_percent)

    plt.show()
    #===========================================

    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
