import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    
    df_file = "Data/Dataframes/sh_stats_Brillouin.data"
    trajectory = DHGridDist(planet,  planet.radius, degree=density_deg)

    # df_file = "Data/Dataframes/sh_stats_LEO.data"
    # trajectory = DHGridDist(planet, planet.radius + 420000.0, degree=density_deg)

    # df_file = "Data/Dataframes/sh_stats_GEO.data"
    # trajectory = DHGridDist(planet, planet.radius + 35786000.0, degree=density_deg)

    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
    grid_true = Grid(trajectory=trajectory, accelerations=a)

    deg_list =  np.linspace(1, 100, 100,dtype=int)[1:]
    deg_list = np.append(deg_list, [110, 150, 200, 215, 250, 300, 400, 500, 700, 900])
    df_all = pd.DataFrame()
    for deg in deg_list:
        
        x, a, u = get_sh_data(trajectory, model_file, max_deg=deg, deg_removed=2)

        grid_pred = Grid(trajectory=trajectory, accelerations=a)
        diff = grid_pred - grid_true
    
        # This ensures the same features are being evaluated independent of what degree is taken off at beginning
        two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
        three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

        rse_stats = mean_std_median(diff.total, prefix='rse')
        sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")
        sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")

        extras = {'deg' : [deg],
                  'max_error' : [np.max(diff.total)]
                  }

        entries = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }
        #for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats, sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)
        
        df = pd.DataFrame().from_dict(entries).set_index('deg')
        df_all = df_all.append(df)
    
    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()