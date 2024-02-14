# %matplotlib inline
'''
Generate 'true' wake field raw_data for use in GP learning procedure
Inputs: Yaw Angles, Freestream Wind Velocity, Freestream Wind Direction, Turbine Topology
Need csv containing 'true' wake characteristics at each turbine (variables) at each time-step (rows).
'''

# git add . & git commit -m "updates" & git push origin
# ssh ahenry@eagle.hpc.nrel.gov
# cd ...
# sbatch ...

from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from floris import tools as wfct
import pandas as pd
from multiprocessing import Pool
import scipy
# from CaseGen_General import CaseGen_General
# from postprocessing import plot_wind_farm
from whoc.config import *
from array import array
from scipy.interpolate import interp1d, LinearNDInterpolator
from collections import defaultdict
from itertools import cycle, chain

# **************************************** Initialization **************************************** #

# Initialize
# fi_sim = wfct.floris_interface.FlorisInterface(WIND_FIELD_CONFIG["floris_input_file"])
# fi_model = wfct.floris_interface.FlorisInterface(floris_model_dir)

# for fi_temp in [fi_sim, fi_model]:
#     assert fi_temp.get_model_parameters()["Wake Deflection Parameters"]["use_secondary_steering"] == False
#     assert "use_yaw_added_recovery" not in fi_temp.get_model_parameters()["Wake Deflection Parameters"] or fi_temp.get_model_parameters()["Wake Deflection Parameters"]["use_yaw_added_recovery"] == False
#     assert "calculate_VW_velocities" not in fi_temp.get_model_parameters()["Wake Deflection Parameters"] or fi_temp.get_model_parameters()["Wake Deflection Parameters"]["calculate_VW_velocities"] == False

# **************************************** GENERATE TIME-VARYING FREESTREAM WIND SPEED/DIRECTION, YAW ANGLE, TURBINE TOPOLOGY SWEEP **************************************** #
# print(f'Simulating {N_CASES} total wake field cases...')

# **************************************** CLASS ********************************************* #
class WindField:
	def __init__(self, **config: dict):
		self.episode_time_step = None
		self.offline_probability = (
			config["offline_probability"] if "offline_probability" in config else 0.001
		)
		
		self.floris_input_file = (
			config["floris_input_file"]
		)
		
		self.wind_farm = wfct.floris_interface.FlorisInterface(self.floris_input_file)
		self.n_turbines = self.wind_farm.floris.farm.n_turbines
		self.wind_farm.turbine_indices = list(range(self.n_turbines))
		max_downstream_dist = max(self.wind_farm.layout_x[t]
		                          for t in range(self.n_turbines))
		min_downstream_dist = min(self.wind_farm.layout_x[t]
		                          for t in range(self.n_turbines))
		# exclude most downstream turbine
		upstream_turbine_indices = [t for t in range(self.n_turbines) if
		                            self.wind_farm.layout_x[t] < max_downstream_dist]
		n_upstream_turbines = len(upstream_turbine_indices)
		self.downstream_turbine_indices = [t for t in range(self.n_turbines) if
		                                   self.wind_farm.layout_x[t] > min_downstream_dist]
		self.n_downstream_turbines = len(self.downstream_turbine_indices)
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_speed_change_probability = config["wind_speed_change_probability"]  # 0.1
		self.wind_dir_change_probability = config["wind_dir_change_probability"]  # 0.1
		self.yaw_angle_change_probability = config["yaw_angle_change_probability"]
		self.ai_factor_change_probability = config["ai_factor_change_probability"]
		
		self.wind_speed_u_range = config["wind_speed_u_range"]
		self.wind_speed_v_range = config["wind_speed_v_range"]
		
		self.wind_speed_var = config["wind_speed_var"]  # 0.5
		self.wind_dir_var = config["wind_dir_var"]  # 5.0
		self.yaw_angle_var = config["yaw_angle_var"]
		self.ai_factor_var = config["ai_factor_var"]
		#
		# self.wind_speed_turb_std = config["wind_speed_turb_std"]  # 0.5
		# self.wind_dir_turb_std = config["wind_dir_turb_std"]  # 5.0
		# self.yaw_angle_turb_std = config["yaw_angle_turb_std"]  # 0
		# self.ai_factor_turb_std = config["ai_factor_turb_std"]  # 0
		self.wind_speed_noise_func = config["wind_speed_noise_func"]
		self.wind_speed_u_noise_args = config["wind_speed_u_noise_args"]
		self.wind_speed_v_noise_args = config["wind_speed_v_noise_args"]
		
		self.episode_max_time_steps = config["episode_max_time_steps"]
		self.wind_speed_sampling_time_step = config["wind_speed_sampling_time_step"]
		self.wind_dir_sampling_time_step = config["wind_dir_sampling_time_step"]
		self.yaw_angle_sampling_time_step = config["yaw_angle_sampling_time_step"]
		self.ai_factor_sampling_time_step = config["ai_factor_sampling_time_step"]
		
		self.yaw_angle_roc = config["yaw_angle_roc"]
		
		self.wind_speed_preview_time = config["wind_speed_preview_time"]
		
		self.n_preview_steps = int(self.wind_speed_preview_time // self.wind_speed_sampling_time_step)
	
	def _generate_online_bools_ts(self):
		return np.random.choice(
			[0, 1], size=(self.episode_max_time_steps, self.n_turbines),
			p=[self.offline_probability, 1 - self.offline_probability])
	
	def _sample_wind_preview(self, current_measurements, n_preview_steps, n_samples, noise_func=np.random.multivariate_normal, noise_args=None):
		"""
        corr(X) = (diag(Kxx))^(-0.5)Kxx(diag(Kxx))^(-0.5)
        low variance and high covariance => high correlation
        """
		noise_args = {}
		noise_args["mean"] = [current_measurements[0] for j in range(n_preview_steps + 1)] \
							+ [current_measurements[1] for j in range(n_preview_steps + 1)]
		# variance of u[j], and v[j], grows over the course of the prediction horizon (by 1+j*0.05, or 5% for each step), and has an initial variance of 0.25 the value of the current measurement
		# prepend 0 variance values for the deterministic measurements at the current (0) time-step
		o = 0.25
		var_u = np.array([current_measurements[0] * (1. + j*0.05) * o for j in range(0, n_preview_steps + 1)])
		var_v = np.array([current_measurements[1] * (1. + j*0.05) * o for j in range(0, n_preview_steps + 1)])
		
		noise_args["cov"] = np.diag(np.concatenate([var_u, var_v]))
		
		cov_u = np.diag(var_u)
		cov_v = np.diag(var_v)
		# covariance of u[j], v[j] 
		# is zero for u and v elements (no correlation),
		# positive and greater for adjacent elements u[j], u[j +/- 1], and v[j], v[j +/- 1] 
		# more off-diagonal covariance elements are farther apart in time over the prediction horizon (i.e. the row number is farther from the column number)
		# for the covariance matrix to be positive definite, off-diagonal elements should be less than diagonal elements, so p should be a fraction
		# requirement on the variance such that covariance does not become negative, (plus safety factor of 5%)
		p = 1.0 * (1 - (1 / (n_preview_steps)))**0.5
		for i in range(1, n_preview_steps + 1):
			# off-diagonal elements
			x = p * var_u[:n_preview_steps - i + 1] - (i - 1) * ((var_u[:n_preview_steps - i + 1] / p) / n_preview_steps)
			cov_u += np.diag(x, k=i)
			cov_u += np.diag(x, k=-i)
			x = p * var_v[:n_preview_steps - i + 1] - (i - 1) * ((var_v[:n_preview_steps - i + 1] / p) / n_preview_steps)
			cov_v += np.diag(x, k=i)
			cov_v += np.diag(x, k=-i)
		
		# np.all(np.diff(np.vstack(y), axis=0) < 0)
		noise_args["cov"] = scipy.linalg.block_diag(cov_u, cov_v)
		# print(np.sum(np.linalg.eigvals(noise_args["cov"]) < 0))
		if False:
			# visualize covariance matrix for testing purposes
			import seaborn as sns
			import matplotlib.pyplot as plt     

			fig, ax = plt.subplots(figsize=(15,15)) 
			sns.heatmap(noise_args["cov"], annot=True, fmt='g', ax=ax, annot_kws={'size': 10})
			#annot=True to annotate cells, ftm='g' to disable scientific notation
			# annot_kws si size  of font in heatmap
			# labels, title and ticks
			ax.set_xlabel('Columns') 
			ax.set_ylabel('Rows')
			ax.set_title('Covariance Matrix: NN') 
			# ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 10)
			# ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 10)
			plt.show()
		
		# if not np.all(np.linalg.eigvals(noise_args["cov"]) > 0):
		# 	print("oh no")
		
		# if not np.all(np.isclose(np.sqrt(noise_args["cov"].T @ noise_args["cov"]), noise_args["cov"])):
		#     print('oh no')
		
		noise_args["size"] = n_samples
		preview = noise_func(**noise_args)

		# can't set zero variance for u,v predictions associated with j=0, since it would make a covariance matrix that is not positive definite,
		# so instead we set the first predictions in the preview to the measured values
		preview[:, 0] = current_measurements[0]
		preview[:, n_preview_steps + 1] = current_measurements[1]
		return preview
	
	def _generate_change_ts(self, val_range, val_var, change_prob, sample_time_step,
	                        noise_func=None, noise_args=None, roc=None):
		# initialize at random wind speed
		init_val = np.random.choice(
			np.arange(val_range[0], val_range[1], val_var)
		)
		
		# randomly increase or decrease mean wind speed or keep static
		random_vals = np.random.choice(
			[-val_var, 0, val_var],
			size=(int(self.episode_max_time_steps * DT // sample_time_step)),
			p=[change_prob / 2,
			   1 - change_prob,
			   change_prob / 2,
			   ]
		)
		if roc is None:
			# if we assume instantaneous change (ie over a single DT)
			a = array('d', [
				y
				for x in random_vals
				for y in (x,) * sample_time_step])  # repeat random value x sample_time_step times
			delta_vals = array('d',
			                   interp1d(np.arange(0, self.episode_max_time_steps * DT, sample_time_step), random_vals,
			                            fill_value='extrapolate', kind='previous')(
				                   np.arange(0, self.episode_max_time_steps * DT, DT)))
		
		else:
			# else we assume a linear change between now and next sample time, considering roc as max slope allowed
			for i in range(len(random_vals) - 1):
				diff = random_vals[i + 1] - random_vals[i]
				if abs(diff) > roc * sample_time_step:
					random_vals[i + 1] = random_vals[i] + (diff / abs(diff)) \
					                     * (roc * sample_time_step)
			
			assert (np.abs(np.diff(random_vals)) <= roc * sample_time_step).all()
			delta_vals = array('d',
			                   interp1d(np.arange(0, self.episode_max_time_steps * DT, sample_time_step),
			                            random_vals, kind='linear', fill_value='extrapolate')(
				                   np.arange(0, self.episode_max_time_steps * DT, DT)))
		
		if noise_func is None:
			noise_vals = np.zeros_like(delta_vals)
		else:
			noise_args["size"] = (int(self.episode_max_time_steps * DT // self.wind_speed_sampling_time_step),)
			noise_vals = noise_func(**noise_args)
		
		# add mean and noise and bound the wind speed to given range
		ts = array('d', [init_val := np.clip(init_val + delta + n, *val_range)
		                 for delta, n in zip(delta_vals, noise_vals)])
		
		return ts
	
	def _generate_stochastic_freestream_wind_speed_ts(self):
		# initialize at random wind speed
		init_val = [
			np.random.choice(np.arange(self.wind_speed_u_range[0], self.wind_speed_u_range[1], self.wind_speed_var)),
			np.random.choice(np.arange(self.wind_speed_v_range[0], self.wind_speed_v_range[1], self.wind_speed_var))
	    ]
		n_time_steps = int(self.episode_max_time_steps * DT // self.wind_speed_sampling_time_step)
		u_ts = np.zeros((n_time_steps,))
		v_ts = np.zeros((n_time_steps,))

		# number of samples generated for each time-step, used to compute running average
		n_u = np.zeros((n_time_steps,)) 
		n_v = np.zeros((n_time_steps,))

		for k in range(n_time_steps):
			horizon_preview = self._sample_wind_preview(init_val, self.n_preview_steps, 1, noise_func=np.random.multivariate_normal, noise_args=None)
			horizon_preview_u = horizon_preview[0, :self.n_preview_steps + 1].squeeze()[:n_time_steps - k]
			horizon_preview_v = horizon_preview[0, self.n_preview_steps + 1:].squeeze()[:n_time_steps - k]
			
			u_ts[k:k + self.n_preview_steps + 1] = ((u_ts[k:k + self.n_preview_steps + 1] * n_u[k:k + self.n_preview_steps + 1]) + horizon_preview_u) / (n_u[k:k + self.n_preview_steps + 1] + 1)
			v_ts[k:k + self.n_preview_steps + 1] = ((v_ts[k:k + self.n_preview_steps + 1] * n_v[k:k + self.n_preview_steps + 1]) + horizon_preview_v) / (n_v[k:k + self.n_preview_steps + 1] + 1)

			n_u[k:k + self.n_preview_steps + 1] += 1
			n_v[k:k + self.n_preview_steps + 1] += 1

			init_val[0] = u_ts[k]
			init_val[1] = v_ts[k]

		return u_ts, v_ts

	def _generate_freestream_wind_speed_u_ts(self):
		
		u_ts = self._generate_change_ts(self.wind_speed_u_range, self.wind_speed_var,
										self.wind_speed_change_probability,
										self.wind_speed_sampling_time_step,
										self.wind_speed_noise_func,
										self.wind_speed_u_noise_args)
		return u_ts
	
	def _generate_freestream_wind_speed_v_ts(self):

		v_ts = self._generate_change_ts(self.wind_speed_v_range, self.wind_speed_var,
									self.wind_speed_change_probability,
									self.wind_speed_sampling_time_step,
									self.wind_speed_noise_func,
									self.wind_speed_v_noise_args)
		return v_ts
	
	def _generate_freestream_wind_dir_ts(self):
		freestream_wind_dir_ts = self._generate_change_ts(WIND_DIR_RANGE, self.wind_dir_var,
		                                                  self.wind_dir_change_probability,
		                                                  self.wind_dir_turb_std,
		                                                  self.wind_dir_sampling_time_step)
		return freestream_wind_dir_ts
	
	def _generate_yaw_angle_ts(self):
		yaw_angle_ts = [self._generate_change_ts(YAW_ANGLE_RANGE, self.yaw_angle_var,
		                                         self.yaw_angle_change_probability,
		                                         self.yaw_angle_turb_std,
		                                         self.yaw_angle_sampling_time_step,
		                                         roc=self.yaw_angle_roc)
		                for i in range(self.n_turbines)]
		yaw_angle_ts = np.array(yaw_angle_ts).T
		
		return yaw_angle_ts
	
	def _generate_ai_factor_ts(self):
		online_bool_ts = self._generate_online_bools_ts()
		
		set_ai_factor_ts = [self._generate_change_ts(AI_FACTOR_RANGE, self.ai_factor_var,
		                                             self.ai_factor_change_probability,
		                                             self.ai_factor_var,
		                                             self.ai_factor_sampling_time_step)
		                    for _ in range(self.n_turbines)]
		set_ai_factor_ts = np.array(set_ai_factor_ts).T
		
		effective_ai_factor_ts = np.array(
			[
				[set_ai_factor_ts[i][t] if online_bool_ts[i][t] else EPS
				 for t in range(self.n_turbines)]
				for i in range(self.episode_max_time_steps)
			]
		)
		
		return effective_ai_factor_ts


def plot_ts(wf):
	# Plot vs. time
	fig_ts, ax_ts = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_ts, '__len__'):
		ax_ts = ax_ts.flatten()
	else:
		ax_ts = [ax_ts]
	
	time = wf.df['Time']
	freestream_wind_speed_u = wf.df['FreestreamWindSpeedU'].to_numpy()
	freestream_wind_speed_v = wf.df['FreestreamWindSpeedV'].to_numpy()
	freestream_wind_mag = np.linalg.norm(np.vstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=0)
	freestream_wind_dir = np.arctan(freestream_wind_speed_u / freestream_wind_speed_v) * (180 / np.pi) + 180
	
	ax_ts[0].plot(time, freestream_wind_speed_u)
	ax_ts[1].plot(time, freestream_wind_speed_v)
	ax_ts[2].plot(time, freestream_wind_mag)
	ax_ts[3].plot(time, freestream_wind_dir)
	
	ax_ts[0].set(title='Freestream Wind Speed, U [m/s]')
	ax_ts[1].set(title='Freestream Wind Speed, V [m/s]')
	ax_ts[2].set(title='Freestream Wind Magnitude [m/s]')
	ax_ts[3].set(title='Freestream Wind Direction [deg]')
	
	for ax in ax_ts:
		ax.set(xticks=time[0:-1:int(60 // DT)], xlabel='Time [s]')
	
	fig_ts.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wind_field_ts.png'))
	# fig_ts.show()


def generate_wind_ts(config, from_gaussian, case_idx):
	wf = WindField(**config)
	print(f'Simulating case #{case_idx}')
	# define freestream time series
	if from_gaussian:
		freestream_wind_speed_u, freestream_wind_speed_v = wf._generate_stochastic_freestream_wind_speed_ts()
	else:
		freestream_wind_speed_u = np.array(wf._generate_freestream_wind_speed_u_ts())
		freestream_wind_speed_v = np.array(wf._generate_freestream_wind_speed_v_ts())
	
	time = np.arange(0, wf.episode_max_time_steps*DT, wf.wind_speed_sampling_time_step)
	# define noise preview
	
	# compute directions
	u_only_dirs = np.zeros_like(freestream_wind_speed_u)
	u_only_dirs[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u >= 0)] = 270
	u_only_dirs[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u < 0)] = 90
	u_only_dirs = (u_only_dirs - 180) * (np.pi / 180)
	
	dirs = np.arctan(np.divide(freestream_wind_speed_u, freestream_wind_speed_v,
	                           out=np.ones_like(freestream_wind_speed_u) * np.nan,
	                           where=freestream_wind_speed_v != 0),
	                 out=u_only_dirs,
	                 where=freestream_wind_speed_v != 0)
	dirs[dirs < 0] = np.pi + dirs[dirs < 0]
	dirs = (dirs * (180 / np.pi)) + 180
	
	# save case raw_data as dataframe
	wind_field_data = {
		'Time': time,
		'FreestreamWindSpeedU': freestream_wind_speed_u,
		'FreestreamWindSpeedV': freestream_wind_speed_v,
		'FreestreamWindMag': np.linalg.norm(np.vstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=0),
		'FreestreamWindDir': dirs
	}
	wind_field_df = pd.DataFrame(data=wind_field_data)
	
	# export case raw_data to csv
	wind_field_df.to_csv(os.path.join(DATA_SAVE_DIR, f'case_{case_idx}.csv'))
	wf.df = wind_field_df
	return wf

def generate_wind_preview(wind_preview_generator, n_preview_steps, n_samples, current_freestream_measurements, time_step):
	
	# define noise preview
	# noise_func = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
	
	wind_preview_data = defaultdict(list)
	noise_preview = wind_preview_generator(current_measurements=current_freestream_measurements, 
										n_preview_steps=n_preview_steps, n_samples=n_samples)
	u_preview = noise_preview[:, :n_preview_steps + 1].squeeze()
	v_preview = noise_preview[:, n_preview_steps + 1:].squeeze()
	mag_preview = np.linalg.norm(np.stack([u_preview, v_preview], axis=2), axis=2)
	
	# compute directions
	u_only_dir = np.zeros_like(u_preview)
	u_only_dir[(v_preview == 0) & (u_preview >= 0)] = 270
	u_only_dir[(v_preview == 0) & (u_preview < 0)] = 90
	u_only_dir = (u_only_dir - 180) * (np.pi / 180)
	
	dir_preview = np.arctan(np.divide(u_preview, v_preview,
	                          out=np.ones_like(u_preview) * np.nan,
	                          where=v_preview != 0),
	                out=u_only_dir,
	                where=v_preview != 0)
	dir_preview[dir_preview < 0] = np.pi + dir_preview[dir_preview < 0]
	dir_preview = (dir_preview * (180 / np.pi)) + 180
	
	# dir = np.arctan([u / v for u, v in zip(u_preview, v_preview)]) * (180 / np.pi) + 180
	
	for j in range(n_preview_steps + 1):
		wind_preview_data[f"FreestreamWindSpeedU_{j}"] += list(u_preview[:, j])
		wind_preview_data[f"FreestreamWindSpeedV_{j}"] += list(v_preview[:, j])
		wind_preview_data[f"FreestreamWindMag_{j}"] += list(mag_preview[:, j])
		wind_preview_data[f"FreestreamWindDir_{j}"] += list(dir_preview[:, j])
	
	return wind_preview_data


def generate_wind_preview_ts(config, case_idx, wind_field_data):
	wf = WindField(**config)
	print(f'Generating noise preview for case #{case_idx}')
	
	time = np.arange(0, wf.episode_max_time_steps * DT, wf.wind_speed_sampling_time_step)
	mean_freestream_wind_speed_u = wind_field_data[case_idx]['FreestreamWindSpeedU'].to_numpy()
	mean_freestream_wind_speed_v = wind_field_data[case_idx]['FreestreamWindSpeedV'].to_numpy()
	
	# save case raw_data as dataframe
	wind_preview_data = defaultdict(list)
	wind_preview_data["Time"] = time
	
	for u, v in zip(mean_freestream_wind_speed_u, mean_freestream_wind_speed_v):
		noise_preview = wf._sample_wind_preview(current_measurements=[u, v], n_preview_steps=wf.n_preview_steps, n_samples=1, noise_func=np.random.multivariate_normal, noise_args=None)
		u_preview = noise_preview[0, :wf.n_preview_steps + 1].squeeze()
		v_preview = noise_preview[0, wf.n_preview_steps + 1:].squeeze()
		mag = np.linalg.norm(np.vstack([u_preview, v_preview]), axis=0)
		
		# compute directions
		u_only_dir = np.zeros_like(u_preview)
		u_only_dir[(v_preview == 0) & (u_preview >= 0)] = 270
		u_only_dir[(v_preview == 0) & (u_preview < 0)] = 90
		u_only_dir = (u_only_dir - 180) * (np.pi / 180)
		
		dir = np.arctan(np.divide(u_preview, v_preview,
		                          out=np.ones_like(u_preview) * np.nan,
		                          where=v_preview != 0),
		                out=u_only_dir,
		                where=v_preview != 0)
		dir[dir < 0] = np.pi + dir[dir < 0]
		dir = (dir * (180 / np.pi)) + 180
		
		# dir = np.arctan([u / v for u, v in zip(u_preview, v_preview)]) * (180 / np.pi) + 180
		
		for i in range(wf.n_preview_steps):
			wind_preview_data[f"FreestreamWindSpeedU_{i}"].append(u_preview[i])
			wind_preview_data[f"FreestreamWindSpeedV_{i}"].append(v_preview[i])
			wind_preview_data[f"FreestreamWindMag_{i}"].append(mag[i])
			wind_preview_data[f"FreestreamWindDir_{i}"].append(dir[i])
	
	wind_preview_df = pd.DataFrame(data=wind_preview_data)
	
	# export case raw_data to csv
	wind_preview_df.to_csv(os.path.join(DATA_SAVE_DIR, f'preview_case_{case_idx}.csv'))
	wf.df = wind_preview_df
	return wf


def plot_distribution_samples(df, n_preview_steps):
	# Plot vs. time
	
	freestream_wind_speed_u = df[[f'FreestreamWindSpeedU_{j}' for j in range(n_preview_steps)]].to_numpy()
	freestream_wind_speed_v = df[[f'FreestreamWindSpeedV_{j}' for j in range(n_preview_steps)]].to_numpy()
	freestream_wind_mag = df[[f'FreestreamWindMag_{j}' for j in range(n_preview_steps)]].to_numpy()
	freestream_wind_dir = df[[f'FreestreamWindDir_{j}' for j in range(n_preview_steps)]].to_numpy()
	
	n_samples = freestream_wind_speed_u.shape[0]
	colors = cm.rainbow(np.linspace(0, 1, n_samples))
	preview_time = np.arange(n_preview_steps)

	fig_scatter, ax_scatter = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_scatter, '__len__'):
		ax_scatter = ax_scatter.flatten()
	else:
		ax_scatter = [ax_scatter]
	
	fig_plot, ax_plot = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_plot, '__len__'):
		ax_plot = ax_plot.flatten()
	else:
		ax_plot = [ax_plot]
	
	for i in range(n_samples):
		ax_scatter[0].scatter(preview_time, freestream_wind_speed_u[i, :], marker='o', color=colors[i])
		ax_scatter[1].scatter(preview_time, freestream_wind_speed_v[i, :], marker='o', color=colors[i])
		ax_scatter[2].scatter(preview_time, freestream_wind_mag[i, :], marker='o', color=colors[i])
		ax_scatter[3].scatter(preview_time, freestream_wind_dir[i, :], marker='o', color=colors[i])
	
	
	for i, c in zip(range(n_samples), cycle(colors)):
		ax_plot[0].plot(preview_time, freestream_wind_speed_u[i, :], color=c)
		ax_plot[1].plot(preview_time, freestream_wind_speed_v[i, :], color=c)
		ax_plot[2].plot(preview_time, freestream_wind_mag[i, :], color=c)
		ax_plot[3].plot(preview_time, freestream_wind_dir[i, :], color=c)
	
	for axs in [ax_scatter, ax_plot]:
		axs[0].set(title='Freestream Wind Speed, U [m/s]')
		axs[1].set(title='Freestream Wind Speed, V [m/s]')
		axs[2].set(title='Freestream Wind Magnitude [m/s]')
		axs[3].set(title='Freestream Wind Direction [deg]')
	
	for ax in chain(ax_scatter, ax_plot):
		ax.set(xticks=preview_time, xlabel='Preview Time-Steps')
	
	# fig_scatter.show()
	# fig_plot.show()
	fig_scatter.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wind_field_preview_samples1.png'))
	fig_plot.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wind_field_preview_samples2.png'))


def plot_distribution_ts(wf):
	# Plot vs. time
	fig_scatter, ax_scatter = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_scatter, '__len__'):
		ax_scatter = ax_scatter.flatten()
	else:
		ax_scatter = [ax_scatter]
	
	fig_plot, ax_plot = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_plot, '__len__'):
		ax_plot = ax_plot.flatten()
	else:
		ax_plot = [ax_plot]
	
	time = wf.df['Time'].to_numpy()
	freestream_wind_speed_u = wf.df[[f'FreestreamWindSpeedU_{i}' for i in range(wf.n_preview_steps)]].to_numpy()
	freestream_wind_speed_v = wf.df[[f'FreestreamWindSpeedV_{i}' for i in range(wf.n_preview_steps)]].to_numpy()
	freestream_wind_mag = (freestream_wind_speed_u ** 2 + freestream_wind_speed_v ** 2) ** 0.5
	# freestream_wind_dir = np.arctan(freestream_wind_speed_u / freestream_wind_speed_v) * (180 / np.pi) + 180
	
	# compute directions
	u_only_dir = np.zeros_like(freestream_wind_speed_u)
	u_only_dir[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u >= 0)] = 270
	u_only_dir[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u < 0)] = 90
	u_only_dir = (u_only_dir - 180) * (np.pi / 180)
	
	freestream_wind_dir = np.arctan(np.divide(freestream_wind_speed_u, freestream_wind_speed_v,
	                                          out=np.ones_like(freestream_wind_speed_u) * np.nan,
	                                          where=freestream_wind_speed_v != 0),
	                                out=u_only_dir,
	                                where=freestream_wind_speed_v != 0)
	freestream_wind_dir[freestream_wind_dir < 0] = np.pi + freestream_wind_dir[freestream_wind_dir < 0]
	freestream_wind_dir = (freestream_wind_dir * (180 / np.pi)) + 180
	
	colors = cm.rainbow(np.linspace(0, 1, wf.n_preview_steps))
	
	idx = slice(600)
	for i in range(wf.n_preview_steps):
		ax_scatter[0].scatter(time[idx] + i * wf.wind_speed_sampling_time_step, freestream_wind_speed_u[idx, i],
		                      marker='o', color=colors[i])
		ax_scatter[1].scatter(time[idx] + i * wf.wind_speed_sampling_time_step, freestream_wind_speed_v[idx, i],
		                      marker='o', color=colors[i])
		ax_scatter[2].scatter(time[idx] + i * wf.wind_speed_sampling_time_step, freestream_wind_mag[idx, i], marker='o',
		                      color=colors[i])
		ax_scatter[3].scatter(time[idx] + i * wf.wind_speed_sampling_time_step, freestream_wind_dir[idx, i], marker='o',
		                      color=colors[i])
	
	idx = slice(10)
	for k, c in zip(range(len(time[idx])), cycle(colors)):
		# i = (np.arange(k * DT, k * DT + wf.wind_speed_preview_time, wf.wind_speed_sampling_time_step) * (
		# 		1 // DT)).astype(int)
		i = slice(k, k + int(wf.wind_speed_preview_time // wf.wind_speed_sampling_time_step), 1)
		ax_plot[0].plot(time[i], freestream_wind_speed_u[k, :], color=c)
		ax_plot[1].plot(time[i], freestream_wind_speed_v[k, :], color=c)
		ax_plot[2].plot(time[i], freestream_wind_mag[k, :], color=c)
		ax_plot[3].plot(time[i], freestream_wind_dir[k, :], color=c)
	
	for axs in [ax_scatter, ax_plot]:
		axs[0].set(title='Freestream Wind Speed, U [m/s]')
		axs[1].set(title='Freestream Wind Speed, V [m/s]')
		axs[2].set(title='Freestream Wind Magnitude [m/s]')
		axs[3].set(title='Freestream Wind Direction [deg]')
	
	for ax in chain(ax_scatter, ax_plot):
		ax.set(xticks=time[idx][0:-1:int(60 // DT)], xlabel='Time [s]')
	
	# fig_scatter.show()
	# fig_plot.show()
	fig_scatter.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wind_field_preview_ts1.png'))
	fig_plot.savefig(os.path.join(FIG_DIR, f'{FARM_LAYOUT}_wind_field_preview_ts2.png'))


def generate_multi_wind_ts(config, n_cases):
	if n_cases == 1:
		wind_field_data = []
		for i in range(n_cases):
			wind_field_data.append(generate_wind_ts(config, True, i))
		plot_ts(wind_field_data[0])
		return wind_field_data
	
	
	else:
		pool = Pool()
		res = pool.map(partial(generate_wind_ts, True, config), range(n_cases))
		pool.close()


def generate_multi_wind_preview_ts(config, n_cases, wind_field_data):
	if n_cases == 1:
		wind_field_preview_data = []
		for i in range(n_cases):
			wind_field_preview_data.append(generate_wind_preview_ts(config, i, wind_field_data))
		plot_distribution_ts(wind_field_preview_data[0])
		return wind_field_preview_data
	
	else:
		pool = Pool()
		res = pool.map(partial(generate_wind_preview_ts, config, wind_field_data), range(n_cases))
		pool.close()


if __name__ == '__main__':
	wind_field_data = generate_multi_wind_ts(WIND_FIELD_CONFIG, N_CASES)
	generate_multi_wind_preview_ts(WIND_FIELD_CONFIG, N_CASES, wind_field_data)
