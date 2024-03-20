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

import whoc
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from floris import tools as wfct
import pandas as pd
from multiprocessing import Pool
import scipy
import os
# from CaseGen_General import CaseGen_General
# from postprocessing import plot_wind_farm
# from whoc.config import *
import yaml
from array import array
from scipy.interpolate import interp1d, LinearNDInterpolator
from collections import defaultdict
from itertools import cycle, chain
from glob import glob

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

		self.fig_dir = config["fig_dir"]
		self.data_save_dir = config["data_save_dir"]

		self.episode_time_step = None
		self.offline_probability = (
			config["offline_probability"] if "offline_probability" in config else 0.001
		)
		
		self.simulation_dt = config["simulation_sampling_time"]

		self.n_turbines = config["n_turbines"]
		
		# set wind speed/dir change probabilities and variability parameters
		self.wind_speed_change_probability = config["wind_speed"]["change_probability"]  # 0.1
		self.wind_dir_change_probability = config["wind_dir"]["change_probability"]  # 0.1
		self.yaw_angle_change_probability = config["yaw_angles"]["change_probability"]
		self.ai_factor_change_probability = config["ai_factors"]["change_probability"]
		
		self.wind_speed_range = config["wind_speed"]["range"]
		self.wind_speed_u_range = config["wind_speed"]["u_range"]
		self.wind_speed_v_range = config["wind_speed"]["v_range"]
		self.wind_dir_range = config["wind_dir"]["range"]
		self.yaw_offsets_range = config["yaw_angles"]["range"]
		self.ai_factor_range = config["ai_factors"]["range"]

		self.wind_dir_turb_std = config["wind_dir"]["turb_std"]
		
		self.wind_speed_var = config["wind_speed"]["var"]  # 0.5
		self.wind_dir_var = config["wind_dir"]["var"]  # 5.0
		self.yaw_angle_var = config["yaw_angles"]["var"]
		self.ai_factor_var = config["ai_factors"]["var"]
		
		self.wind_speed_turb_std = config["wind_speed"]["turb_std"]  # 0.5
		self.wind_dir_turb_std = config["wind_dir"]["turb_std"]  # 5.0
		self.yaw_angle_turb_std = config["yaw_angles"]["turb_std"]  # 0
		self.ai_factor_turb_std = config["ai_factors"]["turb_std"]  # 0
		noise_func_parts = config["wind_speed"]["noise_func"].split(".")
		func = globals()[noise_func_parts[0]]
		for i in range(1, len(noise_func_parts)):
			func = getattr(func, noise_func_parts[i])
		self.wind_speed_noise_func = func
		self.wind_speed_u_noise_args = config["wind_speed"]["u_noise_args"]
		self.wind_speed_v_noise_args = config["wind_speed"]["v_noise_args"]
		
		self.simulation_max_time = config["simulation_max_time"]
		# self.wind_speed_sampling_time_step = config["wind_speed"]["sampling_time_step"]
		# self.wind_dir_sampling_time_step = config["wind_dir"]["sampling_time_step"]
		# self.yaw_angle_sampling_time_step = config["yaw_angles"]["sampling_time_step"]
		# self.ai_factor_sampling_time_step = config["ai_factors"]["sampling_time_step"]
		
		self.yaw_rate = config["yaw_angles"]["roc"]
		
		# self.wind_speed_preview_time = config["wind_speed_preview_time"]
		
		# self.n_preview_steps = int(self.wind_speed_preview_time // self.wind_speed_sampling_time_step)
		# self.preview_dt = int((self.wind_speed_preview_time) // self.wind_speed_sampling_time_step)
		self.simulation_max_time_steps = int(self.simulation_max_time // self.simulation_dt)
	
	def _generate_online_bools_ts(self):
		return np.random.choice(
			[0, 1], size=(self.simulation_max_time_steps, self.n_turbines),
			p=[self.offline_probability, 1 - self.offline_probability])
	
	def _sample_wind_preview(self, current_measurements, n_preview_steps, preview_dt, n_samples, noise_func=np.random.multivariate_normal, noise_args=None):
		"""
		corr(X) = (diag(Kxx))^(-0.5)Kxx(diag(Kxx))^(-0.5)
		low variance and high covariance => high correlation
		"""
		noise_args = {}
		# noise_args["mean"] = [current_measurements[0] for j in range(n_preview_steps + preview_dt)] \
		# 					+ [current_measurements[1] for j in range(n_preview_steps + preview_dt)]
		# mean = [0] * ((n_preview_steps + preview_dt) * 2)
		mean_u = [self.wind_speed_u_range[0] + ((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) / 2)] * (n_preview_steps + preview_dt)
		mean_v = [self.wind_speed_v_range[0] + ((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) / 2)] * (n_preview_steps + preview_dt)

		# noise_args["mean"] = [(self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) / 2 for j in range(n_preview_steps + preview_dt)] \
		# 					+ [(self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) / 2 for j in range(n_preview_steps + preview_dt)]


		# variance of u[j], and v[j], grows over the course of the prediction horizon (by 1+j*0.05, or 5% for each step), and has an initial variance of 0.25 the value of the current measurement
		# prepend 0 variance values for the deterministic measurements at the current (0) time-step
		# o = 0.2
		# we want it to be very unlikely (3 * standard deviations) that the value will stray outside of the desired range
		
		# p = 0.1
		# var_u = np.array([(((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) * p) / 3)**2 * (1. + j*0.02) for j in range(0, n_preview_steps + preview_dt)])
		# var_v = np.array([(((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) * p) / 3)**2 * (1. + j*0.02) for j in range(0, n_preview_steps + preview_dt)])
		# QUESTION: we want growing uncertainty in prediction further along in the prediction horizon, not growing variation - should variance remain the same?
		p = 0.5
		q = 0.000
		var_u = np.array([(((self.wind_speed_u_range[1] - self.wind_speed_u_range[0]) * p * (2. - np.exp(-q * j))) / 3)**2  for j in range(0, n_preview_steps + preview_dt)])
		var_v = np.array([(((self.wind_speed_v_range[1] - self.wind_speed_v_range[0]) * p * (2. - np.exp(-q * j))) / 3)**2 for j in range(0, n_preview_steps + preview_dt)])

		# cov = np.diag(np.concatenate([var_u, var_v]))
		
		cov_u = np.diag(var_u)
		cov_v = np.diag(var_v)
		# covariance of u[j], v[j] 
		# is zero for u and v elements (no correlation),
		# positive and greater for adjacent elements u[j], u[j +/- 1], and v[j], v[j +/- 1] 
		# more off-diagonal covariance elements are farther apart in time over the prediction horizon (i.e. the row number is farther from the column number)
		# for the covariance matrix to be positive definite, off-diagonal elements should be less than diagonal elements, so p should be a fraction
		# requirement on the variance such that covariance does not become negative, (plus safety factor of 5%)
		# greater multiple leads to greater changes between time-steps
		p = 1.0
		q = 0.005
		# a = 1.0 * (1 - (1 / (n_preview_steps)))**0.5
		for i in range(1, n_preview_steps + preview_dt):
		# for i in range(1, n_preview_steps):
			# off-diagonal elements
			# b = var_u[:n_preview_steps - i + preview_dt] * p
			# b = var_u[:n_preview_steps + preview_dt - i] * p
			b = np.array([var_u[0]] * (n_preview_steps + preview_dt - i)) * p
			# x = b * (a - ((i - 1) / (a * (n_preview_steps + preview_dt))))
			x = b * np.exp(-q * i)
			cov_u += np.diag(x, k=i)
			cov_u += np.diag(x, k=-i)

			# b = var_v[:n_preview_steps - i + preview_dt] * p
			# b = var_v[:n_preview_steps + preview_dt - i] * p
			b = np.array([var_v[0]] * (n_preview_steps + preview_dt - i)) * p
			x = b * np.exp(-q * i)
			# x = b * (a - ((i - 1) / (a * (n_preview_steps + preview_dt))))
			cov_v += np.diag(x, k=i)
			cov_v += np.diag(x, k=-i)
		
		# np.all(np.triu(np.diff(cov_u, axis=1)) <= 0)
		# np.all(np.diff(np.vstack(y), axis=0) < 0)
		# cov = scipy.linalg.block_diag(cov_u, cov_v)
		
		if False:
			# visualize covariance matrix for testing purposes
			import seaborn as sns
			import matplotlib.pyplot as plt     

			fig, ax = plt.subplots(1, 2, figsize=(15,15)) 
			sns.heatmap(cov_u, annot=True, fmt='g', ax=ax[0], annot_kws={'size': 10})
			sns.heatmap(cov_v, annot=True, fmt='g', ax=ax[1], annot_kws={'size': 10})
			#annot=True to annotate cells, ftm='g' to disable scientific notation
			# annot_kws si size  of font in heatmap
			# labels, title and ticks
			ax[1].set_xlabel('Columns') 
			ax[0].set_ylabel('Rows')
			ax[0].set_title('Covariance Matrix: NN') 
			# ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 10)
			# ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 10)
			plt.show()
		
		cond_mean_u = mean_u[1:] + cov_u[1:, :1] @ np.linalg.inv(cov_u[:1, :1]) @ (current_measurements[0] - mean_u[:1])
		cond_mean_v = mean_v[1:] + cov_v[1:, :1] @ np.linalg.inv(cov_v[:1, :1]) @ (current_measurements[1] - mean_v[:1])

		cond_cov_u = cov_u[1:, 1:] - cov_u[1:, :1] @ np.linalg.inv(cov_u[:1, :1]) @ cov_u[:1, 1:]
		cond_cov_v = cov_v[1:, 1:] - cov_v[1:, :1] @ np.linalg.inv(cov_v[:1, :1]) @ cov_v[:1, 1:]

		noise_args["mean"] = np.concatenate([cond_mean_u, cond_mean_v])
		noise_args["cov"] = scipy.linalg.block_diag(cond_cov_u, cond_cov_v)
		noise_args["size"] = n_samples
		preview = noise_func(**noise_args)
		# iters = 0

		# cond = (preview[:, :n_preview_steps + 1] > self.wind_speed_u_range[1]) | (preview[:, :n_preview_steps + 1] < self.wind_speed_u_range[0])
		# sample_cond = np.any(cond, axis=1)
		# while np.any(cond):
		# 	noise_args["size"] = np.sum(sample_cond)
		# 	preview[sample_cond, :n_preview_steps + 1] = noise_func(**noise_args)[:, :n_preview_steps + 1]
		# 	cond = (preview[:, :n_preview_steps + 1] > self.wind_speed_u_range[1]) | (preview[:, :n_preview_steps + 1] < self.wind_speed_u_range[0])
		# 	sample_cond = np.any(cond, axis=1)
		# 	iters += 1

		# iters = 0
		# cond = (preview[:, n_preview_steps + 1:] > self.wind_speed_v_range[1]) | (preview[:, n_preview_steps + 1:] < self.wind_speed_v_range[0])
		# sample_cond = np.any(cond, axis=1)
		# while np.any(sample_cond):
		# 	noise_args["size"] = np.sum(sample_cond)
		# 	preview[sample_cond, n_preview_steps + 1:] = noise_func(**noise_args)[:, n_preview_steps + 1:]
		# 	cond = (preview[:, n_preview_steps + 1:] > self.wind_speed_v_range[1]) | (preview[:, n_preview_steps + 1:] < self.wind_speed_v_range[0])
		# 	sample_cond = np.any(cond, axis=1)
		# 	iters += 1
		# np.save("mean_true.npy", noise_args["mean"])
		# np.save("cov_true.npy", noise_args["cov"])
		# np.save("mean_preview.npy", noise_args["mean"])
		# np.save("cov_preview.npy", noise_args["cov"])
		# can't set zero variance for u,v predictions associated with j=0, since it would make a covariance matrix that is not positive definite,
		# so instead we set the first predictions in the preview to the measured values

		# preview[:, :n_preview_steps + preview_dt] += (current_measurements[0] - preview[:, 0])[:, np.newaxis]
		# preview[:, n_preview_steps + preview_dt:] += (current_measurements[1] - preview[:, n_preview_steps + preview_dt])[:, np.newaxis]
		# preview[:, :n_preview_steps + preview_dt] += current_measurements[0]
		# preview[:, n_preview_steps + preview_dt:] += current_measurements[1]

		preview = np.hstack([np.broadcast_to(current_measurements[0], (n_samples, 1)), preview[:, :n_preview_steps + preview_dt], 
					   np.broadcast_to(current_measurements[1], (n_samples, 1)), preview[:, n_preview_steps + preview_dt:]])
		
		# can't set first value of preview like this - it violates the correlation principle between adjacent time-steps
		# preview[:, 0] = current_measurements[0]
		# preview[:, n_preview_steps + preview_dt] = current_measurements[1]
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
			size=(int(self.simulation_max_time // sample_time_step)),
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
							   interp1d(np.arange(0, self.simulation_max_time, sample_time_step), random_vals,
										fill_value='extrapolate', kind='previous')(
								   np.arange(0, self.simulation_max_time, self.simulation_dt)))
		
		else:
			# else we assume a linear change between now and next sample time, considering roc as max slope allowed
			for i in range(len(random_vals) - 1):
				diff = random_vals[i + 1] - random_vals[i]
				if abs(diff) > roc * sample_time_step:
					random_vals[i + 1] = random_vals[i] + (diff / abs(diff)) \
										 * (roc * sample_time_step)
			
			assert (np.abs(np.diff(random_vals)) <= roc * sample_time_step).all()
			delta_vals = array('d',
							   interp1d(np.arange(0, self.simulation_max_time, sample_time_step),
										random_vals, kind='linear', fill_value='extrapolate')(
								   np.arange(0, self.simulation_max_time, self.simulation_dt)))
		
		if noise_func is None:
			noise_vals = np.zeros_like(delta_vals)
		else:
			noise_args["size"] = (int(self.simulation_max_time // self.wind_speed_sampling_time_step),)
			noise_vals = noise_func(**noise_args)
		
		# add mean and noise and bound the wind speed to given range
		ts = array('d', [init_val := np.clip(init_val + delta + n, *val_range)
						 for delta, n in zip(delta_vals, noise_vals)])
		
		return ts
	
	def _generate_stochastic_freestream_wind_speed_ts(self, n_preview_steps, preview_dt, seed=None):
		np.random.seed(seed)
		# initialize at random wind speed
		init_val = [
			np.random.choice(np.arange(self.wind_speed_u_range[0], self.wind_speed_u_range[1], self.wind_speed_var)),
			np.random.choice(np.arange(self.wind_speed_v_range[0], self.wind_speed_v_range[1], self.wind_speed_var))
		]
		n_time_steps = int(self.simulation_max_time // self.simulation_dt) + n_preview_steps
		generate_incrementally = False

		# TODO up/down sample to simulation time-step

		if generate_incrementally:
			
			full_u_ts = []
			full_v_ts = []
			for ts_subset_i in range(int(np.ceil(n_time_steps / n_preview_steps))):
				i = 0
				while 1:
					wind_sample = self._sample_wind_preview(init_val, n_preview_steps, preview_dt, 1, noise_func=np.random.multivariate_normal, noise_args=None)
					# u_ts, v_ts = wind_sample[0, :n_preview_steps + preview_dt], wind_sample[0, n_preview_steps + preview_dt:]
					u_ts, v_ts = wind_sample[0, :n_preview_steps + preview_dt], wind_sample[0, n_preview_steps + preview_dt:]
					if (np.all(u_ts <= self.wind_speed_u_range[1]) and np.all(u_ts >= self.wind_speed_u_range[0]) 
						and np.all(v_ts <= self.wind_speed_v_range[1]) and np.all(v_ts >= self.wind_speed_v_range[0])):
						break
					i += 1

				full_u_ts.append(u_ts)
				full_v_ts.append(v_ts)
				init_val = [u_ts[-1], v_ts[-1]]

			full_u_ts = np.concatenate(full_u_ts)[:n_time_steps + preview_dt]
			full_v_ts = np.concatenate(full_v_ts)[:n_time_steps + preview_dt]
		else:
			wind_sample = self._sample_wind_preview(init_val, n_time_steps, preview_dt, 1, noise_func=np.random.multivariate_normal, noise_args=None)
			full_u_ts, full_v_ts = wind_sample[0, :n_time_steps + preview_dt], wind_sample[0, n_time_steps + preview_dt:]

		return full_u_ts, full_v_ts

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
		freestream_wind_dir_ts = self._generate_change_ts(self.wind_dir_range, self.wind_dir_var,
														  self.wind_dir_change_probability,
														  self.wind_dir_turb_std,
														  self.wind_dir_sampling_time_step)
		return freestream_wind_dir_ts

	def _generate_yaw_angle_ts(self):
		yaw_angle_ts = [self._generate_change_ts(self.yaw_offsets_range, self.yaw_angle_var,
												 self.yaw_angle_change_probability,
												 self.yaw_angle_turb_std,
												 self.yaw_angle_sampling_time_step,
												 roc=self.yaw_rate)
						for i in range(self.n_turbines)]
		yaw_angle_ts = np.array(yaw_angle_ts).T
		
		return yaw_angle_ts
	
	def _generate_ai_factor_ts(self):
		online_bool_ts = self._generate_online_bools_ts()
		
		set_ai_factor_ts = [self._generate_change_ts(self.ai_factor_range, self.ai_factor_var,
													 self.ai_factor_change_probability,
													 self.ai_factor_var,
													 self.ai_factor_sampling_time_step)
							for _ in range(self.n_turbines)]
		set_ai_factor_ts = np.array(set_ai_factor_ts).T
		
		effective_ai_factor_ts = np.array(
			[
				[set_ai_factor_ts[i][t] if online_bool_ts[i][t] else 0.0
				 for t in range(self.n_turbines)]
				for i in range(self.simulation_max_time_steps)
			]
		)
		
		return effective_ai_factor_ts


def plot_ts(df, fig_dir):
	# Plot vs. time
	fig_ts, ax_ts = plt.subplots(2, 2, sharex=True)  # len(case_list), 5)
	if hasattr(ax_ts, '__len__'):
		ax_ts = ax_ts.flatten()
	else:
		ax_ts = [ax_ts]
	
	time = df['Time']
	freestream_wind_speed_u = df['FreestreamWindSpeedU'].to_numpy()
	freestream_wind_speed_v = df['FreestreamWindSpeedV'].to_numpy()
	freestream_wind_mag = df['FreestreamWindMag'].to_numpy()
	freestream_wind_dir = df['FreestreamWindDir'].to_numpy()
	# freestream_wind_mag = np.linalg.norm(np.vstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=0)
	# freestream_wind_dir = np.arctan(freestream_wind_speed_u / freestream_wind_speed_v) * (180 / np.pi) + 180
	
	ax_ts[0].plot(time, freestream_wind_speed_u)
	ax_ts[1].plot(time, freestream_wind_speed_v)
	ax_ts[2].plot(time, freestream_wind_mag)
	ax_ts[3].plot(time, freestream_wind_dir)
	
	ax_ts[0].set(title='Freestream Wind Speed, U [m/s]')
	ax_ts[1].set(title='Freestream Wind Speed, V [m/s]')
	ax_ts[2].set(title='Freestream Wind Magnitude [m/s]')
	ax_ts[3].set(title='Freestream Wind Direction [deg]')
	
	for ax in ax_ts[2:]:
		ax.set(xticks=time[0:-1:int(60 * 10 // (time[1] - time[0]))], xlabel='Time [s]')
	
	fig_ts.savefig(os.path.join(fig_dir, f'wind_field_ts.png'))
	# fig_ts.show()


def generate_wind_ts(config, from_gaussian, case_idx, save_name="", seed=None):
	wf = WindField(**config)
	print(f'Simulating case #{case_idx}')
	# define freestream time series
	if from_gaussian:
		freestream_wind_speed_u, freestream_wind_speed_v = wf._generate_stochastic_freestream_wind_speed_ts(config["n_preview_steps"], config["preview_dt"], seed=seed)
	else:
		freestream_wind_speed_u = np.array(wf._generate_freestream_wind_speed_u_ts())
		freestream_wind_speed_v = np.array(wf._generate_freestream_wind_speed_v_ts())
	
	time = np.arange(len(freestream_wind_speed_u)) * wf.simulation_dt
	# define noise preview
	
	# compute directions
	u_only_dirs = np.zeros_like(freestream_wind_speed_u)
	u_only_dirs[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u >= 0)] = 270.
	u_only_dirs[(freestream_wind_speed_v == 0) & (freestream_wind_speed_u < 0)] = 90.
	u_only_dirs = (u_only_dirs - 180) * (np.pi / 180)
	
	dirs = np.arctan(np.divide(freestream_wind_speed_u, freestream_wind_speed_v,
							   out=np.ones_like(freestream_wind_speed_u) * np.nan,
							   where=freestream_wind_speed_v != 0),
					 out=u_only_dirs,
					 where=freestream_wind_speed_v != 0)
	dirs[dirs < 0] = np.pi + dirs[dirs < 0]
	dirs = (dirs * (180 / np.pi)) + 180

	mags = np.linalg.norm(np.vstack([freestream_wind_speed_u, freestream_wind_speed_v]), axis=0)
	
	# save case raw_data as dataframe
	wind_field_data = {
		'Time': time,
		'FreestreamWindSpeedU': freestream_wind_speed_u,
		'FreestreamWindSpeedV': freestream_wind_speed_v,
		'FreestreamWindMag': mags,
		'FreestreamWindDir': dirs
	}
	wind_field_df = pd.DataFrame(data=wind_field_data)
	
	# export case raw_data to csv
	wind_field_df.to_csv(os.path.join(wf.data_save_dir, f'{save_name}case_{case_idx}.csv'))
	wf.df = wind_field_df
	return wf

def generate_wind_preview(wind_preview_generator, n_preview_steps, preview_dt, n_samples, current_freestream_measurements, simulation_time):
	
	# define noise preview
	# noise_func = wf._sample_wind_preview(noise_func=np.random.multivariate_normal, noise_args=None)
	# TODO consider simulation sampling_time
	
	wind_preview_data = defaultdict(list)
	noise_preview = wind_preview_generator(current_measurements=current_freestream_measurements, 
										n_preview_steps=n_preview_steps, preview_dt=preview_dt, n_samples=n_samples)
	u_preview = noise_preview[:, :n_preview_steps + preview_dt:preview_dt]
	v_preview = noise_preview[:, n_preview_steps + preview_dt::preview_dt]
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
	
	for j in range(int((n_preview_steps + preview_dt) // preview_dt)):
		wind_preview_data[f"FreestreamWindSpeedU_{j}"] += list(u_preview[:, j])
		wind_preview_data[f"FreestreamWindSpeedV_{j}"] += list(v_preview[:, j])
		wind_preview_data[f"FreestreamWindMag_{j}"] += list(mag_preview[:, j])
		wind_preview_data[f"FreestreamWindDir_{j}"] += list(dir_preview[:, j])
	
	return wind_preview_data


def generate_wind_preview_ts(config, case_idx, wind_field_data):
	wf = WindField(**config)
	print(f'Generating noise preview for case #{case_idx}')
	
	time = np.arange(0, wf.simulation_max_time, wf.simulation_dt)
	mean_freestream_wind_speed_u = wind_field_data[case_idx]['FreestreamWindSpeedU'].to_numpy()
	mean_freestream_wind_speed_v = wind_field_data[case_idx]['FreestreamWindSpeedV'].to_numpy()
	
	# save case raw_data as dataframe
	wind_preview_data = defaultdict(list)
	wind_preview_data["Time"] = time
	
	for u, v in zip(mean_freestream_wind_speed_u, mean_freestream_wind_speed_v):
		noise_preview = wf._sample_wind_preview(current_measurements=[u, v], 
										  n_preview_steps=config["n_preview_steps"], preview_dt=wf.preview_dt, n_samples=1, 
										  noise_func=np.random.multivariate_normal, noise_args=None)
		u_preview = noise_preview[0, :config["n_preview_steps"] + 1].squeeze()
		v_preview = noise_preview[0, config["n_preview_steps"] + 1:].squeeze()
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
		
		for i in range(config["n_preview_steps"]):
			wind_preview_data[f"FreestreamWindSpeedU_{i}"].append(u_preview[i])
			wind_preview_data[f"FreestreamWindSpeedV_{i}"].append(v_preview[i])
			wind_preview_data[f"FreestreamWindMag_{i}"].append(mag[i])
			wind_preview_data[f"FreestreamWindDir_{i}"].append(dir[i])
	
	wind_preview_df = pd.DataFrame(data=wind_preview_data)
	
	# export case raw_data to csv
	wind_preview_df.to_csv(os.path.join(wf.data_save_dir, f'preview_case_{case_idx}.csv'))
	wf.df = wind_preview_df
	return wf


def plot_distribution_samples(df, n_preview_steps, fig_dir):
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
	fig_scatter.savefig(os.path.join(fig_dir, f'wind_field_preview_samples1.png'))
	fig_plot.savefig(os.path.join(fig_dir, f'wind_field_preview_samples2.png'))


def plot_distribution_ts(wf, n_preview_steps):
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
	freestream_wind_speed_u = wf.df[[f'FreestreamWindSpeedU_{i}' for i in range(n_preview_steps)]].to_numpy()
	freestream_wind_speed_v = wf.df[[f'FreestreamWindSpeedV_{i}' for i in range(n_preview_steps)]].to_numpy()
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
	
	colors = cm.rainbow(np.linspace(0, 1, n_preview_steps))
	
	idx = slice(600)
	for i in range(n_preview_steps):
		ax_scatter[0].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_speed_u[idx, i],
							  marker='o', color=colors[i])
		ax_scatter[1].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_speed_v[idx, i],
							  marker='o', color=colors[i])
		ax_scatter[2].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_mag[idx, i], marker='o',
							  color=colors[i])
		ax_scatter[3].scatter(time[idx] + i * wf.simulation_dt, freestream_wind_dir[idx, i], marker='o',
							  color=colors[i])
	
	idx = slice(10)
	for k, c in zip(range(len(time[idx])), cycle(colors)):
		# i = (np.arange(k * DT, k * DT + wf.wind_speed_preview_time, wf.wind_speed_sampling_time_step) * (
		# 		1 // DT)).astype(int)
		i = slice(k, k + int(wf.wind_speed_preview_time // wf.simulation_dt), 1)
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
		ax.set(xticks=time[idx][0:-1:int(60 // wf.dt)], xlabel='Time [s]')
	
	# fig_scatter.show()
	# fig_plot.show()
	fig_scatter.savefig(os.path.join(wf.fig_dir, f'wind_field_preview_ts1.png'))
	fig_plot.savefig(os.path.join(wf.fig_dir, f'wind_field_preview_ts2.png'))


def generate_multi_wind_ts(config, save_name="", seed=None):
	if config["n_wind_field_cases"] == 1:
		wind_field_data = []
		for i in range(config["n_wind_field_cases"]):
			wind_field_data.append(generate_wind_ts(config, True, i, save_name, seed))
		plot_ts(wind_field_data[0].df, config["fig_dir"])
		
	else:
		pool = Pool()
		wind_field_data = pool.map(partial(generate_wind_ts, config=config, from_gaussian=True, save_name=save_name, seed=seed), range(config["n_wind_field_cases"]))
		pool.close()
	
	return wind_field_data


def generate_multi_wind_preview_ts(config, wind_field_data):
	if config["n_wind_field_cases"] == 1:
		wind_field_preview_data = []
		for i in range(config["n_wind_field_cases"]):
			wind_field_preview_data.append(generate_wind_preview_ts(config, i, wind_field_data))
		plot_distribution_ts(wind_field_preview_data[0])
		return wind_field_preview_data
	
	else:
		pool = Pool()
		res = pool.map(partial(generate_wind_preview_ts, config, wind_field_data), range(config["n_wind_field_cases"]))
		pool.close()


if __name__ == '__main__':
	# with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml")) as fp:
	# 	wind_field_config = yaml.safe_load(fp)
	# wind_field_data = generate_multi_wind_ts(wind_field_config)
	# generate_multi_wind_preview_ts(wind_field_config, wind_field_data)
	from hercules.utilities import load_yaml
	import seaborn as sns
	sns.set_theme(style="darkgrid")

	regenerate_wind_field = True
	
	with open(os.path.join(os.path.dirname(whoc.__file__), "wind_field", "wind_field_config.yaml"), "r") as fp:
		wind_field_config = yaml.safe_load(fp)

	wind_field_config["simulation_max_time"] = 3600
	wind_field_config["n_preview_steps"] = 600
	wind_field_config["preview_dt"] = 60

	# instantiate wind field if files don't already exist
	wind_field_dir = os.path.join('/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/wind_field_data/raw_data')        
	wind_field_filenames = glob(f"{wind_field_dir}/case_*.csv")
	n_wind_field_cases = 1
	if not os.path.exists(wind_field_dir):
		os.makedirs(wind_field_dir)

	seed = 5
	if not len(wind_field_filenames) or regenerate_wind_field:
		# generate_multi_wind_ts(wind_field_config, save_name="short_", seed=seed)
		generate_multi_wind_ts(wind_field_config, save_name="", seed=seed)
		wind_field_filenames = [f"case_{i}.csv" for i in range(n_wind_field_cases)]
		regenerate_wind_field = True

	# if wind field data exists, get it
	wind_field_data = []
	if os.path.exists(wind_field_dir):
		for fn in wind_field_filenames:
			wind_field_data.append(pd.read_csv(os.path.join(wind_field_dir, fn)))
	
	plot_ts(pd.DataFrame(wind_field_data[0]), wind_field_config["fig_dir"])
	# plt.savefig(os.path.join(wind_field_config["fig_dir"], "wind_field_ts.png"))
	# true wind disturbance time-series
	case_idx = 0
	wind_mag_ts = wind_field_data[case_idx]["FreestreamWindMag"].to_numpy()
	wind_dir_ts = wind_field_data[case_idx]["FreestreamWindDir"].to_numpy()
	wind_u_ts = wind_field_data[case_idx]["FreestreamWindSpeedU"].to_numpy()
	wind_v_ts = wind_field_data[case_idx]["FreestreamWindSpeedV"].to_numpy()

	input_dict = load_yaml(os.path.join(os.path.dirname(whoc.__file__), "../examples/hercules_input_001.yaml"))
	input_dict["controller"]["n_wind_preview_samples"] = 3

	wf = WindField(**wind_field_config)
	stochastic_wind_preview_func = partial(generate_wind_preview, 
								wf._sample_wind_preview, 
								input_dict["controller"]["n_horizon"] * int(input_dict["controller"]["dt"] // input_dict["dt"]),
								int(input_dict["controller"]["dt"] // input_dict["dt"]),
								input_dict["controller"]["n_wind_preview_samples"])
			
	def persistent_wind_preview_func(current_freestream_measurements, time_step):
		wind_preview_data = defaultdict(list)
		for j in range(input_dict["controller"]["n_horizon"] + 1):
			wind_preview_data[f"FreestreamWindMag_{j}"] += [wind_mag_ts[time_step]]
			wind_preview_data[f"FreestreamWindDir_{j}"] += [wind_dir_ts[time_step]]
			wind_preview_data[f"FreestreamWindSpeedU_{j}"] += [wind_u_ts[time_step]]
			wind_preview_data[f"FreestreamWindSpeedV_{j}"] += [wind_v_ts[time_step]]
		return wind_preview_data
	
	def perfect_wind_preview_func(current_freestream_measurements, time_step):
		wind_preview_data = defaultdict(list)
		for j in range(input_dict["controller"]["n_horizon"] + 1):
			delta_k = j * int(input_dict["controller"]["dt"] // input_dict["dt"])
			wind_preview_data[f"FreestreamWindMag_{j}"] += [wind_mag_ts[time_step + delta_k]]
			wind_preview_data[f"FreestreamWindDir_{j}"] += [wind_dir_ts[time_step + delta_k]]
			wind_preview_data[f"FreestreamWindSpeedU_{j}"] += [wind_u_ts[time_step + delta_k]]
			wind_preview_data[f"FreestreamWindSpeedV_{j}"] += [wind_v_ts[time_step + delta_k]]
		return wind_preview_data
	
	idx = 0
	current_freestream_measurements = [
		wind_mag_ts[idx] * np.sin((wind_dir_ts[idx] - 180.) * (np.pi / 180.)),
		wind_mag_ts[idx] * np.cos((wind_dir_ts[idx] - 180.) * (np.pi / 180.))
    ]

	n_time_steps = (input_dict["controller"]["n_horizon"] + 1) * int(input_dict["controller"]["dt"] // input_dict["dt"])
	preview_dt = int(input_dict["controller"]["dt"] // input_dict["dt"])

	tmp = perfect_wind_preview_func(current_freestream_measurements, idx)
	perfect_preview = {}
	perfect_preview["Sample"] = [1] * 2 * n_time_steps
	perfect_preview["Wind Speed"] \
		= np.concatenate([tmp[f"FreestreamWindSpeedU_{j}"] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for j in range(input_dict["controller"]["n_horizon"] + 1)] \
		+ [tmp[f"FreestreamWindSpeedV_{j}"] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
	 							  for j in range(input_dict["controller"]["n_horizon"] + 1)])
	perfect_preview["Wind Component"] = ["U" for j in range(n_time_steps)] + ["V" for j in range(n_time_steps)]

	tmp = persistent_wind_preview_func(current_freestream_measurements, idx)
	persistent_preview = {}
	persistent_preview["Sample"] = [1] * 2 * n_time_steps
	persistent_preview["Wind Speed"] \
		= np.concatenate([tmp[f"FreestreamWindSpeedU_{j}"] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for j in range(input_dict["controller"]["n_horizon"] + 1)] \
		+ [tmp[f"FreestreamWindSpeedV_{j}"] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
	 							  for j in range(input_dict["controller"]["n_horizon"] + 1)])
	persistent_preview["Wind Component"] = ["U" for j in range(n_time_steps)] + ["V" for j in range(n_time_steps)]

	tmp = stochastic_wind_preview_func(current_freestream_measurements, idx)
	stochastic_preview = {}
	stochastic_preview["Sample"] = np.repeat(np.arange(input_dict["controller"]["n_wind_preview_samples"]) + 1, (2 * (n_time_steps),))
	# stochastic_preview["FreestreamWindSpeedU"] = [tmp[f"FreestreamWindSpeedU_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(input_dict["controller"]["n_horizon"] + 1)]
	# stochastic_preview["FreestreamWindSpeedV"] = [tmp[f"FreestreamWindSpeedV_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"])for j in range(input_dict["controller"]["n_horizon"] + 1)]
	# stochastic_preview["Wind Speed"] = [tmp[f"FreestreamWindSpeedU_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(n_time_steps)] \
	# 	+ [tmp[f"FreestreamWindSpeedV_{j}"][m] for m in range(input_dict["controller"]["n_wind_preview_samples"]) for j in range(n_time_steps)]
	stochastic_preview["Wind Speed"] \
		= np.concatenate([np.concatenate([[tmp[f"FreestreamWindSpeedU_{j}"][m]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								   for j in range(input_dict["controller"]["n_horizon"] + 1)] \
		+ [[tmp[f"FreestreamWindSpeedV_{j}"][m]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
	 							    for j in range(input_dict["controller"]["n_horizon"] + 1)]) for m in range(input_dict["controller"]["n_wind_preview_samples"])])
	
	stochastic_preview["Wind Component"] = np.concatenate([["U" for j in range(n_time_steps)] + ["V" for j in range(n_time_steps)] for m in range(input_dict["controller"]["n_wind_preview_samples"])])
	# stochastic_preview = pd.DataFrame(stochastic_preview)

	perfect_preview["Time"] = persistent_preview["Time"] = np.tile(np.arange(n_time_steps), (2, )) *  input_dict["dt"]
	
	stochastic_preview["Time"] = np.tile(np.arange(n_time_steps) * input_dict["dt"], (2 * input_dict["controller"]["n_wind_preview_samples"],))
	
	perfect_preview = pd.DataFrame(perfect_preview)
	perfect_preview["Data Type"] = ["Preview"] * len(perfect_preview.index)
	tmp = pd.DataFrame(perfect_preview)
	tmp["Data Type"] = ["True"] * len(tmp.index)
	# tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
	tmp["Wind Speed"] \
		= np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)] \
		+ [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)])
	perfect_preview = pd.concat([perfect_preview, tmp])

	persistent_preview = pd.DataFrame(persistent_preview)
	persistent_preview["Data Type"] = ["Preview"] * len(persistent_preview.index)
	tmp = pd.DataFrame(persistent_preview)
	tmp["Data Type"] = ["True"] * len(tmp.index)
	# tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
	tmp["Wind Speed"] \
		= np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)] \
		+ [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)])
	persistent_preview = pd.concat([persistent_preview, tmp])

	stochastic_preview = pd.DataFrame(stochastic_preview)
	stochastic_preview["Data Type"] = ["Preview"] * len(stochastic_preview.index)
	tmp = pd.DataFrame(stochastic_preview.loc[stochastic_preview["Sample"] == 1])
	tmp["Data Type"] = ["True"] * len(tmp.index)
	# tmp["Wind Speed"] = [wind_u_ts[k] for k in range(n_time_steps)] + [wind_v_ts[k] for k in range(n_time_steps)]
	tmp["Wind Speed"] \
		= np.concatenate([[wind_u_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)] \
		+ [[wind_v_ts[k]] + [np.nan] * (int((input_dict["controller"]["dt"] - input_dict["dt"]) // input_dict["dt"])) 
								  for k in range(0, n_time_steps, preview_dt)])
	
	stochastic_preview = pd.concat([stochastic_preview, tmp])

	perfect_preview.reset_index(inplace=True, drop=True)
	persistent_preview.reset_index(inplace=True, drop=True)
	stochastic_preview.reset_index(inplace=True, drop=True)
	
	# stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Wind Component"] == "U"), "Wind Speed"].mean()
	# stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Sample"] == 1), "Time"]
	
	assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy() == persistent_preview.loc[persistent_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy())
	assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy() == stochastic_preview.loc[stochastic_preview["Data Type"] == "True", "Wind Speed"].dropna().to_numpy())
	assert np.all(perfect_preview.loc[perfect_preview["Data Type"] == "True", "Time"].dropna().to_numpy() == stochastic_preview.loc[stochastic_preview["Data Type"] == "True", "Time"].dropna().to_numpy())

	# perfect_preview["TrueWindSpeed"] = persistent_preview["TrueWindSpeed"] \
	# 	= [wind_u_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)] + [wind_v_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)]
	# stochastic_preview["TrueWindSpeed"] \
	# 	= np.tile([wind_u_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)] + [wind_v_ts[idx + j] for j in range(input_dict["controller"]["n_horizon"] + 1)], (input_dict["controller"]["n_wind_preview_samples"], ))

	# different hues for u vs k, different style for true vs preview
	# TODO
	fig = plt.figure()
	ax = sns.lineplot(data=perfect_preview.loc[perfect_preview["Data Type"] == "True", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
	ax = sns.lineplot(data=perfect_preview.loc[perfect_preview["Data Type"] == "Preview", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o")
	h, l = ax.get_legend_handles_labels()
	ax.legend(h[:5] + h[9:], l[:5] + l[9:])
	fig.savefig(os.path.join(wf.fig_dir, f'perfect_preview.png'))
	
	# plt.legend(labels=["Preview, U", "Preview, V", "True, U", "True, V"])
	
	fig = plt.figure()
	ax = sns.lineplot(data=persistent_preview.loc[persistent_preview["Data Type"] == "True", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
	ax = sns.lineplot(data=persistent_preview.loc[persistent_preview["Data Type"] == "Preview", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o")
	h, l = ax.get_legend_handles_labels()
	ax.legend(h[:5] + h[9:], l[:5] + l[9:])
	fig.savefig(os.path.join(wf.fig_dir, f'persistent_preview.png'))
	# sns.scatterplot(data=persistent_preview.loc[perfect_preview["Data Type"] == "Preview", :], x="Time", y="Wind Speed", zorder=7)
	# plt.legend(labels=["Preview, U", "Preview, V", "True, U", "True, V"])


	# stochastic_preview.loc[(stochastic_preview["Data Type"] == "Preview") & (stochastic_preview["Wind Component"] == "U"), "Wind Speed"].dropna()
	fig = plt.figure()
	# sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Sample"] == 1, :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4], [1, 0]])
	ax = sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Data Type"] == "True", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[1, 0]])
	ax = sns.lineplot(data=stochastic_preview.loc[stochastic_preview["Data Type"] == "Preview", :], x="Time", y="Wind Speed", hue="Wind Component", style="Data Type", dashes=[[4, 4]], marker="o")
	h, l = ax.get_legend_handles_labels()
	ax.legend(h[:5] + h[9:], l[:5] + l[9:])
	fig.savefig(os.path.join(wf.fig_dir, f'stochastic_preview.png'))
	# labels = ['Wind Component', 'U', 'V', 'Data Type', 'True', 'Preview']
	# handles = [h[labels.index(label)] for label in labels]
	# ax.legend(handles, labels)
	# plt.legend(labels=["Preview, U", "Preview, V", "True, U", "True, V"])

	# mean_true = np.load("mean_true.npy")
	# cov_true = np.load("cov_true.npy")
	# mean_preview = np.load("mean_preview.npy")
	# cov_preview = np.load("cov_preview.npy")

	# np.all(np.isclose(mean_true, mean_preview))
	# np.all(np.isclose(cov_true, cov_preview))

	# TODO consider that current_measurements is somewhere on the Gaussian distribution, can we draw from the skewed distribution centred on that value