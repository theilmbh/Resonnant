import pandas as pd
import numpy as np

def brian2ephys(brianTrials, stims, trialLen, iti, fs):
	''' Converts Brian2 spikemonitor output over multiple trials to ephys-analysis format

	Parameters
	----------
	brianTrials : list
		list of brian2 spikemon objects, each element corresponding to each trial 
	stims : list
		list of stimuli names for each trial (length: ntrials)
	trialLen : float 
		Duration of a trial 
	iti : float 
		Inter trial interval to space trials apart in dataFrames
	fs : int 
		fictious sampling rate
	'''

	ntrials = len(brianTrials)
	trialStarts = np.zeros(ntrials)

	spikes = pd.DataFrame()
	for trial in range(ntrials):
		trialStart =  np.round(trial*(trialLen + iti) * fs)
		trialStarts[trial] = trialStart
		rec = np.zeros(len(brianTrials[trial].i))
		newSpikes = pd.DataFrame({'cluster': np.array(brianTrials[trial].i), 
							      'time_samples': np.round(np.array(brianTrials[trial].t)*fs) + trialStart,
							      'recording': rec})
		spikes = spikes.append(newSpikes, ignore_index=True)
	
	trials = pd.DataFrame({'stimulus': stims,
		                   'time_samples': trialStarts,
		                   'stimulus_end': trialStarts + np.round(trialLen*fs)})
	return (spikes, trials)