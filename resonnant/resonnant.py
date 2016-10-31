import pandas as pd
import numpy as np
from neuraltda import SimplicialComplex as sc 
from neuraltda import topology as tp 

def runBrian():

	pass

def brian2ephys(brianResults):
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

	brianTrials = brianResults['brianTrials']
	stims = brianResults['stims']
	nClusters = brianResults['nClusters']
	trialLen = brianResults['trialLen']
	iti = brianResults['iti']
	fs = brianResults['fs']

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

	clusters = pd.DataFrame({'cluster': np.arange(nClusters), 'quality': nClusters*['Good'] })
	ephysDict = {'spikes': spikes, 'trials': trials, 'clusters': clusters, 'fs': fs}
	return ephysDict

def ephys2binned(ephysDict, binParams):

	spikes = ephysDict['spikes']
	trials = ephysDict['trials']
	clusters = ephysDict['clusters']
	fs = ephysDict['fs']

	windt = binParams['windt']
	period = binParams['period']
	ncellsperm = binParams['ncellsperm']
	nperms = binParams['nperms']
	nshuffs = binParams['nshuffs']

	bfdict = tp.do_dag_bin('./',spikes, trials, clusters, fs, windt, {'period': 1}, ncellsperm, nperms, nshuffs)
	return bfdict

def brian2SimplicialComplex(brianResults, binParams):
	
	ephysDict = brian2ephys(brianResults)
	bfdict = ephys2binned(ephysDict, binParams)
	bfold = bfdict['permuted']
	 
