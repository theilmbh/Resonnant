import pandas as pd
import numpy as np
import h5py as h5 
import os
import glob
from tqdm import tqdm
from neuraltda import SimplicialComplex as sc 
from neuraltda import topology as tp 

def runBrian():

	pass

def brian2ephys(brianResults):
	''' Converts Brian2 spikemonitor output over multiple trials to ephys-analysis format

	Contents of brianResults
	----------
	brianTrials : list
		list of brian2 spikemon objects, each element corresponding to each trial 
	stims : list
		list of stimuli names for each trial (length: ntrials)
	trialLen : float 
		Duration of a trial 
	nCluster : int 
		Number of neurons
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
	'''
	Takes an ephysDict and bins it 
	'''

	spikes = ephysDict['spikes']
	trials = ephysDict['trials']
	clusters = ephysDict['clusters']
	fs = ephysDict['fs']

	windt = binParams['windt']
	period = binParams['period']
	ncellsperm = binParams['ncellsperm']
	nperms = binParams['nperms']
	nshuffs = binParams['nshuffs']
	blockPath = binParams['blockPath']

	bfdict = tp.do_dag_bin(blockPath, spikes, trials, clusters, fs, windt, period, ncellsperm, nperms, nshuffs)
	return bfdict

def b2SCRecursive(dataGroup, dataIDstr, thresh, simpcomplist):
	if 'pop_vec' in dataGroup.keys():
		binaryMat = sc.binnedtobinary(dataGroup['pop_vec'], thresh)
		maxSimplices = sc.BinaryToMaxSimplex(binaryMat)
		if not maxSimplices:
			simpcomplist.append([])
		else:
			simpcomp = sc.SimplicialComplex(maxSimplices, name=dataIDstr)
			simpcomplist.append(simpcomp)
		
	else:
		for ind, perm in enumerate(dataGroup.keys()):
			dataIDstr = dataIDstr +'-%s' % perm 
			b2SCRecursive(dataGroup[perm], dataIDstr, thresh, simpcomplist)
	return

def brian2SimplicialComplex(brianResults, binParams, thresh, computationClass):
	'''
	Takes the results of a brian simulaton, bins it, and 
	creates simplicial complexes from the binned data

	Contents of brianResults
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

	Contents of binParams
	---------------------
	windt : float 
		window length in ms 
	period : dict 
		stimulus period dictionary {'period': 1}
	ncellsperm : int 
		number of cells in permutation 
	nperms : int 
		number of permutations 
	nshuffs : int 
		number of shuffles 
	blockPath : str 
		path to block for storing data 

	Parameters 
	----------
	thresh : float 
		multiple of avg firing rate to include in cell group 
	'''
	
	ephysDict = brian2ephys(brianResults)
	bfdict = ephys2binned(ephysDict, binParams)
	bfold = bfdict[computationClass]

	binnedFile = glob.glob(os.path.join(bfold, '*.binned'))[0]
	binnedData = h5.File(binnedFile)
	simpcomplist = []
	for stim in tqdm(binnedData.keys()):
		dataIDstr = stim 
		b2SCRecursive(binnedData[stim], dataIDstr, thresh, simpcomplist)

	return simpcomplist

def computeRenyiDivergence(sclist, targetDensity, dimension, beta):
	'''
	Computes renyi divergences between the simp comps in sclist 
	against the target density matrices.
	'''
	RD = np.zeros(len(sclist))
	for ind, sc in tqdm(enumerate(sclist)):
		sc.computeDensityMatrix(dimension, beta)
		RD[ind] = sc.computeRD(dimension, beta, targetDensity)

	return RD 
