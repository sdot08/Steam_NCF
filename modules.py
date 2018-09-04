from hyperparams import Hyperparams as hp


def ptime2cat(pt):
    if pt < hp.min_pt:
        return 0
    elif pt < hp.cutoff_pt:
        return 1
    else:
        return 2

def pt2sr(pt):
	cat = ptime2cat(pt)
	# convert confidence to sampling rates
	if cat == 0:
		return 1
	elif cat == 1:
		return 1 if np.random.random() < hp.sr1 else 0 #probability for sampling if the confidence is low
	else:
		return hp.sr3  #largest sampling rate