import pickle as pkl

trace_types = ['filezilla', 'spotifyOnline', 'VLC', 'skype', 'spotifyOffline', 'winscp', 'winrar']


with open("spotifyOnline.pkl", 'rb') as f:
    x= pkl.load(f)
    y=1