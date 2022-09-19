audio = ['data/pitch_tracks/Janani.csv',
'data/pitch_tracks/Karuna Nidhi Illalo.csv',
'data/pitch_tracks/Karunimpa Idi.csv',
'data/pitch_tracks/Koluvamaregatha.csv',
'data/pitch_tracks/Koti Janmani.csv',
'data/pitch_tracks/Mati Matiki.csv',
'data/pitch_tracks/Ninnuvina Marigalada.csv',
'data/pitch_tracks/Palisomma Muddu Sarade.csv',
'data/pitch_tracks/Rama Rama Guna Seema.csv',
'data/pitch_tracks/Ramabhi Rama Manasu.csv',
'data/pitch_tracks/Shankari Shankuru.csv',
'data/pitch_tracks/Sharanu Janakana.csv',
'data/pitch_tracks/Siddhi Vinayakam.csv',
'data/pitch_tracks/Sundari Nee Divya.csv',
'data/pitch_tracks/Vanajaksha Ninne Kori.csv']

stab = [
'data/stability_tracks/Janani.csv',
'data/stability_tracks/Karuna Nidhi Illalo.csv',
'data/stability_tracks/Karunimpa Idi.csv',
'data/stability_tracks/Koluvamaregatha.csv',
'data/stability_tracks/Koti Janmani.csv',
'data/stability_tracks/Mati Matiki.csv',
'data/stability_tracks/Ninnuvina Marigalada.csv',
'data/stability_tracks/Palisomma Muddu Sarade.csv',
'data/stability_tracks/Rama Rama Guna Seema.csv',
'data/stability_tracks/Ramabhi Rama Manasu.csv',
'data/stability_tracks/Shankari Shankuru.csv',
'data/stability_tracks/Sharanu Janakana.csv',
'data/stability_tracks/Siddhi Vinayakam.csv',
'data/stability_tracks/Sundari Nee Divya.csv',
'data/stability_tracks/Vanajaksha Ninne Kori.csv']

import os
from src.pitch import silence_stability_from_file
for a,s in zip(audio, stab):
	if os.path.isfile(a) and not os.path.isfile(s):
		print(a)
		silence_stability_from_file(a, s)



#silence_stability_from_file('./data/pitch_tracks/Lokavana Chatura.csv', './data/stability_tracks/Lokavana Chatura.csv')



