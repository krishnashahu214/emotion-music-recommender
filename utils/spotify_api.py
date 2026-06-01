import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

auth_manager = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
)

sp = spotipy.Spotify(auth_manager=auth_manager)

def get_songs_by_genre(genres):
    results = []

    for genre in genres:
        res = sp.search(q=genre, type='track', limit=5)

        for track in res['tracks']['items']:
            results.append({
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "url": track['external_urls']['spotify']
            })

    return results