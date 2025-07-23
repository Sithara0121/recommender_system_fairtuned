# spotify.py

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import time
import logging
from typing import Optional, List, Dict, Any
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyAPI:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str = None, scope: str = None):
        """
        Initialize Spotify API client
        
        Args:
            client_id: Your Spotify app's client ID
            client_secret: Your Spotify app's client secret
            redirect_uri: Required for user authentication (optional for client credentials)
            scope: Permissions needed for user authentication (optional for client credentials)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        
        # Choose authentication method based on parameters
        if redirect_uri and scope:
            # User authentication - allows access to user's private data
            self.auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope
            )
        else:
            # Client credentials - only public data access
            self.auth_manager = SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            )
        
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

    def get_track_info(self, track_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed track information
        
        Args:
            track_uri: Spotify track URI (e.g., 'spotify:track:6JsZyX0Emuzy7swG21tgt7')
            
        Returns:
            Dict containing track information or None if error
        """
        try:
            # Extract track ID from URI if needed
            track_id = self._extract_id_from_uri(track_uri)
            track = self.sp.track(track_id)
            return track
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching track info for {track_uri}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching track info for {track_uri}: {e}")
            return None

    def get_audio_features(self, track_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get audio features for a track
        
        Args:
            track_uri: Spotify track URI
            
        Returns:
            Dict containing audio features or None if error
        """
        try:
            track_id = self._extract_id_from_uri(track_uri)
            features = self.sp.audio_features([track_id])
            if features and features[0]:
                return features[0]
            else:
                logger.warning(f"No audio features found for {track_uri}")
                return None
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching audio features for {track_uri}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching audio features for {track_uri}: {e}")
            return None

    def get_artist_info(self, artist_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed artist information
        
        Args:
            artist_id: Spotify artist ID
            
        Returns:
            Dict containing artist information or None if error
        """
        try:
            artist_id = self._extract_id_from_uri(artist_id)
            artist = self.sp.artist(artist_id)
            return artist
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching artist info for {artist_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching artist info for {artist_id}: {e}")
            return None

    def get_artist_genres(self, artist_id: str) -> List[str]:
        """
        Get genres for an artist
        
        Args:
            artist_id: Spotify artist ID
            
        Returns:
            List of genre strings
        """
        artist = self.get_artist_info(artist_id)
        if artist and 'genres' in artist:
            return artist['genres']
        return []

    def search(self, query: str, search_type: str = 'track', limit: int = 10) -> Optional[Dict[str, Any]]:
        """
        Search for tracks, artists, albums, or playlists
        
        Args:
            query: Search query string
            search_type: Type of search ('track', 'artist', 'album', 'playlist')
            limit: Maximum number of results to return
            
        Returns:
            Dict containing search results or None if error
        """
        try:
            results = self.sp.search(q=query, type=search_type, limit=limit)
            return results
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error searching for '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error searching for '{query}': {e}")
            return None

    def get_album_tracks(self, album_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all tracks from an album
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            List of track dictionaries or None if error
        """
        try:
            album_id = self._extract_id_from_uri(album_id)
            results = self.sp.album_tracks(album_id)
            return results['items']
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching album tracks for {album_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching album tracks for {album_id}: {e}")
            return None

    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> Optional[List[Dict[str, Any]]]:
        """
        Get an artist's top tracks
        
        Args:
            artist_id: Spotify artist ID
            country: Country code for regional top tracks
            
        Returns:
            List of top tracks or None if error
        """
        try:
            artist_id = self._extract_id_from_uri(artist_id)
            results = self.sp.artist_top_tracks(artist_id, country=country)
            return results['tracks']
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching top tracks for {artist_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching top tracks for {artist_id}: {e}")
            return None

    def get_recommendations(self, seed_artists: List[str] = None, seed_tracks: List[str] = None, 
                          seed_genres: List[str] = None, limit: int = 20, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get track recommendations based on seed artists, tracks, or genres
        
        Args:
            seed_artists: List of artist IDs for recommendations
            seed_tracks: List of track IDs for recommendations
            seed_genres: List of genre strings for recommendations
            limit: Number of recommendations to return
            **kwargs: Additional audio feature parameters (e.g., target_danceability=0.8)
            
        Returns:
            Dict containing recommendations or None if error
        """
        try:
            # Clean up seed data
            if seed_artists:
                seed_artists = [self._extract_id_from_uri(artist_id) for artist_id in seed_artists]
            if seed_tracks:
                seed_tracks = [self._extract_id_from_uri(track_id) for track_id in seed_tracks]
            
            results = self.sp.recommendations(
                seed_artists=seed_artists,
                seed_tracks=seed_tracks,
                seed_genres=seed_genres,
                limit=limit,
                **kwargs
            )
            return results
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error getting recommendations: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting recommendations: {e}")
            return None

    def get_multiple_tracks(self, track_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Get information for multiple tracks at once
        
        Args:
            track_ids: List of track IDs or URIs
            
        Returns:
            List of track dictionaries or None if error
        """
        try:
            # Clean track IDs
            clean_ids = [self._extract_id_from_uri(track_id) for track_id in track_ids]
            
            # Spotify API limits to 50 tracks per request
            all_tracks = []
            for i in range(0, len(clean_ids), 50):
                batch = clean_ids[i:i+50]
                results = self.sp.tracks(batch)
                all_tracks.extend(results['tracks'])
            
            return all_tracks
        except spotipy.exceptions.SpotifyException as e:
            logger.error(f"Spotify API error fetching multiple tracks: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching multiple tracks: {e}")
            return None


    def get_multiple_audio_features(self, track_ids: List[str]) -> Optional[List[Dict[str, Any]]]:
     try:
        clean_ids = [self._extract_id_from_uri(track_id) for track_id in track_ids]
        all_features = []
        for i in tqdm(range(0, len(clean_ids), 100), desc='Fetching audio features'):
            batch = clean_ids[i:i+100]
            results = self.sp.audio_features(batch)
            all_features.extend(results)
        return all_features
     except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    def get_audio_features_sequential(self, track_ids: List[str]) -> List[Dict[str, Any]]:

     features_list = []
     for tid in tqdm(track_ids, desc="Fetching audio features sequentially"):
        try:
            feature = self.get_audio_features(tid)
            if feature:
                features_list.append(feature)
        except Exception as e:
            logger.error(f"Error fetching audio features for {tid}: {e}")
     return features_list


    def _extract_id_from_uri(self, uri_or_id: str) -> str:
        """
        Extract Spotify ID from URI or return ID if already clean
        
        Args:
            uri_or_id: Spotify URI or ID
            
        Returns:
            Clean Spotify ID
        """
        if ':' in uri_or_id:
            return uri_or_id.split(':')[-1]
        return uri_or_id

    def refresh_token(self):
        """
        Refresh the access token
        Note: SpotifyClientCredentials handles token refresh internally,
        but you can force refresh here if needed
        """
        if hasattr(self.auth_manager, 'refresh_access_token'):
            self.auth_manager.refresh_access_token()
        else:
            # For client credentials, create new auth manager
            self.auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        logger.info("Spotify access token refreshed")

    def is_token_expired(self) -> bool:
        """
        Check if the current token is expired
        
        Returns:
            True if token is expired, False otherwise
        """
        try:
            # Try a simple API call to test token validity
            self.sp.user('spotify')  # Get Spotify's official user profile
            return False
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 401:
                return True
            return False
        except Exception:
            return True

# Example usage with enhanced functionality:
if __name__ == "__main__":
    # Load credentials from environment variables
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'YOUR_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')

    # Initialize API
    spotify_api = SpotifyAPI(CLIENT_ID, CLIENT_SECRET)

    # Test track URI
    test_track = 'spotify:track:6JsZyX0Emuzy7swG21tgt7'
    
    # Get track info
    track_info = spotify_api.get_track_info(test_track)
    if track_info:
        print(f"Track: {track_info['name']} by {', '.join([artist['name'] for artist in track_info['artists']])}")
        print(f"Album: {track_info['album']['name']}")
        print(f"Release Date: {track_info['album']['release_date']}")
        print(f"Popularity: {track_info['popularity']}/100")

    # Get audio features
    audio_features = spotify_api.get_audio_features(test_track)
    if audio_features:
        print(f"\nAudio Features:")
        print(f"  Danceability: {audio_features['danceability']:.2f}")
        print(f"  Energy: {audio_features['energy']:.2f}")
        print(f"  Valence: {audio_features['valence']:.2f}")
        print(f"  Tempo: {audio_features['tempo']:.1f} BPM")

    # Get artist info and genres
    if track_info and 'artists' in track_info and len(track_info['artists']) > 0:
        artist_id = track_info['artists'][0]['id']
        genres = spotify_api.get_artist_genres(artist_id)
        print(f"\nArtist Genres: {', '.join(genres) if genres else 'No genres available'}")

    # Search for tracks
    search_results = spotify_api.search("Bohemian Rhapsody", "track", limit=5)
    if search_results:
        print(f"\nSearch Results for 'Bohemian Rhapsody':")
        for track in search_results['tracks']['items']:
            print(f"  - {track['name']} by {track['artists'][0]['name']}")

    # Get recommendations
    recommendations = spotify_api.get_recommendations(
        seed_tracks=[test_track],
        limit=5,
        target_danceability=0.7
    )
    if recommendations:
        print(f"\nRecommendations based on the test track:")
        for track in recommendations['tracks']:
            print(f"  - {track['name']} by {track['artists'][0]['name']}")

    # Check token status
    print(f"\nToken expired: {spotify_api.is_token_expired()}")