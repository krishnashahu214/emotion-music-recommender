function SongList({ songs }) {
  if (!songs || songs.length === 0) return null;

  return (
    <div className="songs-container">
      <h2>Recommended Songs</h2>

      {songs.map((song, index) => (
        <div key={index} className="song-card">
          <h3>{song.name}</h3>

          <p>{song.artist}</p>

          <a
            href={song.url}
            target="_blank"
            rel="noreferrer"
          >
            Listen on Spotify
          </a>
        </div>
      ))}
    </div>
  );
}

export default SongList;