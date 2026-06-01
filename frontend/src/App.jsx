import { useState } from "react";
import EmotionForm from "./components/EmotionForm";
import EmotionResult from "./components/EmotionResult";
import SongList from "./components/SongList";
import "./App.css";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="container">
      <h1>🎵 Emotion Music Recommender</h1>

      <EmotionForm setResult={setResult} />

      <EmotionResult result={result} />

      <SongList songs={result?.songs} />
    </div>
  );
}

export default App;