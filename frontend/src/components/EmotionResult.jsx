function EmotionResult({ result }) {
  if (!result) return null;

  return (
    <div className="result-card">
      <h2>Emotion Analysis</h2>

      <p>
        <strong>Face Emotion:</strong>{" "}
        {result.face_emotion}
      </p>

      <p>
        <strong>Face Confidence:</strong>{" "}
        {result.face_confidence.toFixed(2)}%
      </p>

      <p>
        <strong>Text Emotion:</strong>{" "}
        {result.text_emotion}
      </p>

      <p>
        <strong>Text Confidence:</strong>{" "}
        {result.text_confidence.toFixed(2)}%
      </p>

      <p>
        <strong>Final Emotion:</strong>{" "}
        {result.final_emotion}
      </p>
    </div>
  );
}

export default EmotionResult;