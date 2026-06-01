import { useState } from "react";
import axios from "axios";

function EmotionForm({ setResult }) {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL;

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert("Please select an image");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();

      formData.append("file", file);
      formData.append("text", text);

      const response = await axios.post(
        `${API_URL}/analyze`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (error) {
        console.log("FULL ERROR:", error);
        console.log("RESPONSE:", error.response?.data);

        alert(JSON.stringify(error.response?.data));
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  return (
    <form className="form-container" onSubmit={handleSubmit}>
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      {preview && (
        <img
          src={preview}
          alt="Preview"
          className="preview-image"
        />
      )}

      <textarea
        placeholder="Describe how you are feeling..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button type="submit" disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Emotion"}
      </button>
    </form>
  );
}

export default EmotionForm;