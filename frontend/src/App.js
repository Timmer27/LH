import React, { useState } from "react";
import axios from "axios";

function App() {
  const BACKEND_URL = 'http://localhost:80/api'
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [labels, setLabels] = useState("");

  const [text, setText] = useState("");
  const [topK, setTopK] = useState(3);
  const [workType, setWorkType] = useState("");

  const [loading, setLoading] = useState(false); // State to manage loading spinner

  const handleButtonClick = () => {
    // Handle button click event here
    // You can perform computations similar to the Python code
    // Update state or display output accordingly
  };

  const handleUpload = async () => {
    setLoading(true);
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      console.log('process.env.REACT_BACKEND_URLprocess.env.REACT_BACKEND_URLprocess.env.REACT_BACKEND_URL', BACKEND_URL)
      const response = await axios.post(
        `${BACKEND_URL}/upload`,
        // "http://localhost:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
      // console.log();
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    }
  };

  const handlePredict = async (file_name, file_path) => {
    const formData = new FormData();
    formData.append("file_name", file_name);
    formData.append("file_path", file_path);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict`,
        // "http://localhost:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setLabels(response.data);
      setLoading(false);
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === "image/jpeg") {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert("Please select a JPEG image.");
      setPreview(null);
      return false;
    }
  };

  return (
    <div className="flex">
      <div className="flex flex-col p-12 flex-1">
        <h1 className="text-3xl font-bold mb-8">이미지 모델 데모 시연</h1>
        <label className="mb-2">detect.py 실행</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4 p-2 border rounded"
        />
        {preview && (
          <div className="mt-4">
            <img
              src={preview}
              alt="Preview"
              className="w-64 h-64 object-contain border rounded"
            />
          </div>
        )}
        <button
          disabled={!selectedFile}
          onClick={async () => {
            const data = await handleUpload();
            await handlePredict(data.file_name, data.file_path);
          }}
          className="border p-2 bg-slate-600 text-white rounded-md"
        >
          제출하기
        </button>
        {loading && (
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <div className="border-t-4 border-b-4 border-blue-500 rounded-full w-12 h-12 animate-spin"></div>
          </div>
        )}
        {labels && <div>result labels: {labels}</div>}
      </div>
      <div className="flex flex-col p-12 flex-1">
        <h1 className="text-3xl font-bold mb-8">
          텍스트 모델 데모 시연 (미완료)
        </h1>
        <label className="mb-2">스케줄링 모델을 사용합니다.</label>
        <textarea
          className="border mb-6 p-2"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="입력해주세요"
        />
        {/* Define other input fields */}
        <label className="mb-2">출력 값 개수</label>
        <input
          className="border mb-4 p-2"
          type="number"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
        />
        <button
          onClick={handleButtonClick}
          className="border p-2 bg-slate-600 text-white rounded-md"
        >
          제출하기
        </button>
        {/* Display output */}
      </div>
    </div>
  );
}
export default App;
