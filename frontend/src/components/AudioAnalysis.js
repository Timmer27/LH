import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

const AudioAnalysis = ({ BACKEND_URL, setLoading }) => {
  const fileInput = useRef(null); // Ref for file input element
  const [textLabels, setTextLabels] = useState([]);

  const fileModelSubmitHandler = async () => {
    setLoading(true);
    const audio = fileInput.current.files[0];

    if (!audio) {
      alert("파일을 선택해주세요");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("audio", audio);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict/audio`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      const data = response.data["result"];
      setTextLabels(data);
    } catch (error) {
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileInputChange = () => {
    fileModelSubmitHandler();
  };

  return (
    <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
      <div className="items-center justify-between w-full mb-4">
        <h1 className="text-3xl font-bold mb-8">오디오 파일 분석</h1>
        <div className="flex items-center justify-between w-full">
          <input
            ref={fileInput}
            type="file"
            accept=".wav, .mp3"
            className="hidden"
            onChange={handleFileInputChange}
          />
          <label
            htmlFor="audioFileUpload"
            className="border bg-slate-600 text-white rounded-md mb-6 p-2 cursor-pointer"
          >
            음성파일업로드
          </label>

          {/* <button
                        onClick={() => {
                            setTextLabels([]);
                            fileInput.current.value = null;
                        }}
                        className="border p-2 rounded-md ml-3"
                    >
                        CLEAR
                    </button> */}
        </div>
      </div>
      <div className="w-full">
        {textLabels.map((item, idx) => {
          const values = Object.values(item);
          const cd = values[0];
          const label = values[1];
          const prob = values[2];
          return (
            <div key={idx}>
              <div>
                {idx + 1}. {label} - 확률: {(prob * 100).toFixed(2)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
export default AudioAnalysis;
