import React, { useReducer, useRef, useState } from "react";
import axios from "axios";

function App() {
  // const BACKEND_URL = "http://localhost:5000";
  const BACKEND_URL = "http://192.168.1.241:80/api";
  const [selectedFile, setSelectedFile] = useState(null);
  const textInput = useRef();
  const textNum = useRef();
  const fileInputRef = useRef();
  // const BACKEND_URL = 'http://localhost:5000'
  const [preview, setPreview] = useState(null);
  const [labels, setLabels] = useState("");
  const [textLabels, setTextLabels] = useState([]);
  const [idx, setidx] = useState(0);

  const [loading, setLoading] = useState(false); // State to manage loading spinner

  const TABS = [
    {
      label: "이미지 분석 모델",
      icon: <HomeIcon className="h-5 w-5" />,
    },
    {
      label: "텍스트 분석 모델",
      icon: <SettingsIcon className="h-5 w-5" />,
    },
    {
      label: "스케줄링 기능",
      icon: <SearchIcon className="h-5 w-5" />,
    },
    {
      label: "내역작성 자동화 기능",
      icon: <CalendarIcon className="h-5 w-5" />,
    },
  ];

  const handleTextUpload = async () => {
    // console.log('textInput', textInput.current)
    const textVal = textInput.current.value;
    // const textNumVal = textNum.current.value;
    setLoading(true);
    if (!textVal) {
      alert("분석할 글을 입력해주세요");
      setLoading(false);
      return;
    }
    const formData = new FormData();
    formData.append("text", textVal);
    // formData.append("num", textNumVal);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/text/predict`,
        // "http://localhost:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      const data = response.data["result"];
      setTextLabels(data);
      return true;
      // console.log();
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    } finally {
      setLoading(false);
    }
  };

  console.log("{process.env.REACT_APP_API_URL}", process.env.REACT_APP_API_URL);

  const handleUpload = async () => {
    setLoading(true);
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      console.log(
        "process.env.REACT_BACKEND_URLprocess.env.REACT_BACKEND_URLprocess.env.REACT_BACKEND_URL",
        BACKEND_URL
      );
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
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && ["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
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
    // <div className="flex">
    //   <div className="flex flex-col p-12 flex-1">
    //     <h1 className="text-3xl font-bold mb-8">이미지 모델 데모 시연</h1>
    //     <label className="mb-2">detect.py 실행</label>
    //     <input
    //       type="file"
    //       accept="image/*"
    //       onChange={handleImageChange}
    //       className="mb-4 p-2 border rounded"
    //     />
    //     {preview && (
    //       <div className="mt-4">
    //         <img
    //           src={preview}
    //           alt="Preview"
    //           className="w-64 h-64 object-contain border rounded"
    //         />
    //       </div>
    //     )}
    //     <button
    //       disabled={!selectedFile}
    //       onClick={async () => {
    //         const data = await handleUpload();
    //         await handlePredict(data.file_name, data.file_path);
    //       }}
    //       className="border p-2 bg-slate-600 text-white rounded-md"
    //     >
    //       제출하기
    //     </button>
    //     {loading && (
    //       <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
    //         <div className="border-t-4 border-b-4 border-blue-500 rounded-full w-12 h-12 animate-spin"></div>
    //       </div>
    //     )}
    //     {labels && <div>result labels: {labels}</div>}
    //   </div>
    //   <div className="flex flex-col p-12 flex-1">
    //     <h1 className="text-3xl font-bold mb-8">
    //       텍스트 모델 데모 시연 (미완료)
    //     </h1>
    //     <label className="mb-2">스케줄링 모델을 사용합니다.</label>
    //     <textarea
    //       ref={textInput}
    //       className="border mb-6 p-2"
    //       // value={text}
    //       // onChange={(e) => setText(e.target.value)}
    //       placeholder="입력해주세요"
    //     />
    //     {/* Define other input fields */}
    //     <label className="mb-2">출력 값 개수</label>
    //     <input
    //       ref={textNum}
    //       min={1}
    //       className="border mb-4 p-2"
    //       type="number"
    //       // value={topK}
    //       // onChange={(e) => setTopK(parseInt(e.target.value))}
    //     />
    //     <button
    //       onClick={handleTextUpload}
    //       className="border p-2 bg-slate-600 text-white rounded-md"
    //     >
    //       제출하기
    //     </button>
    //     {/* Display output */}
    //   </div>
    // </div>
    <div className="flex min-h-screen w-full">
      <div className="flex flex-col gap-4 bg-muted p-4 bg-[#d7d7d7] min-w-56">
        {TABS.map((val, index) => (
          <button
            className="flex justify-start gap-2 rounded-md px-3 py-2 text-left transition-colors hover:bg-muted/50"
            onClick={() => {
              setidx(index);
            }}
            style={{
              fontWeight: idx === index && "bold",
              textDecoration: idx === index && "underline",
            }}
          >
            {val.icon}
            {val.label}
          </button>
        ))}
      </div>
      {loading && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="border-t-4 border-b-4 border-blue-500 rounded-full w-12 h-12 animate-spin"></div>
        </div>
      )}
      {idx === 0 && (
        <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
          <div className="flex items-center justify-between w-full mb-4">
            <h1 className="text-3xl font-bold">이미지 파일 분석</h1>
            <div>
              <button
                className="border py-2 px-4 mx-2 bg-white rounded-md ml-auto"
                onClick={() => fileInputRef.current.click()}
              >
                업로드
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="mb-4 p-2 border rounded"
                hidden
              />
              <button
                className="border py-2 px-4 mx-2 bg-slate-600 text-white rounded-md ml-auto"
                disabled={!selectedFile}
                onClick={async () => {
                  const data = await handleUpload();
                  await handlePredict(data.file_name, data.file_path);
                }}
              >
                제출
              </button>
            </div>
          </div>
          <img
            src={preview}
            // src="/placeholder.svg"
            alt={CalendarIcon}
            className="max-w-full *:
            border-slate-600
            rounded-md
            w-full
            max-h-[28rem]
            min-h-[28rem]
            p-2
            object-contain
          "
            placeholder="이미지 업로드"
          />
          {labels && <div>결과물 labels: {labels}</div>}
        </div>
      )}
      {idx === 1 && (
        <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
          <div className="items-center justify-between w-full mb-4">
            <h1 className="text-3xl font-bold mb-8">텍스트 파일 분석</h1>
            <div>
              <textarea
                ref={textInput}
                className="border mb-6 p-2 w-full"
                // value={text}
                // onChange={(e) => setText(e.target.value)}
                placeholder="입력해주세요"
              />
              {/* Define other input fields */}
              {/* <label className="mb-2">출력 값 개수</label>
              <input
                ref={textNum}
                min={1}
                className="border mb-4 p-2 w-full"
                type="number"
                // value={topK}
                // onChange={(e) => setTopK(parseInt(e.target.value))}
              /> */}
            </div>
            <button
              onClick={handleTextUpload}
              className="border p-2 bg-slate-600 text-white rounded-md"
            >
              제출하기
            </button>
            <button
              onClick={() => {
                setTextLabels([]);
                textInput.current.value = "";
              }}
              className="border p-2 rounded-md ml-3"
            >
              CLEAR
            </button>
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
      )}
      {(idx === 2 || idx === 3) && (
        <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
          <div className="items-center justify-between w-full mb-4">
            <h1 className="text-3xl font-bold mb-8">
              미완성 (모델 {idx} 대기 중)
            </h1>
          </div>
        </div>
      )}
    </div>
  );
}
export default App;

function CalendarIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M8 2v4" />
      <path d="M16 2v4" />
      <rect width="18" height="18" x="3" y="4" rx="2" />
      <path d="M3 10h18" />
    </svg>
  );
}

function HomeIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function SearchIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function SettingsIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
