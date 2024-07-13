import React, { useRef, useState } from "react";
import axios from "axios";

const ImageAnalysis = ({ BACKEND_URL, setLoading }) => {
  const fileInputRef = useRef();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [linedImage, setLinedImage] = useState(null);
  const [labels, setLabels] = useState("");

  const imageUploadHandler = async () => {
    setLoading(true);
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/upload`,
        // "http://localhost:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );
      return response.data;
      // console.log();
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    }
  };

  const imageFromBackend = async (file_name) => {
    axios
      .get(`${BACKEND_URL}/image/${file_name}`, { responseType: "blob" })
      .then((response) => {
        const imageObjectURL = URL.createObjectURL(response.data);
        console.log("response.data", response.data);
        console.log("imageObjectURL", imageObjectURL);
        setLinedImage(imageObjectURL);
      })
      .catch((error) => {
        console.error("Error fetching image:", error);
      });
  };

  const imagePredictHandler = async (file_name, file_path, output_path) => {
    const formData = new FormData();
    formData.append("file_name", file_name);
    formData.append("file_path", file_path);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict/image`,
        // "http://localhost:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );
      await imageFromBackend(file_name);
      setLabels(response.data);
      setLoading(false);
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    } finally {
      setLoading(false);
    }
  };

  const imageChangeHandler = (e) => {
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

  // Implement other handlers and rendering logic related to image analysis

  return (
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
            onChange={imageChangeHandler}
            className="mb-4 p-2 border rounded"
            hidden
          />
          <button
            className="border py-2 px-4 mx-2 bg-slate-600 text-white rounded-md ml-auto"
            disabled={!selectedFile}
            onClick={async () => {
              const data = await imageUploadHandler();
              await imagePredictHandler(
                data.file_name,
                data.file_path,
                data.output_path
              );
            }}
          >
            제출
          </button>
        </div>
      </div>
      <img
        src={preview}
        // src="/placeholder.svg"
        className="max-w-full *:
            border-slate-600
            rounded-md
            w-full
            max-h-[28rem]
            min-h-[28rem]
            p-2
            object-contain
            mb-2
          "
      />
      {/* {labels && <div>결과물 labels: {labels}</div>} */}
      <hr />
      {linedImage && (
        <div className="w-full">
          <h1 className="text-3xl font-bold mb-4">예측 사진</h1>
          <img
            src={linedImage}
            // src="/placeholder.svg"
            className="max-w-full *:
            border-slate-600
            rounded-md
            w-full
            max-h-[28rem]
            min-h-[28rem]
            p-2
            object-contain
          "
          />
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;
