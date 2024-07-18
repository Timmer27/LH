import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

const TextAnalysis = ({ BACKEND_URL, setLoading }) => {
  const textInput = useRef();
  const [textLabels, setTextLabels] = useState([]);

  useEffect(() => {
    textInput.current.value = '문이 안열림'
  }, [])

  const textModelSubmitHandler = async () => {
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
        `${BACKEND_URL}/predict/text`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        }
      );
      const data = response.data["result"];
      setTextLabels(data);
      return true;
      // console.log();
    } catch (error) {
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
      <div className="items-center justify-between w-full mb-4">
        <h1 className="text-3xl font-bold mb-8">텍스트 파일 분석</h1>
        <div>
          <textarea
            ref={textInput}
            className="border mb-6 p-2 w-full"
            placeholder="입력해주세요"
          />
        </div>
        <button
          onClick={textModelSubmitHandler}
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
  );
};

export default TextAnalysis;
