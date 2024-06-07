import React, { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [topK, setTopK] = useState(3);
  const [workType, setWorkType] = useState("");
  // Define state for other input fields similarly

  const handleButtonClick = () => {
    // Handle button click event here
    // You can perform computations similar to the Python code
    // Update state or display output accordingly
  };

  return (
    <div className="flex flex-col p-12">
      <h1 className="text-3xl font-bold mb-8">텍스트 모델 데모 시연</h1>
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
  );
}
export default App;
