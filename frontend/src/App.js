import React, { useState } from "react";
import Tabs from "./components/Tabs.js";
import ImageAnalysis from "./components/ImageAnalysis";
import TextAnalysis from "./components/TextAnalysis";
import Scheduling from "./components/Scheduling";
import Automation from "./components/Automation";
import "react-calendar/dist/Calendar.css";

function App() {
  const BACKEND_URL = process.env.REACT_APP_URL || "http://localhost:5000";
  const [idx, setIdx] = useState(0); // State to manage active tab index
  // 로딩 state
  const [loading, setLoading] = useState(false);
  return (
    <div className="flex min-h-screen w-full">
      {loading && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="border-t-4 border-b-4 border-blue-500 rounded-full w-12 h-12 animate-spin"></div>
        </div>
      )}
      <Tabs setIdx={setIdx} idx={idx} />
      {idx === 0 && <ImageAnalysis setLoading={setLoading} BACKEND_URL={BACKEND_URL}/>}
      {idx === 1 && <TextAnalysis setLoading={setLoading} BACKEND_URL={BACKEND_URL}/>}
      {idx === 2 && <Scheduling setLoading={setLoading} BACKEND_URL={BACKEND_URL}/>}
      {idx === 3 && <Automation setLoading={setLoading} BACKEND_URL={BACKEND_URL}/>}
    </div>
  );
}

export default App;
