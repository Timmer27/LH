import React from "react";
import { HomeIcon, SettingsIcon, SearchIcon, CalendarIcon } from "./Icons.js";

const TABS = [
  { label: "이미지 분석 모델", icon: <HomeIcon className="h-5 w-5" /> },
  { label: "텍스트 분석 모델", icon: <SettingsIcon className="h-5 w-5" /> },
  { label: "스케줄링 기능", icon: <SearchIcon className="h-5 w-5" /> },
  { label: "내역작성 자동화 기능", icon: <CalendarIcon className="h-5 w-5" /> },
  { label: "음성파일분석", icon: <SettingsIcon className="h-5 w-5" /> },
];

const Tabs = ({ setIdx, idx }) => {
  return (
    <div className="flex flex-col gap-4 bg-muted p-4 bg-[#d7d7d7] min-w-60">
      {TABS.map((tab, index) => (
        <button
          key={index}
          className="flex justify-start gap-2 rounded-md px-3 py-2 text-left transition-colors hover:bg-muted/50"
          onClick={() => setIdx(index)}
          style={{
            fontWeight: idx === index && "bold",
            textDecoration: idx === index && "underline",
          }}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
};

export default Tabs;
