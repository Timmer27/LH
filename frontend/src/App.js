import React, { useReducer, useRef, useState } from "react";
import axios from "axios";
import Calendar from "react-calendar";
import "react-calendar/dist/Calendar.css"; // css import

function App() {
  // const BACKEND_URL = "http://localhost:5000";
  const BACKEND_URL = process.env.REACT_APP_URL;
  // const BACKEND_URL = "http://192.168.1.241:80/api";
  const [selectedFile, setSelectedFile] = useState(null);
  const textInput = useRef();
  const scriptInput = useRef();
  const textNum = useRef();
  const fileInputRef = useRef();
  // const BACKEND_URL = 'http://localhost:5000'
  const [preview, setPreview] = useState(null);
  const [linedImage, setLinedImage] = useState(null);
  const [labels, setLabels] = useState("");
  const [textLabels, setTextLabels] = useState([]);
  const [idx, setidx] = useState(0);

  const [formValues, setFormValues] = useState({
    spce_cd: "01",
    comp: "수전",
    tp_nm: "누수",
    flw_cts: "수전에서 물이새요",
    flwDsCd: "123456",
    flwDtlSn: "입주사사전방문(웰컴데이)",
  });

  const [datetime, setDatetime] = useState(new Date());
  const [workingPeriod, setWorkingPeriod] = useState({
    vst_fir_dt: [],
    vst_sec_dt: [],
    vst_thi_dt: [],
  });

  const [scriptLabels, setScriptLabels] = useState([]);

  // 로딩 state
  const [loading, setLoading] = useState(false);

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

  const inputChangeHandler = (event) => {
    const { id, value } = event.target;
    setFormValues({
      ...formValues,
      [id]: value,
    });
  };

  // datetime 중간 속해있는 날짜 전부 추출
  const generateDatesBetween = (startDate, endDate) => {
    const dates = [];
    const currentDate = new Date(startDate);

    while (currentDate <= endDate) {
      dates.push(new Date(currentDate));
      currentDate.setDate(currentDate.getDate() + 1);
    }

    return dates;
  };

  // string to datetime
  const convertStringtoDatetime = (dateString) => {
    // 아래 반드시 "20240101" 형식이어야함
    const year = parseInt(dateString.substring(0, 4));
    const month = parseInt(dateString.substring(4, 6)) - 1; // Months are 0-based in JS
    const day = parseInt(dateString.substring(6, 8));

    const date = new Date(year, month, day);
    return date;
  };

  // datetime에 날짜 더하는거
  const modifyDate = (datetime, daysCnt) => {
    const tmp = new Date(datetime);
    tmp.setDate(tmp.getDate() + daysCnt);
    return tmp;
  };

  // datetime 원하는 형식으로 변경
  const changeDatetimeFormat = (datetime) => {
    const newYear = datetime.getFullYear();
    const newMonth = String(datetime.getMonth() + 1).padStart(2, "0"); // Months are 0-based in JS
    const newDay = String(datetime.getDate()).padStart(2, "0");

    return `${newYear}-${newMonth}-${newDay}`;
  };
  // 위 함수들 통하여 date 파싱
  const processDate = (dateString, daysCnt) => {
    const startDate = convertStringtoDatetime(dateString);
    const endDate = modifyDate(startDate, daysCnt);

    const allDates = generateDatesBetween(startDate, endDate);
    const formattedDates = allDates.map((val) => changeDatetimeFormat(val));
    return formattedDates;
  };

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
        `${BACKEND_URL}/text/predict`,
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
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  const scheduleModelHandler = async () => {
    // console.log('textInput', textInput.current)
    // const textVal = textInput.current.value;
    // const textNumVal = textNum.current.value;
    setLoading(true);
    // if (!textVal) {
    //   alert("분석할 글을 입력해주세요");
    //   setLoading(false);
    //   return;
    // }
    const jsonData = {
      defect_info_detail: {
        flwDsCd: formValues.flwDsCd,
        flwDtlSn: formValues.flwDtlSn,
      },
      inp: {
        spce_cd: formValues.spce_cd,
        comp: formValues.comp,
        tp_nm: formValues.tp_nm,
        flw_cts: formValues.flw_cts,
      },
    };
    try {
      const response = await axios.post(
        `${BACKEND_URL}/schedule/predict`,
        jsonData,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = response.data["result"];
      const wrkCnt = data["inp_wrk_cnt"];

      setWorkingPeriod({
        vst_fir_dt: processDate(data["vst_fir_dt"], wrkCnt),
        vst_sec_dt: processDate(data["vst_sec_dt"], wrkCnt),
        vst_thi_dt: processDate(data["vst_thi_dt"], wrkCnt),
      });

      // setTextLabels(data);
      return true;
      // console.log();
    } catch (error) {
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  const automatedModelHandler = async () => {
    // console.log('textInput', textInput.current)
    const textVal = scriptInput.current.value;
    // const textNumVal = textNum.current.value;
    setLoading(true);
    if (!textVal) {
      alert("분석할 글을 입력해주세요");
      setLoading(false);
      return;
    }
    const jsonData = {
      defect_info_detail: {
        flwDsCd: "0123456",
        flwDtlSn: "입주사사전방문(웰컴데이)",
        araHdqCd: "001",
      },
      inp: {
        flw_cts: textVal,
      },
    };
    try {
      const response = await axios.post(
        `${BACKEND_URL}/automated/predict`,
        jsonData,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      const data = response.data["result"];
      console.log("res", response.data);
      return true;
    } catch (error) {
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  console.log("{process.env.REACT_APP_API_URL}", process.env.REACT_APP_API_URL);

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
 
  const imageFromBackend = async (file_name) => {
    axios.get(`${BACKEND_URL}/image/${file_name}`, { responseType: 'blob' })
      .then(response => {
        const imageObjectURL = URL.createObjectURL(response.data);
        console.log('response.data', response.data)
        console.log('imageObjectURL', imageObjectURL)
        setLinedImage(imageObjectURL);
      })
      .catch(error => {
        console.error('Error fetching image:', error);
      });
  }

  const imagePredictHandler = async (file_name, file_path, output_path) => {
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
      await imageFromBackend(file_name)
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

  const handleDateChange = (e) => {
    setDatetime(e);
  };

  function formatDate(dateStr) {
    const dateObj = new Date(dateStr);
    const year = dateObj.getFullYear();
    const month = String(dateObj.getMonth() + 1).padStart(2, "0"); // Months are zero-based
    const day = String(dateObj.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  // 화면 렌더링 
  return (
    <div className="flex min-h-screen w-full">
      <div className="flex flex-col gap-4 bg-muted p-4 bg-[#d7d7d7] min-w-60">
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
            alt={CalendarIcon}
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
              />
            </div>
          )}
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
      )}
      {idx === 2 && (
        <main className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
          <div className="items-center justify-between w-full mb-4">
            <h1 className="text-3xl font-bold mb-8">스케줄링 모델</h1>
            <div>
              <section>
                <form>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">공간코드:</label>
                    <input
                      onChange={inputChangeHandler}
                      id="spce_cd"
                      type="text"
                      className="border py-1 px-2 rounded-md min-w-80"
                      value={formValues.spce_cd}
                      disabled
                    />
                  </div>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">부위자재:</label>
                    <input
                      onChange={inputChangeHandler}
                      className="border py-1 px-2 rounded-md min-w-80"
                      id="comp"
                      type="text"
                      value={formValues.comp}
                    />
                  </div>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">AI 모델 예측 하자유형:</label>
                    <input
                      onChange={inputChangeHandler}
                      className="border py-1 px-2 rounded-md min-w-80"
                      id="tp_nm"
                      type="text"
                      value={formValues.tp_nm}
                    />
                  </div>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">하자내용:</label>
                    <textarea
                      onChange={inputChangeHandler}
                      className="border py-1 px-2 rounded-md min-w-80 min-h-36"
                      id="flw_cts"
                      type="text"
                      value={formValues.flw_cts}
                    />
                  </div>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">하자상세일련번호:</label>
                    <input
                      onChange={inputChangeHandler}
                      className="border py-1 px-2 rounded-md min-w-80"
                      id="flwDsCd"
                      type="text"
                      value={formValues.flwDsCd}
                      disabled
                    />
                  </div>
                  <div className="grid grid-cols-5 mb-2">
                    <label className="font-bold">하자구분코드:</label>
                    <input
                      onChange={inputChangeHandler}
                      className="border py-1 px-2 rounded-md min-w-80"
                      id="flwDtlSn"
                      type="text"
                      value={formValues.flwDtlSn}
                      disabled
                    />
                  </div>
                </form>
                <div className="mb-7">
                  <button
                    onClick={scheduleModelHandler}
                    className="border p-2 bg-slate-600 text-white rounded-md"
                  >
                    제출하기
                  </button>
                </div>
              </section>
              <hr className="mb-4" />
              <section>
                <header className="mb-4 flex gap-4">
                  <div className="flex items-center gap-2">
                    <div className="dot" />
                    <div className="">방문예정일자1</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="arrow" />
                    <div className="">방문예정일자2</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="square" />
                    <div className="">방문예정일자3</div>
                  </div>
                </header>
                <Calendar
                  onChange={handleDateChange}
                  // formatDay={(locale, date) => formatDate(date)}
                  value={datetime}
                  className="w-full text-sm border-b"
                  tileContent={({ date, view }) => {
                    // 날짜 타일에 컨텐츠 추가하기 (html 태그)
                    // 추가할 html 태그를 변수 초기화
                    let html = [];
                    // 현재 날짜가 post 작성한 날짜 배열(mark)에 있다면, dot div 추가
                    if (
                      workingPeriod.vst_fir_dt.find(
                        (x) => x === formatDate(date)
                      )
                    ) {
                      html.push(<div className="dot"></div>);
                    }
                    if (
                      workingPeriod.vst_sec_dt.find(
                        (x) => x === formatDate(date)
                      )
                    ) {
                      html.push(<div className="arrow"></div>);
                    }
                    if (
                      workingPeriod.vst_thi_dt.find(
                        (x) => x === formatDate(date)
                      )
                    ) {
                      html.push(<div className="square"></div>);
                    }
                    // 다른 조건을 주어서 html.push 에 추가적인 html 태그를 적용할 수 있음.
                    return (
                      <>
                        <div className="flex justify-center items-center absoluteDiv">
                          {html}
                        </div>
                      </>
                    );
                  }}
                />
              </section>
            </div>
          </div>
        </main>
      )}
      {idx === 3 && (
        <div className="flex flex-col items-center justify-center bg-background p-8 w-[60rem] min-w-[60rem]">
          <div className="items-center justify-between w-full mb-4">
            <h1 className="text-3xl font-bold mb-8">내역작성 자동화 기능</h1>
            <div>
              <textarea
                ref={scriptInput}
                className="border mb-6 p-2 w-full"
                placeholder="입력해주세요"
                defaultValue={"주방 상부장이 흔들려요"}
              />
            </div>
            <button
              onClick={automatedModelHandler}
              className="border p-2 bg-slate-600 text-white rounded-md"
            >
              제출하기
            </button>
            <button
              onClick={() => {
                setScriptLabels([]);
                scriptInput.current.value = "";
              }}
              className="border p-2 rounded-md ml-3"
            >
              CLEAR
            </button>
          </div>
          {/* <div className="w-full">
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
          </div> */}
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
