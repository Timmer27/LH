import React, { useState } from "react";
import Calendar from "react-calendar";
import axios from "axios";

const Scheduling = ({ BACKEND_URL, setLoading }) => {
  const [formValues, setFormValues] = useState({
    spce_cd: "01",
    comp: "수전",
    tp_nm: "누수",
    flw_cts: "수전에서 물이새요",
    flwDsCd: "123456",
    flwDtlSn: "입주사사전방문(웰컴데이)"
  });
  const [datetime, setDatetime] = useState(new Date());
  const [workingPeriod, setWorkingPeriod] = useState({
    vst_fir_dt: [],
    vst_sec_dt: [],
    vst_thi_dt: []
  });

  const inputChangeHandler = (event) => {
    const { id, value } = event.tdataarget;
    setFormValues({
      ...formValues,
      [id]: value
    });
  };

  const handleDateChange = (e) => {
    setDatetime(e);
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

  function formatDate(dateStr) {
    const dateObj = new Date(dateStr);
    const year = dateObj.getFullYear();
    const month = String(dateObj.getMonth() + 1).padStart(2, "0"); // Months are zero-based
    const day = String(dateObj.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

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
        flwDtlSn: formValues.flwDtlSn
      },
      inp: {
        spce_cd: formValues.spce_cd,
        comp: formValues.comp,
        tp_nm: formValues.tp_nm,
        flw_cts: formValues.flw_cts
      }
    };
    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict/schedule`,
        jsonData,
        {
          headers: {
            "Content-Type": "application/json"
          }
        }
      );
      const data = response.data["result"];
      const wrkCnt = data["inp_wrk_cnt"];

      setWorkingPeriod({
        vst_fir_dt: processDate(data["vst_fir_dt"], wrkCnt),
        vst_sec_dt: processDate(data["vst_sec_dt"], wrkCnt),
        vst_thi_dt: processDate(data["vst_thi_dt"], wrkCnt)
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

  return (
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
                  workingPeriod.vst_fir_dt.find((x) => x === formatDate(date))
                ) {
                  html.push(<div className="dot"></div>);
                }
                if (
                  workingPeriod.vst_sec_dt.find((x) => x === formatDate(date))
                ) {
                  html.push(<div className="arrow"></div>);
                }
                if (
                  workingPeriod.vst_thi_dt.find((x) => x === formatDate(date))
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
  );
};

export default Scheduling;
