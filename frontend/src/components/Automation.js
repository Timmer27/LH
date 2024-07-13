import React, { useRef, useState } from "react";
import axios from "axios";

const Automation = ({BACKEND_URL, setLoading}) => {
  const scriptInput = useRef();
  const [tableData, setTableData] = useState([]);

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
        araHdqCd: "001"
      },
      inp: {
        flw_cts: textVal
      }
    };
    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict/automated`,
        jsonData,
        {
          headers: {
            "Content-Type": "application/json"
          }
        }
      );
      const data = response.data["result"];
      console.log("res", response.data);
      setTableData(data["일위대가"]);
      return true;
    } catch (error) {
      console.error("error", error);
    } finally {
      setLoading(false);
    }
  };

  return (
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
            setTableData([]);
            scriptInput.current.value = "";
          }}
          className="border p-2 rounded-md ml-3"
        >
          CLEAR
        </button>
      </div>
      {tableData.length !== 0 && (
        <table className="custom-table">
          <thead>
            <tr>
              <th>code</th>
              <th>unt_prc_e</th>
              <th>name</th>
              <th>unit_price</th>
              <th>unit_name</th>
              <th>unt_prc_qty</th>
              <th>unt_prc_stdd</th>
              <th>unt_prc_tc</th>
              <th>unt_prc_unt</th>
            </tr>
          </thead>
          <tbody>
            {tableData.map((item, index) => (
              <tr key={index}>
                <td>{item.unt_prc_cd}</td>
                <td>{item.unt_prc_e}</td>
                <td>{item.unt_prc_lc}</td>
                <td>{item.unt_prc_mc}</td>
                <td>{item.unt_prc_nm}</td>
                <td>{item.unt_prc_qty}</td>
                <td>{item.unt_prc_stdd}</td>
                <td>{item.unt_prc_tc}</td>
                <td>{item.unt_prc_unt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default Automation;
