import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as mf
# 生成成交量K线

def vol_rebuild(data=None, vtype="usd", threshold=10000):
    """按照成交量累积进行重构
    """

    if vtype == "usd":
        # 累计量
        data["c"] = data["q_usd"].cumsum()
        # 标记
        data["f"] = data["c"] // threshold

        # 按照f进行aggregate
        tmp = data.groupby("f")

        # 每组的第一行为当根K线的close，需要调整
        c = tmp["f"].head(1) - 1
        data.f[c.index] = c

        # 再次按照f进行aggregate
        tmp = data.groupby("f")

        # 提取o/h/l/c数据
        c = (tmp.tail(1))["p"].reset_index(drop=True)
        o = (tmp.head(1))["p"].reset_index(drop=True)
        h = (tmp["p"].max()).reset_index(drop=True)
        l = (tmp["p"].min()).reset_index(drop=True)

        return pd.DataFrame({"o": o, "h": h, "l": l, "c": c})

    if vtype == "btc":
        # 累计量
        data["c"] = data["q_btc"].cumsum()
        # 标记
        data["f"] = data["c"] // threshold

        # 按照f进行aggregate
        tmp = data.groupby("f")

        # 每组的第一行为当根K线的close，需要调整
        c = tmp["f"].head(1) - 1
        data.f[c.index] = c

        # 再次按照f进行aggregate
        tmp = data.groupby("f")

        # 提取o/h/l/c数据
        c = (tmp.tail(1))["p"].reset_index(drop=True)
        o = (tmp.head(1))["p"].reset_index(drop=True)
        h = (tmp["p"].max()).reset_index(drop=True)
        l = (tmp["p"].min()).reset_index(drop=True)

        return pd.DataFrame({"o": o, "h": h, "l": l, "c": c})


if __name__ == "__main__":

    name = ["deribit_trade_perpetual.BTC.20191201.csv",
            "deribit_trade_perpetual.BTC.20191202.csv",
            "deribit_trade_perpetual.BTC.20191203.csv",
            "deribit_trade_perpetual.BTC.20191204.csv",
            "deribit_trade_perpetual.BTC.20191205.csv"]

    data = None

    for i in range(0, len(name)):
        path = "C:\\Users\\DELL\\Desktop\\ml\\ml\\data_pretreat\\trade_data\\" + name[i]

        if data is None:
            print("Sss")
            data = pd.read_csv(path, usecols=["p", "q", "a"])
        else:
            print("ttt")
            d = pd.read_csv(path, usecols=["p", "q", "a"])
            data = pd.concat([data, d])

    data = data.reset_index()
    print(data.describe())
    print(data.info())

    #
    # 修改列名，增加列。这里的数量单位包含USD和BTC
    data.rename(columns={"q": "q_usd"}, inplace=True)
    data["q_btc"] = round(data["q_usd"] / data["p"], 6)

    # 调整顺序
    data = data[["p", "q_usd", "q_btc", "a"]]

    # 统计信息
    print(data.describe())

    # 分两种方式研究，一种是累计q_usd，一种是累计q_btc
    # 设置 q_usd 累计阈值
    q_usd_vol_threshold = 500000

    data = vol_rebuild(data=data, vtype="usd", threshold=q_usd_vol_threshold)

    print(data.head(1500))

    fig, ax = plt.subplots(facecolor=(0, 0.3, 0.5), figsize=(12, 8))
    fig.subplots_adjust(bottom=0.1)
    # ax.xaxis_date()
    plt.xticks(rotation=45)  # 日期显示的旋转角度
    plt.title('BTC-PERPETUAL')
    plt.xlabel('time')
    plt.ylabel('price')

    mf.candlestick2_ochl(ax=ax, opens=data["o"], highs=data["h"], lows=data["l"], closes=data["c"],
                         colorup='r', colordown='green', width=0.5)

    plt.grid(False)

    plt.show()

