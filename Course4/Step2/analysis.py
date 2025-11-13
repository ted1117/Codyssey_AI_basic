import pandas as pd
import matplotlib.pyplot as plt


def read_csv(csv_path: str) -> pd.DataFrame:
    """CSV 파일 DataFrame 로딩"""
    df = pd.read_csv(csv_path)
    return df


def filter_column(df: pd.DataFrame) -> pd.DataFrame:
    """일반가구원 분석 컬럼 선별"""
    columns_to_drop = [
        col
        for col in df.columns
        if col not in ["일반가구원", "행정구역별(시군구)", "성별", "연령별", "시점"]
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def nums_by_gender_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 남녀 일반가구원 합계 피벗"""
    # 성별에서 '계' 걸러주기
    df = df[df["성별"].isin(["남자", "여자"])].pivot_table(
        values="일반가구원", index="시점", columns="성별", aggfunc="sum"
    )

    return df


def nums_by_ages(df: pd.DataFrame) -> pd.DataFrame:
    """연령대별 일반가구원 합계 산출(65세 이상 제외)"""
    df = (
        df[df["성별"] == "계"]
        .pivot_table(
            values="일반가구원",
            index="연령별",
            aggfunc="sum",
        )
        .drop(index="65세이상")
    )

    return df


def draw_graph(df: pd.DataFrame) -> None:
    """남녀 연령별 일반가구원 꺾은선 그래프"""
    df = df[df["성별"].isin(["남자", "여자"])]

    pivot = df.pivot_table(
        values="일반가구원", index="연령별", columns="성별", aggfunc="sum"
    ).drop(index="합계")
    pivot_scaled = pivot / 10_000

    plt.rcParams["font.family"] = "Apple SD Gothic Neo"

    pivot_scaled.plot(kind="line", marker="o", figsize=(10, 6))
    plt.title("남자 및 여자의 연령별 일반가구원 통계")
    plt.xlabel("연령대")
    plt.ylabel("일반가구원 (단위: 만 명)")
    plt.ylim(bottom=0)
    plt.legend(title="성별")
    plt.grid(True)

    plt.xticks(ticks=range(len(pivot.index)), labels=list(pivot.index), rotation=45)

    plt.show()


if __name__ == "__main__":
    csv_file = (
        "Course4/Step2/성__연령_및_가구주와의_관계별_인구__시군구_20251113024206.csv"
    )
    df = read_csv(csv_file)
    # print(df)
    df = filter_column(df)

    # 출력 1
    print()
    print("#" * 5 + "남자 및 여자의 연도별 일반가구원" + "#" * 5)
    print(nums_by_gender_per_year(df))

    # 출력 2
    print()
    print("#" * 5 + "연령별 일반가구원" + "#" * 5)
    print(nums_by_ages(df))

    # 그래프
    draw_graph(df)
