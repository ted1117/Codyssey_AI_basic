import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기
train = pd.read_csv("spaceship-titanic/train.csv")
test = pd.read_csv("spaceship-titanic/test.csv")

# 2. 두 파일 병합
combined = pd.concat([train, test], ignore_index=True)

print(combined)

# 3. 전체 데이터 수량 출력
print(combined.shape)
print(type(combined.shape))
print("전체 데이터 수량:", combined.shape[0])

# 4. Transported와 관련성 높은 항목 찾기
# True/False를 1/0으로 변환
# train["Transported"] = train["Transported"].astype(int)

# one-hot encoding
encoded = pd.get_dummies(train)

# 변수와 Transported 간 상관관계 계산
corr = encoded.corrwith(encoded["Transported"]).sort_values(ascending=False)
print("\nTransported 항목과의 상관계수:")
print(corr)

# 5. 나이대 구간화 (10대 단위, 70대까지)
bins: list[int] = [0, 19, 29, 39, 49, 59, 69, 79]
labels: list[str] = ["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]
train["AgeGroup"] = pd.cut(train["Age"], bins=bins, labels=labels)

# 6. 연령대별 Transported 여부 집계
# https://wikidocs.net/216695
age_grouped = pd.crosstab(train["AgeGroup"], train["Transported"])
print(age_grouped)

# 7. 막대그래프 그리기
age_grouped.plot(kind="bar", stacked=False, color=["red", "blue"])
plt.title("Transported by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Transported", labels=["False", "True"])
plt.show()
