from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import kmodes
import re
# plt 한글 출력 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# %% 데이터 준비
df_07_회원정보 = pd.read_parquet('./train/1.회원정보/201807_train_회원정보.parquet')
df_08_회원정보 = pd.read_parquet('./train/1.회원정보/201808_train_회원정보.parquet')
df_09_회원정보 = pd.read_parquet('./train/1.회원정보/201809_train_회원정보.parquet')
df_10_회원정보 = pd.read_parquet('./train/1.회원정보/201810_train_회원정보.parquet')
df_11_회원정보 = pd.read_parquet('./train/1.회원정보/201811_train_회원정보.parquet')
df_12_회원정보 = pd.read_parquet('./train/1.회원정보/201812_train_회원정보.parquet')
df_07_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201807_train_승인매출정보.parquet')
df_08_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201808_train_승인매출정보.parquet')
df_09_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201809_train_승인매출정보.parquet')
df_10_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201810_train_승인매출정보.parquet')
df_11_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201811_train_승인매출정보.parquet')
df_12_승인매출정보 = pd.read_parquet('./train/3.승인매출정보/201812_train_승인매출정보.parquet')
df_07_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201807_train_청구정보.parquet')
df_08_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201808_train_청구정보.parquet')
df_09_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201809_train_청구정보.parquet')
df_10_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201810_train_청구정보.parquet')
df_11_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201811_train_청구정보.parquet')
df_12_청구입금정보 = pd.read_parquet('./train/4.청구입금정보/201812_train_청구정보.parquet')
# train 폴더의 parquet 파일 로드
print(df_07_회원정보.shape)
print(df_07_승인매출정보.shape)
print(df_07_청구입금정보.shape)

# %% 분석에 사용할 컬럼만 추출
def select_columns(dfs, cols):
    return [df[cols].copy() for df in dfs]
dfs_by_category = {
    "회원정보": [df_07_회원정보, df_08_회원정보, df_09_회원정보, df_10_회원정보, df_11_회원정보, df_12_회원정보],
    "승인매출정보": [df_07_승인매출정보, df_08_승인매출정보, df_09_승인매출정보, df_10_승인매출정보, df_11_승인매출정보, df_12_승인매출정보],
    "청구입금정보": [df_07_청구입금정보, df_08_청구입금정보, df_09_청구입금정보, df_10_청구입금정보, df_11_청구입금정보, df_12_청구입금정보],
}
cols_by_category = {
    "회원정보": ['기준년월', 'ID', '남녀구분코드', '연령', '회원여부_이용가능', '소지카드수_이용가능_신용',
           '회원여부_연체', '가입통신회사코드', '거주시도명', '직장시도명', '마케팅동의여부',
           '이용카드수_신용체크', '이용카드수_신용', '이용카드수_신용_가족', '이용카드수_체크',
           '이용카드수_체크_가족', '이용금액_R3M_신용체크', '이용금액_R3M_신용', '이용금액_R3M_신용_가족',
           '이용금액_R3M_체크', '이용금액_R3M_체크_가족', '_1순위카드이용금액', '_1순위카드이용건수',
            '_1순위신용체크구분', '_2순위카드이용금액', '_2순위카드이용건수', '_2순위신용체크구분',
            '이용여부_3M_해외겸용_본인', '이용여부_3M_해외겸용_신용_본인',
            '연회비발생카드수_B0M', '기본연회비_B0M', '제휴연회비_B0M', 'Life_Stage'],
    "승인매출정보": ['기준년월', 'ID', '이용건수_신용_B0M', '이용건수_신판_B0M', '이용건수_일시불_B0M',
            '이용건수_할부_B0M', '이용건수_할부_유이자_B0M', '이용건수_할부_무이자_B0M',
            '이용건수_부분무이자_B0M', '이용건수_CA_B0M', '이용건수_체크_B0M',
            '이용건수_카드론_B0M', '이용금액_일시불_B0M', '이용금액_할부_B0M',
            '이용금액_할부_유이자_B0M', '이용금액_할부_무이자_B0M', '이용금액_부분무이자_B0M',
            '이용금액_CA_B0M', '이용금액_체크_B0M', '이용금액_카드론_B0M', '이용가맹점수',
            '이용금액_해외', '쇼핑_도소매_이용금액', '쇼핑_백화점_이용금액',
            '쇼핑_마트_이용금액', '쇼핑_슈퍼마켓_이용금액', '쇼핑_편의점_이용금액',
            '쇼핑_아울렛_이용금액', '쇼핑_온라인_이용금액', '쇼핑_기타_이용금액',
            '교통_주유이용금액', '교통_정비이용금액', '교통_통행료이용금액',
            '교통_버스지하철이용금액', '교통_택시이용금액', '교통_철도버스이용금액',
            '여유_운동이용금액', '여유_Pet이용금액', '여유_공연이용금액', '여유_공원이용금액',
            '여유_숙박이용금액', '여유_여행이용금액', '여유_항공이용금액', '여유_기타이용금액',
            '납부_통신비이용금액', '납부_관리비이용금액', '납부_렌탈료이용금액',
            '납부_가스전기료이용금액', '납부_보험료이용금액', '납부_유선방송이용금액',
            '납부_건강연금이용금액', '납부_기타이용금액', '_1순위업종', '_1순위업종_이용금액',
            '_2순위업종', '_2순위업종_이용금액', '_3순위업종', '_3순위업종_이용금액',
            '_1순위쇼핑업종', '_1순위쇼핑업종_이용금액', '_2순위쇼핑업종',
            '_2순위쇼핑업종_이용금액', '_3순위쇼핑업종', '_3순위쇼핑업종_이용금액',
            '_1순위교통업종', '_1순위교통업종_이용금액', '_2순위교통업종',
            '_2순위교통업종_이용금액', '_3순위교통업종', '_3순위교통업종_이용금액',
            '_1순위여유업종', '_1순위여유업종_이용금액', '_2순위여유업종',
            '_2순위여유업종_이용금액', '_3순위여유업종', '_3순위여유업종_이용금액',
            '_1순위납부업종', '_1순위납부업종_이용금액', '_2순위납부업종',
            '_2순위납부업종_이용금액', '_3순위납부업종', '_3순위납부업종_이용금액',
            'RP금액_B0M', 'RP건수_통신_B0M', 'RP건수_아파트_B0M',
            'RP건수_제휴사서비스직접판매_B0M', 'RP건수_렌탈_B0M', 'RP건수_가스_B0M',
            'RP건수_전기_B0M', 'RP건수_보험_B0M', 'RP건수_학습비_B0M', 'RP건수_유선방송_B0M',
            'RP건수_건강_B0M', 'RP건수_교통_B0M', '이용금액_온라인_B0M',
            '이용금액_오프라인_B0M', '이용건수_온라인_B0M', '이용건수_오프라인_B0M',
            '이용금액_페이_온라인_B0M', '이용금액_페이_오프라인_B0M',
            '이용건수_페이_온라인_B0M', '이용건수_페이_오프라인_B0M',
            '이용금액_간편결제_B0M', '이용건수_간편결제_B0M'],
    "청구입금정보": ['기준년월', 'ID', '포인트_마일리지_건별_B0M', '포인트_포인트_건별_B0M',
            '할인건수_B0M', '할인금액_B0M', '혜택수혜금액'],
}
dfs_customer = select_columns(dfs_by_category['회원정보'], cols_by_category['회원정보'])
df_07_회원정보, df_08_회원정보, df_09_회원정보, df_10_회원정보, df_11_회원정보, df_12_회원정보 = dfs_customer
dfs_sales = select_columns(dfs_by_category['승인매출정보'], cols_by_category['승인매출정보'])
df_07_승인매출정보, df_08_승인매출정보, df_09_승인매출정보, df_10_승인매출정보, df_11_승인매출정보, df_12_승인매출정보 = dfs_sales
dfs_charge = select_columns(dfs_by_category['청구입금정보'], cols_by_category['청구입금정보'])
df_07_청구입금정보, df_08_청구입금정보, df_09_청구입금정보, df_10_청구입금정보, df_11_청구입금정보, df_12_청구입금정보 = dfs_charge
print(df_07_회원정보.shape)
print(df_07_승인매출정보.shape)
print(df_07_청구입금정보.shape)

# %% 월별로 데이터 병합 후 CSV로 저장
df_07 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_07_회원정보, df_07_승인매출정보, df_07_청구입금정보])
df_08 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_08_회원정보, df_08_승인매출정보, df_08_청구입금정보])
df_09 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_09_회원정보, df_09_승인매출정보, df_09_청구입금정보])
df_10 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_10_회원정보, df_10_승인매출정보, df_10_청구입금정보])
df_11 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_11_회원정보, df_11_승인매출정보, df_11_청구입금정보])
df_12 = reduce(lambda x, y : pd.merge(x, y, on=['기준년월', 'ID'], how='left'), [df_12_회원정보, df_12_승인매출정보, df_12_청구입금정보])
df_all = pd.concat([df_07, df_08, df_09, df_10, df_11, df_12], axis = 0)
print(df_07.shape)
print(df_all.shape)
print(df_all.head())
df_07.to_csv('./merged/merged_07.csv', encoding = 'utf-8-sig', index=False)
df_08.to_csv('./merged/merged_08.csv', encoding = 'utf-8-sig', index=False)
df_09.to_csv('./merged/merged_09.csv', encoding = 'utf-8-sig', index=False)
df_10.to_csv('./merged/merged_10.csv', encoding = 'utf-8-sig', index=False)
df_11.to_csv('./merged/merged_11.csv', encoding = 'utf-8-sig', index=False)
df_12.to_csv('./merged/merged_12.csv', encoding = 'utf-8-sig', index=False)
df_all.to_csv('./merged/merged_all.csv', encoding = 'utf-8-sig', index=False)

# %% CSV 로드 확인
df_07 = pd.read_csv('./merged/merged_07.csv', encoding = 'utf-8-sig')
df_08 = pd.read_csv('./merged/merged_08.csv', encoding = 'utf-8-sig')
df_09 = pd.read_csv('./merged/merged_09.csv', encoding = 'utf-8-sig')
df_10 = pd.read_csv('./merged/merged_10.csv', encoding = 'utf-8-sig')
df_11 = pd.read_csv('./merged/merged_11.csv', encoding = 'utf-8-sig')
df_12 = pd.read_csv('./merged/merged_12.csv', encoding = 'utf-8-sig')
df_all = pd.read_csv('./merged/merged_all.csv', encoding = 'utf-8-sig')
print(df_all.head())
print(df_all.shape)
print(df_all['기준년월'].value_counts())

# %% 탐색적 데이터 분석
print(df_07['남녀구분코드'].value_counts())
print(df_07['연령'].value_counts())
# 표본의 개수는 총 40만개, 남녀 비율은 거의 동일, 연령의 경우 40대가 가장 많았으며 40대에서 멀어질수록 비율이 감소했음
print(df_07['거주시도명'].value_counts())
print(df_07['직장시도명'].value_counts())
# 거주시도명 및 직장시도명의 경우 절반 이상이 수도권
# 전국 인구 분포와 비교시 서울/대전/충북 과대표집, 비수도권 지역 대체로 과소표집
print(df_07['회원여부_이용가능'].value_counts())
print(df_07['회원여부_연체'].value_counts())
# 블랙리스트 등재 고객은 표본의 최대 약 5%, 연체 고객은 표본의 최대 약 2% 수준
# 블랙리스트 등재 또는 연체 고객에게는 추천 신용카드를 출력하지 않는등 신용도를 고려한 추천 시스템 구축 필요
# %%
print(df_07['이용카드수_신용'].describe())
print(df_07['이용카드수_체크'].describe())
print(df_07['이용금액_R3M_신용'].describe())
print(df_07['이용금액_R3M_체크'].describe())
# %%
print(df_07['기본연회비_B0M'].describe())
# 대체로 이용하는 신용카드의 수가 체크카드의 수보다 많았음
# 대다수(Q1 ~ Q3) 고객은 신용카드 1~2개, 체크카드 0개를 이용했음
# 연회비 무료 카드가 대다수, 기존 카드의 연회비 고려하여 신용카드 추천하여야 함
# %%
print(df_07['Life_Stage'].value_counts())
# 생애주기의 경우 자녀성장기 - 자녀출산기 - 가족구축기 순서로 비율이 높았음

# %% 히스토그램 시각화 함수
def visualize_histogram(col, title, ylabel='고객수', ylim_lower=10000, ylim_upper=100000, color='orange', hue=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # `hue` 옵션 사용을 위한 조건부 설정
    plot_kwargs = {'bins': 100}
    if hue:
        plot_kwargs['hue'] = hue
    else:
        plot_kwargs['color'] = color

    # `ylim_lower`와 `ylim_upper`가 모두 0인 경우, 하나의 그래프만 표시
    if ylim_lower == 0 and ylim_upper == 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df_07, x=col, **plot_kwargs)
        ax = plt.gca()
        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.show()
        return

    # 두 개의 서브플롯 생성 (기존 로직)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 4]}, figsize=(8, 6))

    # 상단 서브플롯 (ax1)
    sns.histplot(data=df_07, x=col, **plot_kwargs, ax=ax1)
    ax1.set_ylim(ylim_upper, )
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # 하단 서브플롯 (ax2)
    sns.histplot(data=df_07, x=col, **plot_kwargs, ax=ax2)
    ax2.set_ylim(0, ylim_lower)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    # 축 생략을 위한 물결표(대각선) 추가
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # 전체 플롯에 제목과 라벨 설정
    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.04, col, ha='center', fontsize=12)
    fig.text(0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=12)

    # 레이아웃 조정 및 저장
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

# %% 신용카드 및 체크카드 이용액 분포 시각화
visualize_histogram('이용금액_R3M_신용', '고객 신용카드 이용액 분포', '고객수', 25000, 50000, 'b')
visualize_histogram('이용금액_R3M_체크', '고객 체크카드 이용액 분포', '고객수', 10000, 30000, 'skyblue')
# 신용카드의 경우 최근 3개월간 이용금액이 0원인 경우가 최빈값이었으며 이를 제외하더라도 오른꼬리 분포를 이룸
# 사용자군에 따른 심리적/사회학적 특성 고려하여 추천 카드의 종류 달리해야
# 체크카드 사용금액이 매우 낮으나 신용카드 이용액인 0인 고객의 존재를 생각해보았을때 신규 신용카드 추천 보조 지표로 활용

# %% 1순위카드 및 2순위카드 시각화
visualize_histogram('_1순위카드이용금액', '고객 1순위카드 이용액 분포', '고객수', 25000, 50000, 'steelblue')
visualize_histogram('_2순위카드이용금액', '고객 2순위카드 이용액 분포', '고객수', 10000, 30000, 'cyan')
print(df_07['_1순위카드이용금액'].describe())
print(df_07['_2순위카드이용금액'].describe())
print(df_07['_1순위신용체크구분'].value_counts())
print(df_07['_2순위신용체크구분'].value_counts())
# 1순위 카드는 신용카드가 압도적이나 2순위 카드는 체크카드의 비중이 다소 있었음
# 주 카드로 신용카드를 사용하고 체크카드를 보조적으로 사용하는 경향성 존재

# %% 남녀별 소비패턴 차이
visualize_histogram('이용건수_신판_B0M', '남녀별 신용카드 이용건수 분포', '고객수', 0, 0, 'g', '남녀구분코드')
visualize_histogram('이용건수_체크_B0M', '남녀별 체크카드 이용건수 분포', '고객수', 2000, 150000, 'g', '남녀구분코드')
visualize_histogram('이용금액_일시불_B0M', '남녀별 신용카드 이용금액 분포', '고객수', 8000, 40000, 'g', '남녀구분코드')
visualize_histogram('이용금액_체크_B0M', '남녀별 체크카드 이용금액 분포', '고객수', 2000, 150000, 'g', '남녀구분코드')
print(df_07.groupby('남녀구분코드')['이용금액_신판_B0M'].describe())
print(df_07.groupby('남녀구분코드')['이용금액_체크_B0M'].describe())
# 1인 경우 이용건수가 더 많고 거래당 이용액은 차이가 없었음
def calculate_top10_average_usage(df_07):
    long_frames = []
    pairs = [('_1순위업종', '_1순위업종_이용금액'), ('_2순위업종', '_2순위업종_이용금액'), ('_3순위업종', '_3순위업종_이용금액'), ('_1순위쇼핑업종', '_1순위쇼핑업종_이용금액'), ('_2순위쇼핑업종', '_2순위쇼핑업종_이용금액'), ('_3순위쇼핑업종', '_3순위쇼핑업종_이용금액'), ('_1순위교통업종', '_1순위교통업종_이용금액'), ('_2순위교통업종', '_2순위교통업종_이용금액'), ('_3순위교통업종', '_3순위교통업종_이용금액'), ('_1순위여유업종', '_1순위여유업종_이용금액'), ('_2순위여유업종', '_2순위여유업종_이용금액'), ('_3순위여유업종', '_3순위여유업종_이용금액'), ('_1순위납부업종', '_1순위납부업종_이용금액'), ('_2순위납부업종', '_2순위납부업종_이용금액'), ('_3순위납부업종', '_3순위납부업종_이용금액')]

    for upjong_col, amt_col in pairs:
        tmp = df_07[['남녀구분코드', upjong_col, amt_col]].copy()
        tmp.columns = ['성별', '업종', '이용금액']
        long_frames.append(tmp)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=['이용금액'])
    long_df = long_df[long_df['이용금액'] > 0]

    avg_amt = long_df.groupby(['성별', '업종'], as_index=False)['이용금액'].mean()
    top10_each_gender = avg_amt.sort_values(['성별', '이용금액'], ascending=[True, False]).groupby('성별').head(10).copy()
    top10_each_gender['순위'] = top10_each_gender.groupby('성별')['이용금액'].rank(method='first', ascending=False).astype(int)

    male = top10_each_gender[top10_each_gender['성별'] == 1].sort_values('순위')[['순위', '업종', '이용금액']].reset_index(drop=True)
    female = top10_each_gender[top10_each_gender['성별'] == 2].sort_values('순위')[['순위', '업종', '이용금액']].reset_index(drop=True)
    combined = pd.concat([male, female], axis=1, keys=[1, 2])

    return combined
result_top10_average_usage = calculate_top10_average_usage(df_07)
print(result_top10_average_usage)
# 남녀코드별 상위 이용 업종 조회, 경향성은 유사하나 일부 차이점 존재

# %% 컬럼 별 왜도 및 0 비율 시각화
from scipy.stats import skew
def calculate_zero_ratio_and_skewness(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("수치형 변수가 없습니다.")
        return pd.DataFrame()

    results = []
    for col in numeric_cols:
        zero_ratio = (df[col] == 0).mean()
        skewness = skew(df[col].dropna())  # 결측값 제거 후 왜도 계산

        results.append({
            'column': col,
            'zero_ratio': zero_ratio,
            'skewness': skewness
        })

    result_df = pd.DataFrame(results).set_index('column')
    result_df['zero_ratio'] = result_df['zero_ratio'].round(4)
    result_df['skewness'] = result_df['skewness'].round(4)

    return result_df

result_zero_ratio_and_skewness = calculate_zero_ratio_and_skewness(df_07)
print(result_zero_ratio_and_skewness)
# 대다수 컬럼의 경우 0비율이 상당히 높았으며 왜도 또한 높았음