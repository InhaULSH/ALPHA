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


# %%
df_final_customer = pd.read_csv('all_customer.csv', encoding='utf-8-sig')
df_final_card = pd.read_excel('./all_card.xlsx')
# df_final_customer 는 군집화된 고객 전체 데이터
# df_final_card 는 군집화된 카드 전체 데이터
# %%
customer_cluster_to_card_cluster = { 0 : 0,
                                     1 : 1,
                                     2 : 2,
                                     3 : 3,
                                     4 : 4 }
# customer_cluster_to_card_cluster 는 { 고객 군집명 : 카드 군집명 } 형태의 딕셔너리

def filter_by_previous(df_customer, df_card, ID): # 고객 전체 데이터, 매칭된 카드 데이터, 고객 ID를 인자로 입력
    base_amount = df_customer[df_customer['ID'] == ID]['이용금액_신판_B0M'].values[0] * 1000
    # 고객 ID에 해당하는 고객의 총이용금액 저장
    filtered_df = df_card[df_card['직전 1개월 합계 실적 금액(만원)'] * 10000 <= base_amount]
    # 고객 ID에 해당하는 고객의 총이용금액 보다 직전 실적이 더 작은 카드 데이터 프레임만 남기고
    return filtered_df
    # 그 데이터 프레임을 그대로 반환
# %%
df_final_customer = df_final_customer[df_final_customer['ID'] == 'TRAIN_000001']
# %%
df_final_customer = df_final_customer[df_final_customer['ID'] == 'TRAIN_000024']
# %%
customer_dict = {} # { 고객 ID : 고객 ID에 매칭되는 카드 군집 데이터프레임 } 형태의 딕셔너리
for IDs in df_final_customer['ID']: # 고객 데이터 상의 고객을 모두 순환하면서 매 반복마다 고객 ID를 받아와서
    customer_cluster = df_final_customer[df_final_customer['ID'] == IDs]['군집'].values[0]
    # 고객 ID에 해당하는 고객의 군집명을 받아오고
    customer_dict[IDs] = filter_by_previous(df_final_customer, df_final_card[df_final_card['cluster'] == customer_cluster_to_card_cluster[customer_cluster]].copy(), IDs)
    # customer_cluster_to_card_cluster[customer_cluster] 는 딕셔너리이므로 고객 군집명을 넣으면 고객 군집에 매칭되는 카드 군집이 반환
    # df_final_card[df_final_card['군집'] == customer_cluster_to_card_cluster[customer_cluster]
    # 는 카드 전체 데이터 중에 고객 군집에 매칭되는 카드 데이터를 반환
    # 고객 데이터 및 카드 데이터 중에 고객 군집에 매칭되는 카드 및 고객 ID를 함수에 넣어 반환된 데이터 프레임을
    # customer_dict에 { "TRAIN_0000' : 고객 정보에 따라 필터링되고 남은 카드 데이터프레임 } 형태로 저장
    # 이후 customer_dict[조회하려는 고객 ID] 로 호출하면 고객에 대응되는 필터링된 카드 데이터를 바로 쓸 수 있음

df_card_TRAIN_000024 = customer_dict['TRAIN_000024']
# %%
print(df_final_customer[df_final_customer['ID'] == 'TRAIN_000024']['군집'].values[0])
print(df_card_TRAIN_000024['cluster'].value_counts())

print(df_final_customer[df_final_customer['ID'] == 'TRAIN_000024']['이용금액_신판_B0M'].values[0] * 1000)
print(df_card_TRAIN_000024['직전 1개월 합계 실적 금액(만원)'].describe())
print(df_final_card['직전 1개월 합계 실적 금액(만원)'].describe())


# %% 연회비 기반 필터링
# def filter_by_annualfee(df_customer, df_card, ID):
#     # 기준연회비 가져오기 (ID에 해당하는 고객)
#     anunualfee_customer = df_customer[df_customer['ID'] == ID]['총연회비_B0M'].values[0] * 1000
#     if anunualfee_customer < 30000:
#         filtered_df = df_card[df_card['해외_연회비(만원)'] * 10000 < 50000].copy()
#         return filtered_df
#     elif anunualfee_customer < 50000:
#         filtered_df = df_card[df_card['해외_연회비(만원)'] * 10000 < 100000].copy()
#         return filtered_df
#     elif anunualfee_customer < 100000:
#         filtered_df = df_card[df_card['해외_연회비(만원)'] * 10000 < 300000].copy()
#         return filtered_df
#     elif anunualfee_customer < 300000:
#         filtered_df = df_card[df_card['해외_연회비(만원)'] * 10000 < 300000].copy()
#         return filtered_df
#     else :
#         filtered_df = df_card.copy()
#         return filtered_df
#
# for IDs in df_final_customer['ID']:
#         customer_dict[IDs] = filter_by_annualfee(df_final_customer, customer_dict[IDs].copy(), IDs)

# %%
