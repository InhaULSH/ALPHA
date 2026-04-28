from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import re
# plt 한글 출력 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# %% 데이터 전처리, 정규화
df_all_customer = pd.read_csv('./all_customer_original.csv',encoding = 'utf-8-sig')
# 데이터 로드
from sklearn.preprocessing import OrdinalEncoder
life_stage_order = ['독신', '가족구축기', '자녀출산기', '자녀성장(1)', '자녀성장(2)', '자녀독립기', '노년생활']
life_stage_encoder = OrdinalEncoder(categories=[life_stage_order])
print(df_all_customer['Life_Stage'].value_counts())
df_all_customer['Life_Stage'] = life_stage_encoder.fit_transform(df_all_customer[['Life_Stage']]).astype(int)
print(df_all_customer['Life_Stage'].value_counts())
# 고객 데이터 Life_Stage 컬럼 수치형 변환
df_all_customer['연령'] = df_all_customer['연령'].str.extract(r'(\d+)').astype(int)
# 고객 데이터 연령 컬럼 수치형 변환
df_all_customer['이용금액_페이_B0M'] = df_all_customer['이용금액_페이_온라인_B0M'] + df_all_customer['이용금액_페이_오프라인_B0M']
df_all_customer = df_all_customer.drop(['이용금액_페이_온라인_B0M', '이용금액_페이_오프라인_B0M'], axis = 1)
# 컬럼 정리
df_all_customer_scaled = df_all_customer.copy()
numeric_columns = ['연령', '이용금액_해외', '쇼핑_도소매_이용금액', '쇼핑_백화점_이용금액', '쇼핑_마트_이용금액', '쇼핑_슈퍼마켓_이용금액', '쇼핑_편의점_이용금액',
                   '쇼핑_아울렛_이용금액', '쇼핑_온라인_이용금액', '쇼핑_기타_이용금액', '교통_주유이용금액', '교통_정비이용금액', '교통_통행료이용금액', '교통_버스지하철이용금액', '교통_택시이용금액',
           '교통_철도버스이용금액', '여유_운동이용금액', '여유_Pet이용금액', '여유_공연이용금액', '여유_공원이용금액', '여유_숙박이용금액', '여유_여행이용금액', '여유_항공이용금액',
           '여유_기타이용금액', '납부_통신비이용금액', '납부_관리비이용금액', '납부_렌탈료이용금액', '납부_가스전기료이용금액', '납부_보험료이용금액', '납부_유선방송이용금액',
           '납부_건강연금이용금액', '납부_기타이용금액', '이용금액_온라인_B0M', '이용금액_오프라인_B0M', '이용금액_페이_B0M','할인금액_B0M', '혜택수혜금액',
           '이용금액_신판_B0M', '총연회비_B0M']
df_all_customer_scaled[numeric_columns] = df_all_customer_scaled[numeric_columns].apply(lambda x: np.log1p(x))
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
df_all_customer_scaled[numeric_columns] = data_scaler.fit_transform(df_all_customer_scaled[numeric_columns])
# 수치형 데이터 로그변환 후 표준정규화
print(df_all_customer.columns)
print(df_all_customer_scaled.columns)


# %% 데이터 분포 확인
def plot_histograms(df, cols_per_row=3, figsize=(15, 5), bins=30):
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        print("수치형 컬럼이 없습니다.")
        return

    # 행과 열 개수 계산
    n_cols = cols_per_row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    # 전체 figure 크기 조정
    total_figsize = (figsize[0], figsize[1] * n_rows)

    # subplot 생성
    fig, axes = plt.subplots(n_rows, n_cols, figsize=total_figsize)

    # axes를 1차원 배열로 변환 (단일 행인 경우 처리)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    # 각 수치형 컬럼에 대해 히스토그램 생성
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=bins, color='skyblue',
                     edgecolor='black', alpha=0.7)
        axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].set_ylabel('빈도', fontsize=10)
        axes[i].grid(True, alpha=0.3)

    # 사용하지 않는 subplot 제거
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
plot_histograms(df_all_customer)


# %% 차원축소 - 업종별 이용금액(정규화)
df_clust_rawscaled = df_all_customer_scaled.drop(['기준년월', 'ID', '이용여부_3M_해외겸용_본인', '대중교통_점수', '자가용_점수',
                                             '해외_점수', '여행_점수', '문화생활_점수', '쇼핑_점수', '생필품_점수',
                                             '납부(고정지출)_점수', '디지털결제_점수', '가족_점수', '할인금액_B0M'], axis = 1).copy()
# %% 차원축소 - 점수
df_clust_score = df_all_customer[['남녀구분코드', '대중교통_점수', '자가용_점수', '해외_점수', '여행_점수', '문화생활_점수', '쇼핑_점수',
                                 '생필품_점수', '납부(고정지출)_점수', '디지털결제_점수', '가족_점수']].copy()
df_clust_score_percent = df_clust_score.copy()
df_clust_score_percent['총점'] = df_clust_score_percent[['대중교통_점수', '자가용_점수', '해외_점수', '여행_점수', '문화생활_점수',
                                                       '쇼핑_점수', '생필품_점수', '납부(고정지출)_점수', '디지털결제_점수', '가족_점수'
                                                       ]].sum(axis = 1)
columns = ['대중교통_점수', '자가용_점수', '해외_점수', '여행_점수', '문화생활_점수', '쇼핑_점수', '생필품_점수', '납부(고정지출)_점수',
           '디지털결제_점수', '가족_점수']
for col in columns:
    df_clust_score_percent[col] = df_clust_score_percent[col] / df_clust_score_percent['총점']
print(df_clust_score_percent.columns)


# %% 대표 샘플 추출
from sklearn.model_selection import train_test_split
def dataframe_sampling(df, sample_size, stratify_col):
    train_idx, sample_idx = train_test_split(np.arange(len(df)),
        test_size = sample_size,
        stratify = df[stratify_col],
        random_state = 42
    )
    df_sample = df.iloc[sample_idx]
    return df_sample
# 대표 샘플 추출 함수
# %% - 업종별 이용금액(정규화)
df_clust_rawscaled['stratifyKey'] = (df_clust_rawscaled['남녀구분코드'].astype(str) + '_' + df_clust_rawscaled['Life_Stage'].astype(
    str))
df_clust_rawscaled_sampled = dataframe_sampling(df_clust_rawscaled, 18755, 'stratifyKey')
df_clust_rawscaled = df_clust_rawscaled.drop('stratifyKey', axis = 1)
df_clust_rawscaled_sampled = df_clust_rawscaled_sampled.drop('stratifyKey', axis = 1)
df_clust_rawscaled = df_clust_rawscaled.drop(['가족_금액합', '가족_금액점수', '가족_라이프점수'], axis = 1)
df_clust_rawscaled_sampled = df_clust_rawscaled_sampled.drop(['가족_금액합', '가족_금액점수', '가족_라이프점수'], axis = 1)
# %% - 점수
df_clust_score_percent['stratifyKey'] = (df_clust_score_percent['남녀구분코드'].astype(str) + '_')
df_clust_score_percent_sampled = dataframe_sampling(df_clust_score_percent, 18755, 'stratifyKey')
df_clust_score_percent = df_clust_score_percent.drop(['stratifyKey', '총점', '남녀구분코드'], axis = 1)
df_clust_score_percent_sampled = df_clust_score_percent_sampled.drop(['stratifyKey', '총점', '남녀구분코드'], axis = 1)

print(df_clust_score_percent_sampled['대중교통_점수'].describe())
print(df_clust_score_percent['대중교통_점수'].describe())
print(df_clust_score_percent_sampled['쇼핑_점수'].describe())
print(df_clust_score_percent['쇼핑_점수'].describe())


# %% 군집화 자원 소모량 평가 함수
from memory_profiler import memory_usage
import time
def profile_function(func, *args, **kwargs):
    print(f"Function: {func.__name__}")
    start = time.time()
    mem_usage, result = memory_usage((func, args, kwargs), retval = True)
    end = time.time()
    print(f"Time: {end - start:.4f}s | Memory used: {max(mem_usage) - min(mem_usage):.4f} MiB")
    return result
# 함수의 시간 및 메모리 사용량 출력


# %% 군집화 (KMeans)
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
def visualize_optimal_k(df, k_range=range(2, 11)):
    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        print("오류: 수치형 컬럼이 없습니다. K-means는 수치형 데이터만 사용 가능합니다.")
        return None, None

    print(f"수치형 컬럼 ({len(numeric_cols)}개): {numeric_cols}")

    # 수치형 데이터만 추출
    X = df[numeric_cols].values

    # 결측치 처리
    if pd.DataFrame(X).isnull().sum().sum() > 0:
        print("경고: 결측치가 발견되어 0으로 채웁니다.")
        X = pd.DataFrame(X).fillna(0).values

    print(f"데이터 형태: {X.shape}")

    # 결과 저장용 리스트
    inertias = []  # Within-cluster sum of squares (WCSS)
    silhouette_scores = []

    print("\nK-means 클러스터링 성능 평가 중...")

    for k in k_range:
        print(f"K={k} 처리 중...")

        if df.shape[0] < 100000 :
            # K-means 클러스터링
            kmeans = KMeans(
                n_clusters = k,
                init = 'k-means++',
                random_state = 42,
                n_init = 30,
                max_iter = 300,
                algorithm = 'lloyd'
            )
        else :
            kmeans = MiniBatchKMeans(
                n_clusters = k,
                init = 'k-means++',
                batch_size = 1000,
                random_state = 42,
                n_init = 30,
                max_iter = 300
            )

        labels = kmeans.fit_predict(X)

        # 1) 엘보우 기법: Inertia (WCSS - Within-Cluster Sum of Squares)
        inertias.append(kmeans.inertia_)

        # 2) 실루엣 점수
        try:
            silhouette_avg = silhouette_score(X, labels, metric='euclidean')
            silhouette_scores.append(silhouette_avg)
            print(f"K={k}: Inertia = {kmeans.inertia_:.2f}, Silhouette Score = {silhouette_avg:.3f}")
        except Exception as e:
            print(f"실루엣 점수 계산 실패 (K={k}): {e}")
            silhouette_scores.append(np.nan)

    # 시각화
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 왼쪽 축: Inertia (엘보우 기법)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (WCSS - lower is better)', color='tab:blue')
    line1 = ax1.plot(k_range, inertias, 'o-', color='tab:blue', linewidth=2, markersize=6, label='Inertia (WCSS)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # 오른쪽 축: 실루엣 점수
    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette Score (higher is better)', color='tab:green')
    line2 = ax2.plot(k_range, silhouette_scores, 's--', color='tab:green', linewidth=2, markersize=6,
                     label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 범례
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Elbow Method (Inertia) vs Silhouette Score for Optimal K\n(K-means Clustering)',
              fontsize=14)
    plt.tight_layout()
    plt.show()

    return inertias, silhouette_scores
# 모든 컬럼에 대하여 엘보우 기법으로, 수치형 컬럼에 대하여 실루엣 계수로 시각화
def cluster_by_kmeans(df, optimal_k = 4):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values

    kmeans = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        random_state=42,
        n_init=30,
        max_iter=300,
        algorithm='lloyd'
    )

    cluster_labels = kmeans.fit_predict(X)
    clustered_df = df.copy()
    clustered_df['군집'] = cluster_labels

    return clustered_df
# 최적 K 값으로 군집화
# %% 대표 표본 대상 - 정규화
try:
    profile_function(visualize_optimal_k, df_clust_rawscaled_sampled)
    print("실행 완료!")
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()
# %%
df_kmeans_customer_rawscaled = cluster_by_kmeans(df_clust_rawscaled_sampled, 6)
# %% 대표 표본 대상 - 점수
try:
    profile_function(visualize_optimal_k, df_clust_score_percent_sampled)
    print("실행 완료!")
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()
# %%
df_kmeans_customer_score = cluster_by_kmeans(df_clust_score_percent_sampled, 5)
# %%
df_kmeans_customer_score2 = cluster_by_kmeans(df_clust_score_percent_sampled, 6)

# %% 군집화 (AgglomerativeClustering)
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
def visualize_optimal_agglomerative(df, linkage_method='ward', metric='euclidean'):
    # 1. 데이터 전처리
    range_n_clusters = range(2, 11)
    X = df.select_dtypes(include=[np.number]).values
    selected_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"자동 선택된 컬럼: {selected_columns}")

    # 결측치 처리
    if pd.DataFrame(X).isnull().sum().sum() > 0:
        print("경고: 결측치가 발견되어 0으로 채웁니다.")
        X = pd.DataFrame(X).fillna(0).values

    print(f"데이터 형태: {X.shape}")

    # 준비
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    print(f"\n클러스터링 성능 평가 중 (linkage: '{linkage_method}', metric: '{metric}')...")

    if linkage_method == 'ward':
        metric = 'euclidean'

    # 2. 각 군집 수에 대해 평가지표 계산
    for n_clusters in range_n_clusters:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=metric
        )
        cluster_labels = clustering.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        calinski_harabasz_avg = calinski_harabasz_score(X, cluster_labels)
        calinski_harabasz_scores.append(calinski_harabasz_avg)

        davies_bouldin_avg = davies_bouldin_score(X, cluster_labels)
        davies_bouldin_scores.append(davies_bouldin_avg)

        print(f"군집수 {n_clusters}: 실루엣 점수 = {silhouette_avg:.3f}, CH 지수 = {calinski_harabasz_avg:.3f}, DB 지수 = {davies_bouldin_avg:.3f}")

    # 3. 세 평가지표 그래프 시각화
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 실루엣 점수
    ax1.plot(range_n_clusters, silhouette_scores, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_xlabel('군집수')
    ax1.set_ylabel('실루엣 점수')
    ax1.set_title('실루엣 점수에 따른 최적 군집수')
    ax1.grid(True, alpha=0.3)

    # 칼린스키-하라바즈 지수
    ax2.plot(range_n_clusters, calinski_harabasz_scores, 's-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('군집수')
    ax2.set_ylabel('칼린스키-하라바즈 지수')
    ax2.set_title('칼린스키-하라바즈 지수에 따른 최적 군집수')
    ax2.grid(True, alpha=0.3)

    # 데이비스-볼딘 지수
    ax3.plot(range_n_clusters, davies_bouldin_scores, '^-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('군집수')
    ax3.set_ylabel('데이비스-볼딘 지수')
    ax3.set_title('데이비스-볼딘 지수에 따른 최적 군집수')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 4. 최적 군집수 결정
    best_silhouette_n = list(range_n_clusters)[np.argmax(silhouette_scores)]
    best_ch_n = list(range_n_clusters)[np.argmax(calinski_harabasz_scores)]
    best_db_n = list(range_n_clusters)[np.argmin(davies_bouldin_scores)]

    # 5. 덴드로그램 시각화를 위한 헬퍼 함수
    def plot_dendrogram_from_model(X, n_clusters, title):
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage=linkage_method,
            metric=metric
        )
        model = model.fit(X)

        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)

        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([
            model.children_,
            model.distances_,
            counts
        ]).astype(float)

        plt.figure(figsize=(12, 8))
        dendrogram(
            linkage_matrix,
            truncate_mode='level',
            p=5,
            leaf_rotation=90,
            leaf_font_size=10
        )
        plt.title(f'{title}', fontsize=14)
        plt.xlabel('샘플 인덱스 또는 클러스터 크기')
        plt.ylabel('거리')
        plt.tight_layout()
        plt.show()

    # 6. 최적 군집수에 따른 덴드로그램 시각화
    if best_silhouette_n == best_ch_n == best_db_n:
        print(f"\n세 지표가 모두 {best_silhouette_n}개 군집을 최적으로 선택했습니다.")
        plot_dendrogram_from_model(X, best_silhouette_n, "세 지표 기준 최적 군집수 덴드로그램")
    else:
        print(f"\n최적 군집수가 각기 다릅니다. 각 지표별로 덴드로그램을 표시합니다.")
        plot_dendrogram_from_model(X, best_silhouette_n, "실루엣 점수 기준 최적 군집수 덴드로그램")
        plot_dendrogram_from_model(X, best_ch_n, "칼린스키-하라바즈 지수 기준 최적 군집수 덴드로그램")
        plot_dendrogram_from_model(X, best_db_n, "데이비스-볼딘 지수 기준 최적 군집수 덴드로그램")

    return silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores
def cluster_by_agglomerative(df, optimal_k = 4, linkage_method = 'ward', metric = 'euclidean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[numeric_cols].values

    if linkage_method == 'ward':
        metric = 'euclidean'

    model = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage=linkage_method,
        metric=metric
    )

    cluster_labels = model.fit_predict(X)
    clustered_df = df.copy()
    clustered_df['군집'] = cluster_labels

    return clustered_df
# %% 대표 표본 대상 - 정규화
try:
    profile_function(visualize_optimal_agglomerative, df_clust_rawscaled_sampled)
    profile_function(visualize_optimal_agglomerative, df_clust_rawscaled_sampled, 'average')
    profile_function(visualize_optimal_agglomerative, df_clust_rawscaled_sampled, 'complete')
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()
# %%
df_agglo_customer_rawscaled = cluster_by_agglomerative(df_clust_rawscaled_sampled, 4, 'complete')
df_agglo_customer_rawscaled_age = cluster_by_agglomerative(df_clust_rawscaled_sampled_age, 6, 'complete')
# %% 대표 표본 대상 - 점수
try:
    profile_function(visualize_optimal_agglomerative, df_clust_score_sampled)
    profile_function(visualize_optimal_agglomerative, df_clust_score_sampled, 'average')
    profile_function(visualize_optimal_agglomerative, df_clust_score_sampled, 'complete')
except Exception as e:
    print(f"에러 발생: {e}")
    import traceback
    traceback.print_exc()


# %% 분류 (비지도 의사 결정 트리 - 최적 깊이 결정)
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')
def visualize_optimal_tree_depth(df, target_col='군집', max_depth_range=range(1, 11), corr_threshold=0.9, random_state=42):
    print("종합적 의사결정 트리 평가")
    print("=" * 50)

    # ───────── 데이터 준비 ─────────
    feature_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]

    if X_raw.empty or len(y.unique()) < 2 or X_raw.shape[0] < 10:
        print("데이터가 분석에 적합하지 않습니다.")
        return None

    # 1) 높은 상관관계 피처 제거
    corr = X_raw.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    X = X_raw.drop(columns=drop_cols)
    print(f"제거된 피처 수: {len(drop_cols)} / {X_raw.shape[1]} (|corr|>{corr_threshold})")

    # 2) 결측치 0으로 대체
    X = X.fillna(0)

    print(f"데이터 형태: X={X.shape}, 클래스 수={len(y.unique())}")

    # ───────── 모델 평가 ─────────
    min_samples = y.value_counts().min()
    cv_folds = max(2, min(5, min_samples))
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state)

    try:
        train_scores, test_scores = validation_curve(
            DecisionTreeClassifier(random_state=random_state),
            X, y,
            param_name='max_depth',
            param_range=max_depth_range,
            cv=cv,
            scoring='f1_macro'
        )

        test_means  = test_scores.mean(axis=1)
        test_stds   = test_scores.std(axis=1)
        train_means = train_scores.mean(axis=1)

        # 3) 최고 F1-macro 깊이(들) 찾기
        max_f1       = np.max(test_means)
        best_indices = np.where(test_means == max_f1)[0]
        best_depths  = [list(max_depth_range)[i] for i in best_indices]

        print(f"\n최고 검증 F1-macro: {max_f1:.4f}")
        print(f"해당 깊이: {best_depths}")

        print("\n상세 정보")
        print("-" * 40)
        for idx in best_indices:
            depth = list(max_depth_range)[idx]
            gap   = train_means[idx] - test_means[idx]
            print(f"깊이 {depth:<2} | F1_val={test_means[idx]:.4f} "
                  f"(±{test_stds[idx]:.4f}) | F1_train={train_means[idx]:.4f} "
                  f"| 과적합={gap:.4f}")

        # 4) 최종 깊이: 동일 F1이면 더 얕은 트리 선택
        best_idx   = best_indices[0]   # np.where 결과는 오름차순
        best_depth = best_depths[0]
        gap_sel    = train_means[best_idx] - test_means[best_idx]

        print("\n최종 선택 깊이:", best_depth)
        print(f"검증 F1-macro: {max_f1:.4f} (±{test_stds[best_idx]:.4f})")
        print(f"훈련 F1-macro: {train_means[best_idx]:.4f}")
        print(f"과적합 정도  : {gap_sel:.4f}")

        return best_depth

    except Exception as e:
        print("validation_curve 실행 실패:", e)
        return None
# %% - 정규화
visualize_optimal_tree_depth(df_kmeans_customer_rawscaled)
# %% - 점수
visualize_optimal_tree_depth(df_kmeans_customer_score)
visualize_optimal_tree_depth(df_kmeans_customer_score2)

# %% 분류 (비지도 의사 결정 트리 - 하이퍼 파라미터 튜닝 및 분류)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
def tune_and_classify_dt(df, model_name, df_target, max_depth):
    # 입력 데이터에서 특성과 타겟 분리
    X = df_target.drop(columns=['군집']).copy()
    _X = df.drop(columns=['군집']).copy()
    y = df_target['군집'].copy()

    # 의사결정트리 모델 생성
    dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth)

    # 하이퍼파라미터 후보 설정
    param_grid = {
        'min_samples_split': [100, 200, 300, 500],
        'min_samples_leaf': [50, 100, 150, 200],
        'max_leaf_nodes': [6, 8, 10, 15, 20],
        'max_features': [None, 'sqrt']
    }

    # GridSearchCV를 통한 하이퍼파라미터 튜닝
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='f1_macro')
    grid_search.fit(X, y)

    # 최적 하이퍼파라미터 출력
    best_params = grid_search.best_params_

    # 최적 하이퍼파라미터로 재학습 및 예측
    best_dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth, **best_params)
    best_dt.fit(X, y)
    y_pred = best_dt.predict(_X)

    # 원본 데이터프레임에 예측 결과 컬럼 추가
    df['군집'] = y_pred

    plt.figure(figsize=(20, 10))
    plot_tree(best_dt,
              feature_names=df.drop(columns=['군집']).copy().columns,
              class_names=None,
              filled=True,
              rounded=True,
              fontsize=10,
              max_depth=max_depth)
    plt.title(f"{model_name} - 의사결정 트리 다이어그램", fontsize=16, pad=20)
    plt.show()

    return df
# 전체 표본 군집에 적용 후 트리 다이어그램 시각화
# %% - 정규화
df_clust_rawscaled['군집'] = -1
df_kmeans_customer_rawscaled_all = tune_and_classify_dt(df_clust_rawscaled, "K-Means (대표  샘플 추출) (비점수화 / 정규화)",
                                                        df_kmeans_customer_rawscaled, 4)
# %%
df_clust_score_percent['군집'] = -1
df_kmeans_customer_score_all = tune_and_classify_dt(df_clust_score_percent, "K-Means (대표  샘플 추출) (점수)",
                                                    df_kmeans_customer_score, 10)
# %%
df_kmeans_customer_score2_all = tune_and_classify_dt(df_clust_score_percent, "K-Means (대표  샘플 추출) (점수)",
                                                    df_kmeans_customer_score2, 10)

# %% 근집 분리도 분석
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import numpy as np
def cluster_evaluation(df, cluster_column = '군집'):
    X = df.drop(columns=[cluster_column], axis = 1).copy()
    labels = df[cluster_column]

    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    print("군집 평가 지표:")
    print(f"Silhouette Score: {silhouette:.4f} (높을수록 좋음)")
    print(f"Calinski-Harabasz Score: {ch_score:.4f} (높을수록 좋음)")
    print(f"Davies-Bouldin Score: {db_score:.4f} (낮을수록 좋음)")
cluster_evaluation(df_kmeans_customer_rawscaled_all)
cluster_evaluation(df_kmeans_customer_score_all)

# %% 군집 EDA
import os
def fragment_dataframe(df, n_parts = 10, path ='merged', algo = 'k') :
    chunk_size = len(df) // n_parts
    for i in range(n_parts):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_parts - 1 else len(df)
        df_chunk = df.iloc[start:end]
        filepath = f'./{path}/df_part_{i}_{algo}.csv'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_chunk.to_csv(filepath, encoding='utf-8-sig', index=False)
# 군집을 분할된 CSV로 저장
# 이후 Perplexity를 통해 EDA
fragment_dataframe(df_kmeans_customer_rawscaled, 10, 'raw_scaled', 'k6')
fragment_dataframe(df_kmeans_customer_score, 10, 'score', 'k4')

# %% 전체 표본과 대표 표본의 분포 확인
plot_histograms(df_kmeans_customer_rawscaled)
plot_histograms(df_kmeans_customer_raw)
plot_histograms(df_kmeans_customer_rawscaled_all)
plot_histograms(df_kmeans_customer_raw_all)

# %% df_all_customer 적용 후 저장
df_all_customer['군집'] = df_kmeans_customer_score2_all['군집'].copy()
df_all_customer['대중교통_소비비율'] = df_kmeans_customer_score2_all['대중교통_점수'].copy()
df_all_customer['자가용_소비비율'] = df_kmeans_customer_score2_all['자가용_점수'].copy()
df_all_customer['해외_소비비율'] = df_kmeans_customer_score2_all['해외_점수'].copy()
df_all_customer['여행_소비비율'] = df_kmeans_customer_score2_all['여행_점수'].copy()
df_all_customer['문화생활_소비비율'] = df_kmeans_customer_score2_all['문화생활_점수'].copy()
df_all_customer['쇼핑_소비비율'] = df_kmeans_customer_score2_all['쇼핑_점수'].copy()
df_all_customer['생필품_소비비율'] = df_kmeans_customer_score2_all['생필품_점수'].copy()
df_all_customer['납부(고정지출)_소비비율'] = df_kmeans_customer_score2_all['납부(고정지출)_점수'].copy()
df_all_customer['디지털결제_소비비율'] = df_kmeans_customer_score2_all['디지털결제_점수'].copy()
df_all_customer['가족_소비비율'] = df_kmeans_customer_score2_all['가족_점수'].copy()
# %%
df_all_customer.to_csv('./all_customer.csv', encoding = 'utf-8-sig', index=False)


# %% 모델 선정 및 군집 대표값 선정
from scipy import stats
def create_cluster_representative_df(df, cluster_column='cluster', id_column='ID'):
    # 집계 함수 정의
    agg_dict = {}

    # 수치형 컬럼들
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop([cluster_column], errors='ignore')

    for col in numeric_cols:
        agg_dict[col] = 'mean'

    # 범주형 컬럼들
    categorical_cols = []
    for col in categorical_cols:
        if col in df.columns:
            agg_dict[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None

    # groupby 실행
    summary = df.groupby(cluster_column).agg(agg_dict).reset_index()
    summary[id_column] = summary[cluster_column]

    # 컬럼 순서 정리
    original_columns = [col for col in df.columns if col != cluster_column]
    available_columns = [col for col in original_columns if col in summary.columns]

    return summary[available_columns]
df_customer_representative = None
df_customer_representative = create_cluster_representative_df(df_all_customer, '군집')
df_customer_percent_representative = create_cluster_representative_df(df_kmeans_customer_score_all, '군집')
df_customer_representative = pd.concat([df_customer_representative, df_customer_percent_representative], axis=1)
# %%
df_customer_representative.to_csv('./cluster_representative.csv', encoding = 'utf-8-sig', index=False)