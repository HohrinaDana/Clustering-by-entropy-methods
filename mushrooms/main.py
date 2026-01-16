import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


eps = 1e-9
# Определение НИЗ
def niz_f(df):
    n = df.shape[0]
    col = df.shape[1]

    # Определение НИП

    # энтропия для каждого признака
    def entropy11(sign):
        # вероятность появления признака i в произвольном элементе
        p = sign.value_counts() / n
        h = - sum(p * np.log2(p +eps))
        return h

    # энтропия для каждой пары признаков
    def entropy12(sign1, sign2):
        n = len(sign1)
        df2 = pd.DataFrame({'sign1': sign1, 'sign2': sign2})
        p = df2.value_counts() / n
        h = - sum(p * np.log2(p + eps))
        return h

    # print('Признаки: ', df.columns, '\n')

    e11 = {}
    for i in range(col): e11[f'H({i})'] = entropy11(df.iloc[:, i])
    # print('Энтропия для каждого признака:', e11, '\n')

    e12 = {}
    for i in range(col):
        for j in range(col):
            e12[f'H({i}, {j})'] = entropy12(df.iloc[:, i], df.iloc[:, j])
    # print('Энтропия для каждой пары признаков:', e12, '\n')

    # мера информативности признаков
    m = np.zeros(col)
    for i in range(col):
        koef_h = 0
        koef_H = 0
        for j in range(col):
            if j == i:
                continue
            # часть j-ого признака обьясняющаяся i-м
            koef_h += e11[f'H({i})'] + e11[f'H({j})'] - e12[f'H({i}, {j})']
            # совместная энтропия
        m[i] = koef_h  # абсолютное кол-во информации в битах
    # print('Мера информативности признаков:', m, '\n')

    # НИП
    nip_index = m.argmax()
    print('НИП:', df.columns[nip_index], '\n')
    nip = df.iloc[:, nip_index]
    value_nip = sorted(nip.unique())

    # Определение НИЗ
    # информация о значениях признака i
    p21 = nip.value_counts().sort_index() / n
    h21 = - (p21 * np.log2(p21 + eps) + (1 - p21) * np.log2(1 - p21 - eps))
    e21 = {}
    for i in range(len(nip.unique())):
        e21[f'H({i})'] = float(h21.iloc[i])
    # print(f'Энтропия для всех значений НИПа:', e21, '\n')

    # информация о комбинациях значениях признака i со значением q признака j
    e22 = {}
    # цикл по значениям НИП (r)
    for i in range(len(nip.unique())):
        mask = (nip == value_nip[i])
        df2 = df.loc[mask]
        df3 = df.loc[~mask]

        # цикл по остальным признакам
        for j in range(col):
            if j == nip_index:
                continue
            sign = df2.iloc[:, j]
            value = sign.unique()  # только комбинирующиеся значения
            valueAll = df.iloc[:, j].unique()
            valueNot = list(set(valueAll) - set(value))  # некомбин. знач. признака

            # вероятность появления комбинирующегося значения в признаке j вместе со знач i
            p1 = np.array([(df2.iloc[:, j] == v).sum() / n for v in value])
            p1[p1 == 0] = 1e-9

            # вероятность появления комбинирующегося значения в признаке j без знач i
            p2 = np.array([(df3.iloc[:, j] == v).sum() / n for v in value])
            p2[p2 == 0] = 1e-9

            # вероятность появления НЕкомбинирующегося значения в признаке j
            p3 = (df.iloc[:, j].isin(valueNot)).sum() / n
            if p3 == 0:
                p3 = 1e-9

            h = - (float(np.sum(p1 * np.log2(p1 + eps))) + float(np.sum(p2 * np.log2(p2 + eps))) + p3 * np.log2(p3 + eps))

            e22[f'H({i}, {j})'] = float(h)  # i = значение НИЗ и j = признак

    # print(f'Энтропия для всех признаков в комбинации со всеми значениями НИПа:', e22, '\n')

    # мера информативности значения
    m2 = np.zeros(len(value_nip))
    for i in range(len(value_nip)):
        koef_h = 0
        koef_H = 0
        for j in range(col):
            if j == nip_index:
                continue
            # часть энтропии значений признака j, которая известна,
            # если известно значение признака i
            koef_h += e21[f'H({i})'] + e11[f'H({j})'] - e22[f'H({i}, {j})']
        m2[i] = koef_h

    # НИЗ
    niz_index = m2.argmax()
    # print('НИЗ:', value_nip[niz_index], '\n')

    return nip_index, niz_index


df = pd.read_excel('ВидыГрибов.xlsx')

# замена пропусков на 'NaN'
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna('NaN')

clusters = []
value_clusters = []
homogeneity_scores = []

# Однородность кластеров
def get_homogeneity(df_cluster):
    # u = sum(n_max_j) / (n * m)

    n = df_cluster.shape[0]  # кол-во элементов
    m = df_cluster.shape[1]  # кол-во признаков

    if n == 0: return 0.0

    sum_n_max = 0
    # Проходим по всем столбцам
    for col in df_cluster.columns:
        # n_max(j) - сколько раз встречается самое частое значение
        n_max_j = df_cluster[col].value_counts().max()
        sum_n_max += n_max_j

    u = sum_n_max / (n * m)
    return u

# Кластеризация
for i in range(20):
    nip, niz = niz_f(df)
    mask = (df.iloc[:, nip] == sorted(df.iloc[:, nip].unique())[niz])
    current_cluster = df[mask]

    clusters.append(current_cluster)
    u = get_homogeneity(current_cluster)
    homogeneity_scores.append(u)

    value_clusters.append([df.columns[nip], sorted(df.iloc[:, nip].unique())[niz]])
    df = df[~mask]

    if len(df) < 10:
        clusters.append(df)
        value_clusters.append(['Остаток'])
        homogeneity_scores.append(get_homogeneity(df))
        print('Всего кластеров:', i + 2)
        break


# Визуализация

# Подсчет ядовитых грибов в каждом кластере
cluster_sum = []
for i, cluster in enumerate(clusters):
    counts = cluster['class'].value_counts()
    cluster_sum.append(counts)

    counts.name = ': '.join(map(str, value_clusters[i]))

cluster_sum = pd.DataFrame(cluster_sum)

# Контроль однородности
all_homogeneity = pd.DataFrame({
    'Однородность': homogeneity_scores,
    'Размер': cluster_sum.sum(axis=1)
}, index=cluster_sum.index)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

cluster_sum.plot(
    kind='bar',
    stacked=True,
    ax=axes[0],
    color=['green', 'red']
)
axes[0].set_ylabel('Кол-во грибов')
axes[0].set_title('Распределение грибов по ядовитости')
axes[0].legend(['Съедобные', 'Ядовитые'], title='Класс гриба')
axes[0].tick_params(axis='x', rotation=45)

bars = axes[1].bar(
    all_homogeneity.index,
    all_homogeneity['Однородность'],
    alpha=0.8
)

axes[1].set_title('Оценка однородности кластеров', fontsize=14)
axes[1].set_ylabel('Коэффициент однородности', fontsize=12)
axes[1].set_xlabel('НИП и НИЗ', fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.2f}', ha='center')

axes[1].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='Полная однородность')
axes[1].legend()

plt.tight_layout()
plt.show()
