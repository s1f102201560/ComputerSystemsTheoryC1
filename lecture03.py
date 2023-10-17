# 平均到着パケット数
lmd = 1200
# print("lmd:", lmd)

# 平均サービス時間　= (8*平均パケット長(bytes)) / リンク容量(bps)
# = 1bit分が出ていく時間
ave_of_serve_time = (8 * 500) / (5 * 10**6)
# print("平均サービス時間:", ave_of_serve_time)

# 利用率 = 平均到着パケット数 / 平均サービス時間(先に出したのは1/uだから掛けている)
# = 時間当たりどれくらい忙しくしているか、満たされているか、仕事があるか
# = パケット数 / 可処分時間　= 時間当たりにさばけるパケットの数
utilization = lmd * ave_of_serve_time
# print("利用率:", utilization)

# 平均遅延時間 = 平均サービス時間 / (1-利用率)
# (1 - 利用率) = 時間当たりにさばいていないパケットがどれくらいあるか
# = (1bit分が出ていく時間) / (さばいてないパケットの割合≒さばいていないパケットがどれくらいあるか)
# 平均遅延時間 = さばいていないパケットをさばく平均時間はどれくらいか
T = ave_of_serve_time/(1-utilization) 
# print("平均遅延時間[s](理論値):", T)
print(f"平均遅延時間[s](理論値): ({T:.5f})")


####################################################################

# 標本データの取得

import numpy as np




# リンク容量(bps)
C = 5 * (10**6)

# パケットの処理時間のリスト生成
def S_i(L_lst):
    length = len(L_lst)
    S_lst = []
    for i in range(length):
        S_lst.append((L_lst[i] * 8) / C)
    return S_lst


# パケットの退去時刻のリスト生成(これは一つの時、被っていない時、今回はm_dの引数で使うだけ)
def D_i(A_lst, S_lst):
    length = len(A_lst) # len(S_lst)でも可
    D_lst = []
    for i in range(length):
        D_lst.append(A_lst[i] + S_lst[i])
    return D_lst

# D_iとS_iは違う、グラフを見ればわかる
# その差を表現しているのが上の関数と下の関数の違い

# パケットiの退去時刻のリストで最大のもの(これをメインで使う)
def max_D_i(D_lst, A_lst, S_lst):
    length = len(D_lst) # len(A_lst)でもlen(S_lst)でも可
    m_D_lst = []
    m_D = -1
    for i in range(length):
        m_D = max(D_lst[i-1], A_lst[i]) + S_lst[i]
        m_D_lst.append(m_D)
    return m_D_lst


# パケットiの遅延時間のリスト生成
def T_i(m_D_lst, A_lst):
    length = len(m_D_lst) # len(A_lst)でも可
    T_lst = []
    for i in range(length):
        T_lst.append(m_D_lst[i] - A_lst[i])
    return T_lst

# print(T_lst[0:10])

# 乱数のシードを設定
np.random.seed(321)

# 取得したい標本の初期化
sample = []
for _ in range(7):

    # 平均到500(bytes)(8*500(bit))の指数分布に従うパケット長を10000個生成
    # パケットのパケット長リスト
    L_lst = np.random.exponential(8*500, 10000)

    # 平均到着パケット数が1200(pps)になるようにパケットを10000個生成
    # パケットの到着時間リスト

    # 平均到着率 (pps)
    lambda_rate = 1200
    # 平均到着間隔 (s)
    mean_interval = 1.0 / lambda_rate
    # 到着間隔を指数分布からサンプリング
    intervals = np.random.exponential(mean_interval, 10000)
    # 累積和を取ることで、到着時刻を得る
    A_lst = np.cumsum(intervals)

    S_lst = S_i(L_lst)
    D_lst = D_i(A_lst, S_lst)
    m_D_lst = max_D_i(D_lst, A_lst, S_lst)
    T_lst = T_i(m_D_lst, A_lst)
    
    delay_ave = np.mean(T_lst)
    sample.append(delay_ave)

print(sample)


# グラフの描画
import matplotlib.pyplot as plt

# 新しい図を作成
plt.figure()

# 折れ線グラフをプロット
plt.plot(range(1, 8), sample, marker='o')  # 7つの点があるため、1から8までの範囲を使用

# グラフのタイトルと軸ラベルを追加
plt.title('Average Packet Delay over Multiple Simulations')
plt.xlabel('Simulation Run')
plt.ylabel('Average Delay (s)')

# グリッドを追加
plt.grid(True)

# グラフを表示
plt.tight_layout()
plt.savefig('graph.png')  # PNG形式でグラフを保存

####################################################################


# T_lstから信頼区間を求める

import numpy as np
from scipy import stats

# 標本データはsample

# 標本平均と標準偏差を計算
mean = np.mean(sample)
std_err = np.std(sample, ddof=1) / np.sqrt(len(sample))  # 標準誤差

# tスコアを計算（95%信頼区間のため、2つの側面を考慮すると0.025と0.975を使用）
alpha = 0.025
df = len(sample) - 1  # 自由度
t_score = stats.t.ppf(1 - alpha/2, df)

# 95%信頼区間を計算
margin_error = t_score * std_err
confidence_interval = (mean - margin_error, mean + margin_error)

print(f"95%信頼区間: ({confidence_interval[0]:.5f}, {confidence_interval[1]:.5f})")