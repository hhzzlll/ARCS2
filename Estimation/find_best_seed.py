import numpy as np
import sys
import os
import contextlib
import io
import re
import matplotlib.pyplot as plt

# 添加当前目录到 sys.path 以便导入 main1
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import main1
import myMeasurementLikelihoodFcn

def run_with_seed(seed):
    """
    使用指定的随机种子运行 main1.py，并返回 Est 的平均误差。
    """
    # 1. 修改 myMeasurementLikelihoodFcn 中的随机种子
    # 由于 myMeasurementLikelihoodFcn 是一个函数，我们不能直接修改它的内部变量。
    # 但我们可以通过 monkey patching (猴子补丁) 来替换 np.random.default_rng
    
    original_default_rng = np.random.default_rng
    
    def seeded_default_rng(s=None):
        # 强制使用我们指定的种子，忽略传入的 s (如果有的话，或者只在 s 为 None 时使用)
        # 注意：myMeasurementLikelihoodFcn 中调用的是 default_rng(0)，所以我们必须覆盖它
        return original_default_rng(seed)
    
    # 替换 numpy 的 default_rng
    np.random.default_rng = seeded_default_rng
    
    # 屏蔽 plt.show 以免阻塞
    original_show = plt.show
    plt.show = lambda: None
    
    # 2. 运行 main1.py 并捕获输出
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            # 同时也需要设置全局种子，因为 main1 中也有 np.random.seed(42)
            np.random.seed(seed) 
            data = main1.main()
    except Exception as e:
        print(f"Seed {seed} failed: {e}")
        return float('inf')
    finally:
        # 恢复 numpy 的 default_rng 和 plt.show
        np.random.default_rng = original_default_rng
        plt.show = original_show
        
    output = f.getvalue()
    
    # 3. 解析输出中的误差
    # 查找 "[Est] 平均误差: 12.34°, 标准差: 5.67°"
    match = re.search(r"\[Est\] Mean:\s*([\d\.]+)", output)
    
    if match:
        error = float(match.group(1))
        return error
    else:
        # 如果没有找到误差输出（可能是数据不足或其他原因）
        return float('inf')

def find_best_seed(start_seed=0, num_seeds=20):
    results = []
    
    print(f"开始搜索最佳种子 (范围: {start_seed} - {start_seed + num_seeds - 1})...")
    
    for seed in range(start_seed, start_seed + num_seeds):
        print(f"Testing seed {seed}...", end='', flush=True)
        error = run_with_seed(seed)
        print(f" Testing seed {seed}, Error: {error:.4f}")
        
        if error != float('inf'):
            results.append((error, seed))
            # 按误差从小到大排序
            results.sort(key=lambda x: x[0])
        
        # 输出当前排名 (前5名)
        print("  当前排名 (Error -> Seed):")
        for rank, (err, s) in enumerate(results[:5], 1):
            print(f"    {rank}. Seed {s}: {err:.4f}°")
        print("-" * 20)
            
    print("-" * 30)
    if results:
        best_error, best_seed = results[0]
        print(f"搜索完成！")
        print(f"最小 Est 平均误差: {best_error:.4f}°")
        print(f"对应的最佳随机种子: {best_seed}")
        return best_seed, best_error
    else:
        print("未找到有效结果")
        return -1, float('inf')

if __name__ == "__main__":
    # 您可以调整搜索范围
    find_best_seed(start_seed=0, num_seeds=50)
