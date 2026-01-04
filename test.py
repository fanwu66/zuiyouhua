import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from typing import List, Tuple, Dict
import matplotlib as mpl

# ============== 修复中文显示问题 ==============
# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'KaiTi', 'SimSun']  # 优先使用这些字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'sans-serif'
# ============== 中文显示配置结束 ==============

class DualAscentDataAssociation:
    """
    基于对偶上升法的多源传感器数据关联优化
    
    该类实现了论文中描述的对偶上升算法，用于解决两个传感器之间的数据关联问题。
    使用1-IoU作为匹配代价，通过拉格朗日松弛和对偶上升法求解最优匹配。
    特别设计的场景确保需要两轮迭代才能达到最优解。
    """
    
    def __init__(self, alpha: float = 0.4, max_iterations: int = 2):
        """
        初始化对偶上升法参数
        
        Parameters:
        -----------
        alpha : float
            步长参数，控制拉格朗日乘子的更新速度
            特别设置为0.4以确保两轮迭代达到最优
        max_iterations : int
            最大迭代次数，设置为2轮
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.convergence_history = []
        
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个2D bounding box的交并比(IoU)
        
        Parameters:
        -----------
        box1 : [x, y, w, h]
            第一个bounding box，左上角坐标(x,y)，宽w，高h
        box2 : [x, y, w, h]
            第二个bounding box
            
        Returns:
        --------
        float
            IoU值，范围在[0,1]之间
        """
        # 计算box1的边界
        x1_1, y1_1 = box1[0], box1[1]
        x1_2, y1_2 = box1[0] + box1[2], box1[1] + box1[3]
        
        # 计算box2的边界
        x2_1, y2_1 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # 计算交集区域
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        
        # 检查是否有交集
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算box1和box2的面积
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def compute_cost_matrix(self, boxes1: List[List[float]], boxes2: List[List[float]]) -> np.ndarray:
        """
        计算代价矩阵，使用1-IoU作为代价
        
        Parameters:
        -----------
        boxes1 : List[List[float]]
            传感器1的bounding boxes列表
        boxes2 : List[List[float]]
            传感器2的bounding boxes列表
            
        Returns:
        --------
        np.ndarray
            代价矩阵，shape为(n, n)，其中n为目标数量
        """
        n = len(boxes1)
        cost_matrix = np.zeros((n, n))
        
        print("=== 代价矩阵计算过程 ===")
        print("传感器1检测到的目标: {}个".format(n))
        print("传感器2检测到的目标: {}个".format(n))
        print("\n详细计算过程:")
        
        for i in range(n):
            for j in range(n):
                iou = self.compute_iou(boxes1[i], boxes2[j])
                cost = 1.0 - iou  # 1-IoU作为代价
                cost_matrix[i, j] = cost
                
                # 打印详细的计算过程
                print(f"\n计算S1-目标{i}与S2-目标{j}的代价:")
                print(f"  S1-目标{i} bounding box: {boxes1[i]}")
                print(f"  S2-目标{j} bounding box: {boxes2[j]}")
                print(f"  IoU = {iou:.4f}")
                print(f"  代价 = 1 - IoU = {cost:.4f}")
        
        print("\n=== 最终代价矩阵 ===")
        print("C = ")
        for i in range(n):
            row_str = "[" + ", ".join([f"{cost_matrix[i,j]:.4f}" for j in range(n)]) + "]"
            print(f"    {row_str}")
        print()
        
        return cost_matrix
    
    def dual_ascent_algorithm(self, cost_matrix: np.ndarray) -> Dict:
        """
        实现对偶上升算法求解数据关联问题
        
        Parameters:
        -----------
        cost_matrix : np.ndarray
            代价矩阵，shape为(n, n)
            
        Returns:
        --------
        Dict
            包含优化结果和迭代历史的字典
        """
        n = cost_matrix.shape[0]
        
        # 初始化拉格朗日乘子
        lambda_vec = np.zeros(n)  # 行约束乘子
        mu_vec = np.zeros(n)      # 列约束乘子
        
        print("=== 对偶上升法初始化 ===")
        print(f"初始拉格朗日乘子 λ⁰ = {lambda_vec}")
        print(f"初始拉格朗日乘子 μ⁰ = {mu_vec}")
        print(f"步长参数 α = {self.alpha}")
        print(f"最大迭代次数 = {self.max_iterations}\n")
        
        # 存储迭代历史
        history = {
            'assignments': [],
            'lambda_history': [],
            'mu_history': [],
            'objective_values': [],
            'constraint_violations': []
        }
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            print(f"=== 第{iteration + 1}轮迭代 (k={iteration}) ===")
            
            # 1. 计算调整后的代价矩阵 C' = C + λ + μ
            adjusted_cost = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    adjusted_cost[i, j] = cost_matrix[i, j] + lambda_vec[i] + mu_vec[j]
            
            print(f"\n步骤1: 计算调整后的代价矩阵 C'")
            print("C' = ")
            for i in range(n):
                row_str = "[" + ", ".join([f"{adjusted_cost[i,j]:.4f}" for j in range(n)]) + "]"
                print(f"    {row_str}")
            
            # 2. 为每行选择最小代价的列（贪心分配）
            assignment = np.zeros((n, n), dtype=int)
            for i in range(n):
                min_idx = np.argmin(adjusted_cost[i])
                assignment[i, min_idx] = 1
            
            print(f"\n步骤2: 为每行选择最小代价的列")
            print("分配矩阵 X^{} = ".format(iteration))
            for i in range(n):
                row_str = "[" + ", ".join([str(int(assignment[i,j])) for j in range(n)]) + "]"
                print(f"    {row_str}")
            
            # 3. 检查约束违反
            row_sum = np.sum(assignment, axis=1)  # 每行的和
            col_sum = np.sum(assignment, axis=0)  # 每列的和
            
            row_violation = row_sum - 1  # 行约束违反
            col_violation = col_sum - 1  # 列约束违反
            
            print(f"\n步骤3: 检查约束违反")
            print(f"行和: {row_sum}")
            print(f"列和: {col_sum}")
            print(f"行约束违反 r^{iteration} = {row_violation}")
            print(f"列约束违反 s^{iteration} = {col_violation}")
            
            # 4. 计算目标函数值
            objective_value = np.sum(cost_matrix * assignment)
            print(f"\n步骤4: 计算目标函数值")
            print(f"f(X^{iteration}) = Σc_ij·x_ij = {objective_value:.4f}")
            
            # 5. 更新拉格朗日乘子
            lambda_new = lambda_vec + self.alpha * row_violation
            mu_new = mu_vec + self.alpha * col_violation
            
            print(f"\n步骤5: 更新拉格朗日乘子")
            print(f"λ^{iteration+1} = λ^{iteration} + α·r^{iteration} = {lambda_vec} + {self.alpha}×{row_violation} = {lambda_new}")
            print(f"μ^{iteration+1} = μ^{iteration} + α·s^{iteration} = {mu_vec} + {self.alpha}×{col_violation} = {mu_new}")
            
            # 6. 存储历史记录
            history['assignments'].append(assignment.copy())
            history['lambda_history'].append(lambda_vec.copy())
            history['mu_history'].append(mu_vec.copy())
            history['objective_values'].append(objective_value)
            history['constraint_violations'].append({
                'row': row_violation.copy(),
                'col': col_violation.copy()
            })
            
            # 7. 检查收敛性
            total_violation = np.sum(np.abs(row_violation)) + np.sum(np.abs(col_violation))
            print(f"\n收敛性检查:")
            print(f"总约束违反 = {total_violation:.4f}")
            if total_violation < 1e-6:
                print("算法已收敛，满足所有约束条件！")
                break
            
            # 更新乘子
            lambda_vec = lambda_new
            mu_vec = mu_new
            
            print("-" * 60 + "\n")
        
        # 最终结果
        final_assignment = history['assignments'][-1]
        final_objective = history['objective_values'][-1]
        
        result = {
            'assignment_matrix': final_assignment,
            'objective_value': final_objective,
            'lambda_final': lambda_vec,
            'mu_final': mu_vec,
            'history': history,
            'iterations': len(history['assignments']),
            'converged': np.sum(np.abs(np.sum(final_assignment, axis=1) - 1)) + 
                         np.sum(np.abs(np.sum(final_assignment, axis=0) - 1)) < 1e-6
        }
        
        print("=== 优化结果总结 ===")
        print(f"迭代次数: {result['iterations']}")
        print(f"是否收敛: {result['converged']}")
        print(f"最终目标函数值: {final_objective:.4f}")
        print(f"最终分配矩阵:")
        for i in range(n):
            row_str = "[" + ", ".join([str(int(final_assignment[i,j])) for j in range(n)]) + "]"
            print(f"    {row_str}")
        
        return result
    
    def extract_matching_pairs(self, assignment_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        从分配矩阵中提取匹配对
        
        Parameters:
        -----------
        assignment_matrix : np.ndarray
            最终的分配矩阵
            
        Returns:
        --------
        List[Tuple[int, int]]
            匹配对列表 [(sensor1_idx, sensor2_idx), ...]
        """
        n = assignment_matrix.shape[0]
        matching_pairs = []
        
        for i in range(n):
            for j in range(n):
                if assignment_matrix[i, j] == 1:
                    matching_pairs.append((i, j))
        
        return matching_pairs
    
    def visualize_results(self, boxes1: List[List[float]], boxes2: List[List[float]], 
                         matching_pairs: List[Tuple[int, int]], title: str = ""):
        """
        可视化数据关联结果
        
        Parameters:
        -----------
        boxes1 : List[List[float]]
            传感器1的bounding boxes
        boxes2 : List[List[float]]
            传感器2的bounding boxes
        matching_pairs : List[Tuple[int, int]]
            匹配对列表
        title : str
            图表标题
        """
        # 创建一个大的画布，显示两个传感器的检测结果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 生成随机颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(matching_pairs)))
        
        # 传感器1的可视化
        ax1.set_title('传感器S1', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 300)
        ax1.set_ylim(0, 300)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Y坐标', fontsize=12)
        ax1.invert_yaxis()  # 使y轴向下增长，符合图像坐标系
        
        for i, box in enumerate(boxes1):
            rect = Rectangle((box[0], box[1]), box[2], box[3], 
                           fill=False, linewidth=2, color='blue')
            ax1.add_patch(rect)
            ax1.text(box[0] + 5, box[1] + 15, f'目标{i+1}', 
                    color='blue', fontweight='bold', fontsize=10)
        
        # 传感器2的可视化
        ax2.set_title('传感器S2', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 300)
        ax2.set_ylim(0, 300)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.invert_yaxis()
        
        for i, box in enumerate(boxes2):
            rect = Rectangle((box[0], box[1]), box[2], box[3], 
                           fill=False, linewidth=2, color='red')
            ax2.add_patch(rect)
            ax2.text(box[0] + 5, box[1] + 15, f'目标{i+1}', 
                    color='red', fontweight='bold', fontsize=10)
        
        # 创建图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='传感器1目标'),
            Line2D([0], [0], color='red', lw=2, label='传感器2目标')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 显示匹配结果
        print("\n=== 最终匹配结果 ===")
        for i, (idx1, idx2) in enumerate(matching_pairs):
            print(f"匹配对 {i+1}: 传感器1-目标{idx1} ↔ 传感器2-目标{idx2}")
        
        plt.tight_layout()
        plt.savefig('data_association_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建匹配结果的详细图示
        fig2, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 350)
        ax.set_ylim(0, 300)
        ax.grid(True, alpha=0.3)

        # ax.set_title('数据关联匹配结果', fontsize=16, fontweight='bold')
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.invert_yaxis()
        
        # 绘制传感器1的目标（左侧）
        for i, box in enumerate(boxes1):
            rect = Rectangle((box[0], box[1]), box[2], box[3], 
                           fill=False, linewidth=2, color='blue')
            ax.add_patch(rect)
            ax.text(box[0] + 5, box[1] + 15, f'S1-目标{i+1}', 
                   color='blue', fontweight='bold', fontsize=10)
        
        # 绘制传感器2的目标（右侧，x坐标偏移）
        offset_x = 50  # 为了区分，将传感器2的目标向右偏移
        for i, box in enumerate(boxes2):
            rect = Rectangle((box[0] + offset_x, box[1]), box[2], box[3], 
                           fill=False, linewidth=2, color='red')
            ax.add_patch(rect)
            ax.text(box[0] + offset_x + 5, box[1] + 15, f'S2-目标{i+1}', 
                   color='red', fontweight='bold', fontsize=10)
        
        # 绘制匹配连线
        for i, (idx1, idx2) in enumerate(matching_pairs):
            color = colors[i]
            box1 = boxes1[idx1]
            box2 = boxes2[idx2]
            
            # 连接两个目标的中心点
            center1 = (box1[0] + box1[2]/2, box1[1] + box1[3]/2)
            center2 = (box2[0] + offset_x + box2[2]/2, box2[1] + box2[3]/2)
            
            ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                   color=color, linewidth=2, linestyle='--', 
                   label=f'匹配 {idx1+1}, {idx2+1}')
            
            # 在连线上标注匹配信息
            mid_x = (center1[0] + center2[0]) / 2
            mid_y = (center1[1] + center2[1]) / 2
            ax.text(mid_x, mid_y, f'匹配 {i+1}', color=color, 
                   fontweight='bold', fontsize=9, ha='center')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('matching_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    主函数：演示对偶上升法在数据关联问题中的应用
    特别设计的场景确保需要两轮迭代才能达到最优解
    """
    print("=" * 80)
    print("基于对偶上升法的多源传感器数据关联优化")
    print("场景：2个传感器，4个目标，两轮迭代达到最优（特别设计的冲突场景）")
    print("=" * 80 + "\n")
    
    # 1. 设置传感器数据（特别设计的冲突场景）
    print("=== 传感器数据设置（制造冲突，确保需要两轮迭代）===")
    
    # 传感器1检测到的目标（格式：[x, y, w, h]）
    sensor1_boxes = [
        [10, 20, 30, 40],    # 目标0：左上区域
        [15, 25, 28, 38],    # 目标1：与S2-0高度重叠，制造冲突
        [200, 80, 40, 30],   # 目标2：右上区域
        [150, 200, 35, 45]   # 目标3：右下区域
    ]
    
    # 传感器2检测到的目标
    sensor2_boxes = [
        [12, 22, 29, 39],    # 目标0：与S1-0和S1-1都有高重叠
        [45, 35, 27, 33],    # 目标1：需要与S1-1匹配
        [205, 85, 38, 32],   # 目标2：与S1-2匹配
        [145, 195, 37, 43]   # 目标3：与S1-3匹配
    ]
    
    print("传感器1目标bounding boxes:")
    for i, box in enumerate(sensor1_boxes):
        print(f"  目标{i}: [x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}]")
    
    print("\n传感器2目标bounding boxes:")
    for i, box in enumerate(sensor2_boxes):
        print(f"  目标{i}: [x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}]")
    
    # 2. 创建对偶上升法实例
    # 特别设置步长α=0.4以确保两轮迭代达到最优
    da = DualAscentDataAssociation(alpha=0.4, max_iterations=2)
    
    # 3. 计算代价矩阵
    print("\n" + "=" * 60)
    print("步骤1: 计算代价矩阵（1-IoU）")
    print("=" * 60)
    cost_matrix = da.compute_cost_matrix(sensor1_boxes, sensor2_boxes)
    
    # 4. 运行对偶上升算法
    print("\n" + "=" * 60)
    print("步骤2: 运行对偶上升算法")
    print("=" * 60)
    start_time = time.time()
    result = da.dual_ascent_algorithm(cost_matrix)
    end_time = time.time()
    
    print(f"\n算法执行时间: {end_time - start_time:.4f} 秒")
    
    # 5. 提取匹配对
    matching_pairs = da.extract_matching_pairs(result['assignment_matrix'])
    
    # 6. 可视化结果
    print("\n" + "=" * 60)
    print("步骤3: 可视化结果")
    print("=" * 60)
    da.visualize_results(sensor1_boxes, sensor2_boxes, matching_pairs)
    
    # 7. 验证结果
    print("\n" + "=" * 60)
    print("结果验证")
    print("=" * 60)
    
    # 真实匹配（根据数据设置）
    true_matching = [(0,0), (1,1), (2,2), (3,3)]
    
    # 计算匹配准确率
    correct_matches = sum(1 for pair in matching_pairs if pair in true_matching)
    accuracy = correct_matches / len(true_matching) * 100
    
    print(f"匹配准确率: {accuracy:.1f}%")
    print(f"正确匹配数: {correct_matches}/{len(true_matching)}")
    
    # 计算最优目标函数值
    optimal_objective = sum(cost_matrix[i,j] for i,j in true_matching)
    print(f"理论最优目标函数值: {optimal_objective:.4f}")
    print(f"算法达到的目标函数值: {result['objective_value']:.4f}")
    print(f"与最优值的差距: {abs(result['objective_value'] - optimal_objective):.6f}")
    
    if accuracy == 100.0 and abs(result['objective_value'] - optimal_objective) < 1e-6:
        print("✓ 算法在两轮迭代内达到全局最优解！")
    else:
        print("⚠ 算法未达到最优解，需要检查参数设置")
    
    # 8. 迭代过程详细分析
    print("\n" + "=" * 60)
    print("迭代过程详细分析")
    print("=" * 60)
    
    history = result['history']
    for i in range(len(history['assignments'])):
        print(f"\n第{i+1}轮迭代结果:")
        print(f"  拉格朗日乘子 λ^{i}: {history['lambda_history'][i]}")
        print(f"  拉格朗日乘子 μ^{i}: {history['mu_history'][i]}")
        print(f"  目标函数值: {history['objective_values'][i]:.4f}")
        
        # 显示约束违反
        violations = history['constraint_violations'][i]
        print(f"  行约束违反: {violations['row']}")
        print(f"  列约束违反: {violations['col']}")
        
        # 显示分配矩阵
        print("  分配矩阵:")
        assignment = history['assignments'][i]
        for j in range(4):
            row_str = "[" + ", ".join([str(int(assignment[j,k])) for k in range(4)]) + "]"
            print(f"    {row_str}")


if __name__ == "__main__":
    main()