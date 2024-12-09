def solve_lp_portfolio():
    from pyscipopt import Model, quicksum

    # 创建模型
    model = Model("LP_Portfolio_Optimization")

    # 假设有3个资产
    assets = ['Asset1', 'Asset2', 'Asset3']
    returns = {'Asset1': 0.10, 'Asset2': 0.20, 'Asset3': 0.15}
    budget = 1.0

    # 定义决策变量：每个资产的投资比例
    x = {asset: model.addVar(vtype="CONTINUOUS", name=asset) for asset in assets}

    # 目标函数：最大化预期收益
    model.setObjective(quicksum(returns[i] * x[i] for i in assets), "maximize")

    # 约束条件：总投资比例为1
    model.addCons(quicksum(x[i] for i in assets) == budget, "Budget")

    # 线性风险约束（例如，最大方差）
    risk = 0.05
    variances = {'Asset1': 0.02, 'Asset2': 0.03, 'Asset3': 0.025}
    model.addCons(quicksum(variances[i] * x[i] for i in assets) <= risk, "Risk")

    # 求解模型
    model.optimize()

    # 输出结果
    print("LP 投资组合优化结果:")
    for asset in assets:
        print(f"{asset}: {model.getVal(x[asset])}")
    print(f"预期收益: {model.getObjVal()}")

if __name__ == "__main__":
    solve_lp_portfolio() 