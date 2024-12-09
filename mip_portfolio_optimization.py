def solve_mip_portfolio():
    from pyscipopt import Model, quicksum

    # 创建模型
    model = Model("MIP_Portfolio_Optimization")

    # 假设有5个资产
    assets = ['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
    returns = {'Asset1': 0.10, 'Asset2': 0.20, 'Asset3': 0.15, 'Asset4': 0.12, 'Asset5': 0.18}
    budget = 1.0
    max_assets = 2

    # 定义决策变量：每个资产的投资比例
    x = {asset: model.addVar(vtype="CONTINUOUS", name=f"x_{asset}") for asset in assets}
    # 二进制变量：是否投资该资产
    y = {asset: model.addVar(vtype="BINARY", name=f"y_{asset}") for asset in assets}

    # 目标函数：最大化预期收益
    model.setObjective(quicksum(returns[i] * x[i] for i in assets), "maximize")

    # 约束条件：总投资比例为1
    model.addCons(quicksum(x[i] for i in assets) == budget, "Budget")

    # 约束条件：最多选择max_assets个资产
    model.addCons(quicksum(y[i] for i in assets) <= max_assets, "MaxAssets")

    # 关联x和y变量
    for asset in assets:
        model.addCons(x[asset] <= y[asset], f"Link_{asset}")

    # 求解模型
    model.optimize()

    # 输出结果
    print("MIP 投资组合优化结果:")
    selected = []
    for asset in assets:
        investment = model.getVal(x[asset])
        if investment > 1e-6:
            selected.append(asset)
            print(f"{asset}: {investment}")
    print(f"选择的资产数量: {len(selected)}")
    print(f"预期收益: {model.getObjVal()}")

if __name__ == "__main__":
    solve_mip_portfolio() 