def solve_cp_portfolio():
    from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

    # 创建模型
    model = Model("CP_Portfolio_Optimization")

    # 假设有4个资产，分属于两个行业
    assets = ['Asset1', 'Asset2', 'Asset3', 'Asset4']
    returns = {'Asset1': 0.10, 'Asset2': 0.20, 'Asset3': 0.15, 'Asset4': 0.12}
    budget = 1.0
    industries = {'Asset1': 'Tech', 'Asset2': 'Finance', 'Asset3': 'Tech', 'Asset4': 'Healthcare'}

    # 定义决策变量：每个资产的投资比例
    x = {asset: model.addVar(vtype="CONTINUOUS", name=f"x_{asset}") for asset in assets}
    # 二进制变量：是否投资该资产
    y = {asset: model.addVar(vtype="BINARY", name=f"y_{asset}") for asset in assets}

    # 目标函数：最大化预期收益
    model.setObjective(quicksum(returns[i] * x[i] for i in assets), "maximize")

    # 约束条件：总投资比例为1
    model.addCons(quicksum(x[i] for i in assets) == budget, "Budget")

    # 引入复杂约束：每个行业至少投资一定比例
    industry_requirements = {'Tech': 0.3, 'Finance': 0.2, 'Healthcare': 0.1}
    for industry, req in industry_requirements.items():
        model.addCons(
            quicksum(x[i] for i in assets if industries[i] == industry) >= req,
            f"Industry_{industry}_Requirement"
        )

    # 限制同时投资的资产数量
    max_assets = 3
    model.addCons(quicksum(y[i] for i in assets) <= max_assets, "MaxAssets")
    for asset in assets:
        model.addCons(x[asset] <= y[asset], f"Link_{asset}")

    # 求解模型
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setParam("display/verblevel", 3)
    model.optimize()

    # 输出结果
    print("CP 投资组合优化结果:")
    selected = []
    for asset in assets:
        investment = model.getVal(x[asset])
        if investment > 1e-6:
            selected.append(asset)
            print(f"{asset}: {investment} (Industry: {industries[asset]})")
    print(f"选择的资产数量: {len(selected)}")
    print(f"预期收益: {model.getObjVal()}") 

    status = model.getStatus()
    print(f"Optimization Status: {status}")

if __name__ == "__main__":
    solve_cp_portfolio() 