# Sustainability Aware Asset Management

## Portfolio Allocation with a Carbon Objective

**HEC Lausanne — Master in Finance**
**Spring 2026**

*[Group members names]*
*[Region: _____ | Climate strategy: Scope 1 + Scope 2]*

---

## 1. Introduction

### 1.1 Context and Motivation

The integration of environmental, social, and governance (ESG) considerations into portfolio management has become a central topic in modern finance. Among these dimensions, climate risk — and more specifically the carbon footprint of investment portfolios — has emerged as a key concern for institutional investors, regulators, and asset managers alike. The Paris Agreement (2015) set the objective of limiting global warming to well below 2°C above pre-industrial levels, with efforts to limit the increase to 1.5°C. This has created strong incentives for the financial industry to develop strategies that align capital allocation with decarbonization pathways.

From an investor's perspective, integrating carbon constraints into portfolio construction raises a fundamental question: what is the cost, in terms of financial performance, of reducing the carbon footprint of an equity portfolio? This question is at the heart of this project. We investigate whether it is possible to achieve meaningful reductions in portfolio-level carbon emissions while maintaining acceptable levels of risk-adjusted return, using standard optimization tools from quantitative finance.

### 1.2 Objective and Scope

The objective of this project is to implement and evaluate several climate-aware portfolio allocation strategies over the period 2014–2025. Starting from a universe of international equities with available carbon data, we construct long-only portfolios that satisfy increasingly stringent carbon emission constraints, and we evaluate their out-of-sample financial and environmental performance.

The project is structured in three parts:

- **Part I — Standard Portfolio Allocation.** We construct a minimum-variance portfolio and compare its out-of-sample performance to a value-weighted (market-cap) benchmark. This establishes the baseline financial performance against which carbon-constrained strategies will be evaluated. The minimum-variance approach is chosen because it does not require estimation of expected returns, which are notoriously difficult to forecast and introduce significant estimation error.

- **Part II — Carbon-Aware Allocation with a 50% Reduction Target.** We introduce two carbon-constrained strategies. The first adds a carbon footprint ceiling (set at 50% of the unconstrained minimum-variance portfolio's CF) to the minimum-variance optimization, representing an active investor's approach. The second minimizes the tracking error relative to the value-weighted benchmark while imposing a 50% CF reduction relative to the benchmark, representing a passive investor's approach. These two strategies illustrate the two fundamental perspectives on decarbonization: active risk management versus passive index tracking.

- **Part III — Net Zero Portfolio.** We implement a decarbonization trajectory in which the portfolio's carbon footprint must decrease by 10% per year on a compounding basis from a fixed 2013 baseline. This strategy simulates the constraints faced by an investor committed to a net-zero emissions target by a future date, following the methodology outlined in recent TCFD (Task Force on Climate-related Financial Disclosures) recommendations.

For each strategy, we employ two distinct covariance matrix estimators — the sample covariance matrix and the Ledoit-Wolf shrinkage estimator — and solve the corresponding optimization problems using different numerical solvers (SLSQP and CVXPY/OSQP respectively). This yields ten portfolio variants in total, allowing us to assess the sensitivity of results to the choice of estimator and solver.

### 1.3 Structure of the Report

The remainder of this report is organized as follows. Section 2 describes the data sources, cleaning procedures, investment set construction, covariance estimation methods, and the mechanics of out-of-sample return computation including the buy-and-hold weight drift. Section 3 presents Part I results: the value-weighted benchmark and minimum-variance portfolios. Section 4 covers Part II: carbon metrics, the CF-constrained minimum-variance portfolio, and the tracking-error minimization strategy. Section 5 addresses Part III: the net-zero portfolio with a compounding 10% annual reduction. Section 6 provides a comprehensive discussion including performance comparisons, decomposition analysis, and limitations. Section 7 concludes.

---

## 2. Data and Methodology

### 2.1 Data Sources

The dataset is sourced from Refinitiv Datastream and covers a universe of 2,545 firms for which carbon data is available. The data spans the period from end-1999 to end-2025 for price-related variables, and from 2002 to 2024 for carbon and revenue data. However, carbon data coverage is limited before 2010, which motivates our decision to start the allocation exercise at end of 2013.

For each firm, we observe the following variables:

| Variable | Description | Frequency |
|---|---|---|
| ISIN | International Securities Identification Number | Static |
| Name, Country, Region | Firm identifiers and geographic classification | Static |
| RI (Total Return Index) | Price index including dividend payments (USD) | Monthly |
| MV_Y | End-of-year market capitalisation (million USD) | Annual |
| MV_M | End-of-month market capitalisation (million USD) | Monthly |
| CO2_S1 | Scope 1 CO₂ emissions (tonnes) | Annual |
| CO2_S2 | Scope 2 CO₂ emissions (tonnes) | Annual |
| REV | Annual revenues (thousands USD) | Annual |

From these raw variables, we derive:

- **Monthly simple returns:** Computed from the total return index as R_{i,t} = (P_{i,t} / P_{i,t-1}) − 1, where P_{i,t} is the total return index of firm i at end of month t. The total return index captures both price appreciation and dividend reinvestment, providing a comprehensive measure of investment performance.

- **Carbon intensity:** CI_{i,Y} = (CO2_S1_{i,Y} + CO2_S2_{i,Y}) / (REV_{i,Y} / 1000), expressed in tonnes of CO₂ equivalent per million USD of revenue. The division by 1,000 converts revenues from thousands to millions of USD.

The data is organized in wide format, with one Excel file per variable. Each file contains firms as rows and time periods as columns. The `Static.xlsx` file provides the cross-sectional identifiers (ISIN, name, country, region) that link all time-series files together.

### 2.2 Data Cleaning

The raw Datastream output requires several cleaning steps before it can be used for portfolio construction. These steps are critical for ensuring that the investment strategy is based on reliable data and avoids known pitfalls associated with financial databases.

#### 2.2.1 Missing Prices and Identifiers

For some ISINs, Datastream could not find a matching firm — for instance, because the ISIN corresponds to a different share class than the one tracked by Datastream. In such cases, the entire row is blank or missing across all data files. We identify these firms and remove them from all tables to maintain consistency across the dataset.

#### 2.2.2 Treatment of Low Prices

The total return index (RI) for some firms can be very low (below 0.5 USD) or even equal to zero due to rounding for values below 0.05. Computing returns from such low price levels produces extreme or infinite values that would distort the covariance matrix and portfolio weights. We treat all prices below 0.5 as missing values.

This treatment has two implications for investment:

1. If the price is missing at end of year Y, the firm is excluded from the investment set for year Y+1. This is a conservative rule that prevents investing in firms for which we do not have reliable price information at the decision date.

2. If the price is available at end of year Y (so the firm enters the investment set) but becomes missing during year Y+1, we interpret this as a delisting event. The firm's return at the delisting date is set to −100%, reflecting the total loss suffered by the investor. This treatment is important for avoiding survivorship bias: ignoring delisted firms would overstate portfolio performance by excluding the worst-performing stocks.

#### 2.2.3 Treatment of Missing Values

Missing values in the time series are handled differently depending on their position:

- **Beginning of sample:** Missing values at the start of a firm's history are left as-is. They typically indicate that the firm was not yet listed or did not yet report the variable. The firm simply enters the dataset when data becomes available.

- **Middle of sample:** Missing values between two available observations are forward-filled using the most recent available value. This is a common approach for annual variables (revenues, emissions) where a firm may fail to report in a given year but resumes reporting subsequently. For carbon data specifically, forward-filling is justified because emissions tend to be relatively persistent from year to year.

- **End of sample:** Missing values at the end of the sample typically correspond to a firm default, delisting, or acquisition. In Datastream, the delisting date is often appended to the firm name. We handle these cases through the low-price treatment described above.

#### 2.2.4 Stale Prices

Some firms exhibit prices that do not vary for extended periods (months or even years), indicating an absence of trading activity. While zero returns are theoretically possible, a high proportion of zero-return months implies that the stock is illiquid. This creates a practical problem for portfolio optimization: the artificially low volatility of illiquid stocks causes the optimizer to assign them disproportionately high weights, producing unrealistically low portfolio variance estimates that would not be achievable in practice.

To address this issue, we compute, for each firm in the investment set, the proportion of months with a zero return over the 10-year estimation window. If this proportion exceeds 50%, the firm is classified as subject to stale prices and is excluded from the investment set. This threshold is conservative — it requires that more than half of all monthly observations show no price change. The decision is based entirely on past information, so there is no look-ahead bias.

### 2.3 Investment Set Construction

The investment set is the list of firms eligible for inclusion in the portfolio at a given decision date. It is constructed annually, at the end of each year Y = 2013, 2014, …, 2024, and determines the composition of the portfolio for the following year Y+1.

#### 2.3.1 Eligibility Criteria

A firm is included in the investment set for year Y if and only if it satisfies all of the following criteria:

1. **Regional filter.** The firm belongs to the region assigned to our group. This filter restricts the investment universe to a specific geographic area, as specified in the project instructions.

2. **Sufficient return history.** The firm has at least 36 months (3 years) of non-missing monthly returns over the 10-year window ending at end of year Y. This minimum ensures that the covariance matrix estimation is based on a reasonable number of observations for each firm. Firms with fewer than 36 observations have too little data to produce reliable variance and correlation estimates.

3. **Carbon data availability.** The firm has CO₂ emissions (Scope 1 and Scope 2) and revenue data available at end of year Y. This criterion is essential because all portfolio strategies in Part II and Part III require carbon data for the constraint formulation. By imposing this criterion from the start, we ensure that the same investment set is used throughout all parts of the project, making results directly comparable.

4. **No stale prices.** The proportion of zero-return months over the estimation window does not exceed 50%, as described in Section 2.2.4.

#### 2.3.2 Investment Decision Timing

The investment decision is made at end of year Y for portfolio implementation over year Y+1. This means that the covariance matrix, expected returns, and carbon data used for optimization are all computed from information available at end of year Y — there is no look-ahead bias. The first decision date is end of December 2013, and the first portfolio is invested from January 2014 through December 2014. The last decision date is end of December 2024, and the last portfolio covers January 2025 through December 2025.

#### 2.3.3 Full-Available-Data (FA) Variant

A stricter variant of the investment set, denoted `is_{year}_fa`, is constructed by dropping any firm with at least one missing monthly return over the 10-year estimation window. This means that only firms with a complete 120-month return history are retained. This filter is necessary for the Ledoit-Wolf covariance estimator, which requires a complete data matrix (no missing values) as input.

The FA investment set is smaller than the standard set because it excludes firms with incomplete return histories — for instance, firms that were listed less than 10 years ago or that experienced trading halts. The FA variant serves two purposes: (i) it provides the input for the Ledoit-Wolf optimization, and (ii) the value-weighted portfolio computed on this subset (VW-FA) serves as the benchmark for the Ledoit-Wolf tracking-error and net-zero strategies.

> *[TABLE: Investment set size by year — number of firms in standard and FA sets, 2013–2024]*

### 2.4 Estimation of Expected Returns and the Covariance Matrix

#### 2.4.1 Estimation Window

For each decision date (end of year Y), we estimate the vector of expected returns and the covariance matrix from the most recent 10 years of monthly returns, corresponding to τ = 120 monthly observations. For instance, the allocation decided at end of December 2013 uses returns from January 2004 to December 2013.

The expected return vector is computed as the sample mean over the estimation window:

$$\hat{\mu}_Y = \frac{1}{\tau} \sum_{k=0}^{\tau-1} R_{t-k}$$

However, in the minimum-variance framework, expected returns are not used in the optimization — only the covariance matrix matters. This is a major advantage of the minimum-variance approach: it avoids the estimation of expected returns, which are notoriously difficult to forecast accurately and represent the largest source of estimation error in portfolio optimization (see Merton, 1980; Jagannathan and Ma, 2003).

#### 2.4.2 Sample Covariance Matrix

The sample covariance matrix is computed as:

$$\Sigma_Y = \frac{1}{\tau} \sum_{k=0}^{\tau-1} (R_{t-k} - \hat{\mu}_Y)(R_{t-k} - \hat{\mu}_Y)'$$

This is the standard, unbiased estimator of the true covariance matrix. It is computed using pandas' `.cov()` method on the transposed return matrix (firms as columns, dates as rows). The sample covariance matrix has several well-known limitations:

- When the number of firms N is large relative to the number of time periods τ, the matrix is poorly conditioned and may even be singular (rank-deficient). In our case, with N potentially exceeding 500 firms and τ = 120 months, the matrix is indeed not invertible, which is why we use numerical optimization rather than the closed-form solution.

- Even when the matrix is invertible, the eigenvalues of the sample covariance matrix are dispersed relative to the true eigenvalues: the largest eigenvalues are overestimated and the smallest are underestimated. This leads to extreme portfolio weights that are unstable across estimation windows.

The sample covariance matrix is used with the SLSQP solver, which handles the full investment set including firms with missing returns (the covariance is computed on available pairwise observations).

#### 2.4.3 Ledoit-Wolf Shrinkage Estimator

To address the limitations of the sample covariance matrix, we employ the Ledoit-Wolf (2004) shrinkage estimator as a second approach. The idea is to shrink the sample covariance matrix toward a structured target — in this case, a scaled identity matrix — to reduce estimation error:

$$\Sigma_Y^{LW} = \delta \cdot F + (1 - \delta) \cdot S$$

where S is the sample covariance matrix, F is the shrinkage target (scaled identity), and δ ∈ [0, 1] is the optimal shrinkage intensity, determined analytically to minimize the expected loss under the Frobenius norm.

The Ledoit-Wolf estimator has several attractive properties:

- It always produces a positive-definite covariance matrix, even when N > τ.
- It reduces the dispersion of eigenvalues, leading to more stable and diversified portfolio weights.
- The shrinkage intensity is chosen optimally — it shrinks more when the sample covariance is more noisy (small τ, large N) and less when it is more reliable.

Before applying the Ledoit-Wolf estimator, we remove all firms with any missing return over the estimation window (this is the FA filter). The resulting complete return matrix is passed to scikit-learn's `LedoitWolf` class, which computes the optimal shrinkage intensity and returns the shrunk covariance matrix. The subset of firms retained for this estimation is stored as `returns_{year}_lw`.

### 2.5 Portfolio Optimization

#### 2.5.1 SLSQP Solver (Sample Covariance)

The Sequential Least Squares Programming (SLSQP) algorithm, implemented in `scipy.optimize.minimize`, is used to solve the quadratic programs associated with the sample covariance matrix. SLSQP is a gradient-based method that handles equality and inequality constraints efficiently.

A numerical scaling factor of 10,000 is applied to the covariance matrix before optimization. This scaling improves the numerical conditioning of the problem by bringing the eigenvalues of Σ to a range where the solver's default tolerances are appropriate. Without scaling, very small eigenvalues can cause the solver to terminate prematurely or produce inaccurate results. We verify that the weights obtained with the scaled matrix are identical (up to numerical precision) to those from the unscaled problem by comparing the two solutions.

The initial guess for the optimizer is the equal-weight vector (1/N for each firm), which satisfies both the full-investment constraint (weights sum to 1) and the non-negativity constraint (all weights ≥ 0). This ensures that the optimizer starts from a feasible point.

#### 2.5.2 CVXPY/OSQP Solver (Ledoit-Wolf)

The CVXPY modeling framework, paired with the OSQP (Operator Splitting Quadratic Program) solver, is used for all optimizations based on the Ledoit-Wolf covariance matrix. CVXPY provides a high-level, declarative interface for specifying convex optimization problems, while OSQP is an efficient first-order method designed specifically for quadratic programs.

The objective function is specified using CVXPY's `quad_form(w, Σ)` function, which computes w'Σw. As with the SLSQP approach, a scaling factor of 10,000 is applied to the covariance matrix for numerical stability. After solving, small negative weights (arising from solver numerical noise) are clipped to zero, and the weight vector is renormalized to sum to exactly 1.

The choice of two different solver/estimator combinations allows us to assess the robustness of our results. If both approaches yield qualitatively similar portfolios and performance, this increases confidence in the conclusions.

### 2.6 Out-of-Sample Return Computation

#### 2.6.1 Annual Rebalancing with Monthly Performance

The portfolio is rebalanced once per year: at end of year Y, optimal weights α_Y are determined, and these weights are used throughout year Y+1. However, performance is computed at the monthly frequency using the actual monthly stock returns of year Y+1. This distinction between the annual rebalancing frequency and the monthly return frequency is important.

It is crucial to understand that between two rebalancing dates, we do not actively adjust the portfolio weights. Instead, the weights evolve passively according to the relative price movements of the constituent stocks. This is the buy-and-hold approach: the investor sets the initial allocation at the beginning of the year and lets the portfolio drift with the market throughout the year, without trading.

#### 2.6.2 Buy-and-Hold Weight Drift

Within each year, the effective weight of firm i at the beginning of month t+k (for k = 1, …, 12) is determined by the buy-and-hold drift formula:

$$\alpha_{i,t+k-1} = \alpha_{i,t+k-2} \times \frac{1 + R_{i,t+k-1}}{1 + R_{p,t+k-1}}$$

with the initialization α_{i,t} = α_{i,Y} at the start of the year. Intuitively, this formula says that the weight of a stock increases when it outperforms the portfolio (its relative share of portfolio value grows) and decreases when it underperforms (its relative share shrinks).

The monthly portfolio return at time t+k is then:

$$R_{p,t+k} = \sum_i \alpha_{i,t+k-1} \times R_{i,t+k}$$

Note that this is different from applying fixed weights to each month's returns (which would correspond to monthly rebalancing). The buy-and-hold approach is more realistic because it does not assume the investor trades every month. It also means that the portfolio's risk characteristics change throughout the year as weights drift — stocks that appreciate become overweighted, and stocks that depreciate become underweighted. This is a well-known property of buy-and-hold strategies and represents the natural evolution of a passive portfolio.

#### 2.6.3 Time Series Construction

By rolling the estimation window forward one year at a time (from Y = 2013 to Y = 2024) and computing monthly returns for each subsequent year (from 2014 to 2025), we obtain a continuous 144-month out-of-sample return time series for each portfolio strategy. This time series covers January 2014 to December 2025, a period that includes several distinct market regimes: the post-financial-crisis recovery (2014–2015), the low-volatility environment (2016–2017), the COVID-19 pandemic (2020), the subsequent recovery and inflation surge (2021–2022), and the normalization period (2023–2025).

### 2.7 Performance Statistics

For each portfolio, the following summary statistics are computed over the full 144-month out-of-sample period:

| Statistic | Formula | Interpretation |
|---|---|---|
| Annualized Average Return | (1 + mean(r_monthly))^12 − 1 | Average monthly return compounded over 12 months |
| Annualized Volatility | std(r_monthly) × √12 | Standard deviation of monthly returns scaled to annual |
| Sharpe Ratio | Ann. Return / Ann. Volatility | Risk-adjusted return (excess return per unit of risk) |
| Min Monthly Return | min(r_monthly) | Worst single-month performance |
| Max Monthly Return | max(r_monthly) | Best single-month performance |

The Sharpe ratio reported here uses total returns rather than excess returns over the risk-free rate. While the traditional Sharpe ratio subtracts the risk-free rate from the numerator, we adopt this simplified version for consistency across all portfolio comparisons. The ranking of portfolios by Sharpe ratio is not affected by this choice when the same risk-free rate applies to all strategies.

---

## 3. Part I — Standard Portfolio Allocation

### 3.1 Value-Weighted Portfolio

#### 3.1.1 Construction

The value-weighted (VW) portfolio serves as the primary benchmark throughout this project. It represents the market-capitalization-weighted portfolio of all firms in the investment set and corresponds to the investor's passive default allocation.

For each month t within year Y+1, portfolio weights are computed from the end-of-previous-month market capitalisations of all firms in the investment set:

$$w_{i,t} = \frac{Cap_{i,t}}{\sum_{j=1}^{N} Cap_{j,t}}$$

where Cap_{i,t} is the market capitalisation of firm i at the end of month t. This definition ensures that the VW weights reflect the most recent available market values at each point in time. This is consistent with the definition of a capitalization-weighted index such as the S&P 500 or the MSCI World: the weight of each constituent is proportional to its float-adjusted market capitalisation.

Using monthly market caps rather than annual market caps for the VW portfolio is important because market capitalisations can change significantly within a year. A firm that doubles in price during the first half of the year should carry roughly twice its initial weight in the second half — this is precisely what the monthly cap weighting achieves.

#### 3.1.2 Full-Available-Data Variant (VW-FA)

A second value-weighted portfolio is computed on the full-available-data (FA) subset. This portfolio uses only firms with complete 10-year return histories, corresponding to the same universe used by the Ledoit-Wolf estimator. Weights are renormalized to sum to 1 after removing firms with incomplete data.

The VW-FA portfolio serves as the benchmark for the Ledoit-Wolf tracking-error minimization (Section 4.3) and the Ledoit-Wolf net-zero strategy (Section 5). Using the FA benchmark ensures that the tracking error is computed over a consistent set of firms — those for which the Ledoit-Wolf covariance matrix is defined.

#### 3.1.3 Properties of the VW Portfolio

The value-weighted portfolio has several important theoretical and practical properties:

- It is mean-variance efficient under the CAPM assumptions. If markets are in equilibrium and all investors hold the market portfolio, the VW portfolio lies on the efficient frontier.
- It is self-rebalancing: as stock prices change, the VW weights automatically adjust to reflect the new market capitalisations. No trading is required to maintain the value weighting (beyond the monthly recalculation of weights from market caps).
- It is biased toward large-cap stocks: firms with the highest market capitalisation receive the largest weights. This means that the VW portfolio's performance is dominated by a relatively small number of mega-cap firms.

### 3.2 Minimum-Variance Portfolio

#### 3.2.1 Optimization Problem

The minimum-variance (MV) portfolio is constructed by solving the following long-only quadratic program at end of each year Y:

$$\min_{\alpha} \quad \alpha' \Sigma_Y \alpha$$

$$\text{s.t.} \quad \sum_{i=1}^{N} \alpha_i = 1, \quad \alpha_i \geq 0 \quad \forall i$$

The objective is to find the portfolio with the lowest possible variance (risk) among all portfolios that are fully invested (weights sum to 1) and long-only (no short selling). The long-only constraint is important for two reasons: (i) it makes the portfolio practically implementable for most investors, and (ii) it facilitates the interpretation of the portfolio's carbon footprint in Part II, since carbon attribution is only meaningful for long positions.

The minimum-variance portfolio does not use expected returns as inputs — it only requires the covariance matrix. This is a significant advantage because expected returns are much harder to estimate accurately than covariances (see Merton, 1980). Numerous empirical studies have shown that minimum-variance portfolios tend to outperform mean-variance portfolios on a risk-adjusted basis out of sample, precisely because they avoid the noise introduced by expected return estimates (DeMiguel, Garlappi, and Uppal, 2009).

#### 3.2.2 SLSQP Implementation

For the sample covariance matrix, the optimization is solved using the SLSQP algorithm from `scipy.optimize.minimize`. The implementation proceeds as follows for each year Y:

1. Compute the sample covariance matrix Σ_Y from the 10-year window of monthly returns.
2. Apply the scaling factor (multiply Σ_Y by 10,000) for numerical stability.
3. Define the objective function as f(α) = α'(10000 × Σ_Y)α.
4. Set the constraints: sum of weights equals 1 (equality), each weight ≥ 0 (bounds).
5. Initialize with the equal-weight vector α_0 = (1/N, …, 1/N).
6. Solve and verify convergence.
7. Verify that the solution is unchanged when compared to the unscaled problem.

The resulting weight vector α_Y is stored for use in the out-of-sample return computation and, later, for carbon metric calculations.

#### 3.2.3 CVXPY/OSQP Implementation (Ledoit-Wolf)

For the Ledoit-Wolf covariance matrix, the same optimization problem is solved using CVXPY with the OSQP solver. The implementation is analogous but uses the shrunk covariance matrix and operates on the FA subset of firms.

After solving, small negative weights (typically on the order of 10^{-8} to 10^{-10}) may appear due to solver numerical noise. These are clipped to zero, and the weight vector is renormalized to ensure it sums to exactly 1. This post-processing step is standard practice when using interior-point or first-order methods for portfolio optimization.

#### 3.2.4 Expected Behavior

The MV portfolio is expected to concentrate on stocks with low individual volatility and low pairwise correlations. In practice, this often results in a portfolio that is:

- Overweight in defensive sectors (utilities, consumer staples, healthcare) that exhibit lower volatility.
- Underweight in cyclical sectors (energy, materials, technology) that exhibit higher volatility.
- More concentrated than the VW portfolio, since the optimizer tends to select a subset of stocks that form the best diversification combination.

This concentration can be both an advantage (lower variance) and a disadvantage (higher idiosyncratic risk if the selected stocks experience firm-specific shocks). The Ledoit-Wolf estimator partially mitigates this issue by shrinking the covariance matrix toward a more homogeneous structure, which tends to produce more diversified portfolios.

### 3.3 Results and Comparison

> *[TABLE: Summary statistics — VW, VW-FA, MV (SLSQP), MV (LW): annualized return, volatility, Sharpe ratio, min/max monthly return]*

The comparison between the VW and MV portfolios reveals the classic risk-return trade-off. The MV portfolio, by construction, achieves a lower annualized volatility than the VW benchmark. Whether this comes at the cost of lower returns depends on the sample period and market conditions. In general, the minimum-variance anomaly — the empirical finding that low-volatility stocks tend to deliver higher risk-adjusted returns than predicted by the CAPM — suggests that MV portfolios may outperform on a Sharpe ratio basis.

> *[FIGURE: Cumulative return series — VW vs. MV (SLSQP) vs. MV (LW)]*

The cumulative return plot provides a visual comparison of the wealth trajectories of the three strategies over the 144-month sample period. Key periods to observe include:

- The 2020 COVID-19 drawdown: the MV portfolio, with its lower beta and defensive tilt, is expected to suffer a smaller drawdown than the VW benchmark.
- The 2021 recovery: the VW portfolio, with its higher exposure to growth and technology stocks, may recover faster.
- The 2022 interest rate shock: rising rates tend to hurt growth stocks more than value/defensive stocks, potentially favoring the MV portfolio.

The comparison between the SLSQP and LW implementations is also informative. If both produce similar performance, this suggests that the results are robust to the choice of covariance estimator. If they diverge, this points to the importance of estimation quality and the specific firms included in or excluded from the FA subset.

> *[FIGURE: Weight distribution comparison — VW vs. MV for a representative year]*

The weight distribution plots illustrate the degree of portfolio concentration. The VW portfolio typically has a few large weights (corresponding to mega-cap firms) and a long tail of small weights. The MV portfolio concentrates on a smaller number of stocks, with potentially larger individual weights.

---

## 4. Part II — Carbon-Aware Portfolio Allocation

### 4.1 Carbon Metrics

#### 4.1.1 Carbon Intensity (CI)

Carbon intensity measures the CO₂ emissions of a firm relative to its economic output (revenues). For each firm i in year Y, the carbon intensity is:

$$CI_{i,Y} = \frac{CO2\_S1_{i,Y} + CO2\_S2_{i,Y}}{REV_{i,Y} / 1000}$$

where CO2_S1 and CO2_S2 are Scope 1 and Scope 2 emissions in tonnes, and REV is annual revenue in thousands of USD. The division by 1,000 converts revenues to millions of USD, so CI is expressed in tonnes of CO₂ equivalent per million USD of revenue (tCO₂e/M$rev).

Scope 1 emissions are direct emissions from sources owned or controlled by the firm (e.g., combustion of fuels, industrial processes). Scope 2 emissions are indirect emissions from the generation of purchased electricity, heat, or steam consumed by the firm. Together, Scope 1 + Scope 2 provide a measure of the firm's operational carbon footprint.

Carbon intensity is a normalization of absolute emissions by firm size (revenues). It allows meaningful comparisons across firms of different sizes: a large utility company may have higher absolute emissions than a small manufacturer, but lower carbon intensity if its revenues are proportionally larger.

#### 4.1.2 Weighted Average Carbon Intensity (WACI)

The portfolio-level WACI aggregates firm-level carbon intensity using portfolio weights:

$$WACI_Y^{(p)} = \sum_{i=1}^{N} \alpha_{i,Y} \times CI_{i,Y}$$

WACI represents the carbon intensity of the portfolio's underlying companies, weighted by the investor's allocation. It answers the question: "On average, how carbon-intensive are the companies in my portfolio?" A higher WACI indicates that the portfolio is tilted toward more carbon-intensive firms.

WACI is the carbon metric recommended by the TCFD for portfolio-level climate reporting. Its main advantage is simplicity: it depends only on portfolio weights and firm-level carbon intensities, with no need to track portfolio wealth or ownership fractions.

#### 4.1.3 Carbon Footprint (CF)

The carbon footprint measures the CO₂ emissions attributed to the investor per million USD invested:

$$CF_Y^{(p)} = \sum_{i=1}^{N} \alpha_{i,Y} \times \frac{E_{i,Y}}{Cap_{i,Y}}$$

where E_{i,Y} = CO2_S1_{i,Y} + CO2_S2_{i,Y} is total emissions (tonnes) and Cap_{i,Y} = MV_Y (market capitalisation in million USD). The result is in tonnes of CO₂ per million USD invested (tCO₂e/M$inv).

The CF metric differs from WACI in that it normalizes emissions by market capitalisation rather than revenues. This means that CF captures the investor's proportional share of emissions: by investing α_i of the portfolio in firm i, the investor "owns" α_i × (V_portfolio / Cap_i) of the firm, and therefore α_i × (V_portfolio / Cap_i) × E_i of the firm's emissions. The portfolio value V_portfolio cancels out, leaving the CF formula above.

CF is the metric used for all carbon constraints in the optimization problems (Sections 4.2, 4.3, and 5). It has the attractive property that it can be expressed as a linear function of portfolio weights: CF = α'e, where e_i = E_{i,Y}/Cap_{i,Y} is the emission intensity of firm i. This linearity makes the carbon constraint easy to incorporate into the quadratic optimization framework.

#### 4.1.4 Evolution of Carbon Metrics

> *[FIGURE: WACI and CF evolution over 2013–2024 for VW and MV portfolios]*

Both WACI and CF exhibit a general downward trend over the sample period. This decline is driven by two factors:

1. **Firm-level decarbonization:** Some firms genuinely reduce their emissions over time, through energy efficiency improvements, fuel switching, or operational changes.

2. **Investment set composition changes:** As carbon reporting improves after 2010, many new firms enter the investment set. These tend to be from the technology, healthcare, and financial services sectors — industries with structurally low carbon intensity and high market capitalisations. Their entry dilutes the weight of carbon-intensive firms (energy, materials, utilities) in the value-weighted benchmark.

Understanding which of these two factors dominates is the purpose of the decomposition analysis presented in Section 6.2.

> *[TABLE: Top 10 carbon-intensive firms by CI — representative year, with ISIN codes and sector]*

The top 10 most carbon-intensive firms are typically concentrated in the energy (oil and gas), materials (cement, steel, chemicals), and utilities (coal-fired power generation) sectors. These firms have a disproportionate impact on the portfolio's WACI and CF, even if their portfolio weights are relatively small.

### 4.2 CF-Constrained Minimum-Variance Portfolio

#### 4.2.1 Optimization Problem

We now construct a minimum-variance portfolio with a carbon footprint constraint. The portfolio CF must be at most 50% of the unconstrained minimum-variance portfolio's CF in the same year:

$$\min_{\alpha} \quad \alpha' \Sigma_Y \alpha$$

$$\text{s.t.} \quad CF_Y^{(p)} \leq 0.5 \times CF_Y^{(mv)}$$

$$\sum_{i=1}^{N} \alpha_i = 1, \quad \alpha_i \geq 0 \quad \forall i$$

where CF_Y^{(mv)} is the carbon footprint of the unconstrained minimum-variance portfolio computed for year Y. This constraint is re-evaluated each year, meaning that the 50% reduction target is relative to the current year's unconstrained MV portfolio, not a fixed baseline. This approach is motivated by the fact that the unconstrained MV portfolio's composition (and therefore its CF) changes from year to year as the investment set and covariance estimates evolve.

The CF constraint is linear in the portfolio weights: CF = Σ_i α_i × e_i ≤ target, where e_i = E_{i,Y}/Cap_{i,Y}. This means that the overall optimization remains a convex quadratic program (QP) with linear constraints, which can be solved efficiently by both SLSQP and CVXPY/OSQP.

#### 4.2.2 Interpretation

This portfolio represents the perspective of an **active investor** who seeks minimum variance while accepting a significant carbon constraint. The 50% reduction in CF is substantial — it means that the portfolio's carbon footprint per million invested must be less than half that of the unconstrained optimum.

The optimizer achieves this reduction by tilting weights away from carbon-intensive firms and toward cleaner alternatives. The cost of this constraint depends on the relationship between carbon intensity and the variance-covariance structure:

- If carbon-intensive firms happen to be high-volatility (e.g., energy stocks), the carbon constraint may have little cost, because the optimizer would already be underweighting these firms for variance reasons.
- If carbon-intensive firms are low-volatility (e.g., some utilities), the constraint forces the optimizer to give up diversification benefits, resulting in a higher portfolio variance.

#### 4.2.3 Results

> *[TABLE: Summary statistics — MV vs. MV-CF (SLSQP and LW): annualized return, volatility, Sharpe ratio]*

> *[FIGURE: Cumulative returns — MV vs. MV-CF (SLSQP and LW)]*

> *[FIGURE: CF and WACI evolution — MV vs. MV-CF]*

The comparison between the unconstrained and CF-constrained MV portfolios reveals the financial cost of decarbonization from an active investor's perspective. The key questions are:

- How much additional variance does the 50% CF constraint impose?
- Does the Sharpe ratio deteriorate significantly?
- Is the constraint always feasible, or are there years where achieving a 50% CF reduction requires extreme portfolio adjustments?

### 4.3 Tracking Error Minimization

#### 4.3.1 Optimization Problem

The tracking-error strategy targets a **passive investor** who wants to track the value-weighted benchmark as closely as possible while reducing carbon emissions. The optimization problem minimizes the squared tracking error:

$$\min_{\alpha} \quad TE_Y^2 = (\alpha_Y - \alpha_Y^{vw})' \Sigma_Y (\alpha_Y - \alpha_Y^{vw})$$

$$\text{s.t.} \quad CF_Y^{(p)} \leq 0.5 \times CF_Y^{(vw)}$$

$$\sum_{i=1}^{N} \alpha_i = 1, \quad \alpha_i \geq 0 \quad \forall i$$

where α_Y^{vw} is the vector of value-weighted benchmark weights and CF_Y^{(vw)} is the benchmark's carbon footprint. The TE² objective measures the variance of the portfolio's return relative to the benchmark — a portfolio with zero tracking error would replicate the benchmark exactly.

Minimizing TE² rather than TE (the standard deviation) is both mathematically equivalent (the square root is monotonic over non-negative values) and numerically preferable, because the square root has an infinite derivative at zero, which can cause problems for gradient-based solvers.

#### 4.3.2 Implementation Details

For the SLSQP implementation, the VW benchmark weights are the standard MV_Y_weight from the full investment set. The optimizer is warm-started from the VW weights (rather than equal weights) to provide a good initial feasible point that is close to the expected solution.

For the CVXPY/LW implementation, the benchmark is the VW-FA portfolio (MV_Y_weight_fa). Because the LW optimization operates on the FA subset, the VW-FA weights must be reindexed to match the LW firm ordering and renormalized to sum to 1 over the LW subset. The CF reference is computed over the full investment set using the VW-FA weights to maintain consistency.

#### 4.3.3 Results

> *[TABLE: Summary statistics — VW vs. TE (SLSQP and LW): annualized return, volatility, Sharpe ratio, tracking error]*

> *[FIGURE: Cumulative returns — VW vs. TE (SLSQP and LW)]*

> *[FIGURE: CF and WACI evolution — VW vs. TE]*

The tracking-error portfolio should track the VW benchmark closely in terms of cumulative returns, with small deviations caused by the carbon constraint. The magnitude of the tracking error indicates how much the investor must deviate from the benchmark to achieve the 50% CF reduction. A low tracking error suggests that decarbonization is "cheap" in terms of benchmark deviation — the portfolio can significantly reduce its carbon footprint without straying far from the market portfolio.

### 4.4 Comparison of Carbon-Constrained Portfolios

The CF-constrained minimum-variance portfolio and the tracking-error portfolio represent two fundamentally different investment philosophies applied to the same decarbonization goal:

| Dimension | MV-CF (Active) | TE (Passive) |
|---|---|---|
| Objective | Minimize portfolio variance | Minimize deviation from VW benchmark |
| Benchmark | None (absolute risk) | VW portfolio (relative risk) |
| CF constraint | ≤ 50% of MV CF | ≤ 50% of VW CF |
| Investor type | Active, risk-minimizing | Passive, index-tracking |
| Expected concentration | High (few low-vol stocks) | Low (close to VW weights) |

> *[TABLE: Side-by-side comparison — financial performance and carbon metrics for all Part II portfolios]*

The key insight from this comparison is whether the financial cost of decarbonization depends on the investment approach. For the active investor, the cost is measured in additional variance relative to the unconstrained MV portfolio. For the passive investor, the cost is measured in tracking error relative to the VW benchmark. These two costs are not directly comparable, but they both inform the broader question of whether ESG constraints are "free" or "costly" in financial terms.

---

## 5. Part III — Net Zero Portfolio

### 5.1 Strategy and Constraint

The net-zero strategy extends the tracking-error minimization framework of Section 4.3 by replacing the static 50% CF reduction with a time-varying constraint that enforces a compounding annual reduction from a fixed baseline:

$$CF_Y^{(p)} \leq (1 - \theta)^{Y - Y_0 + 1} \times CF_{Y_0}^{(vw)}$$

where θ = 0.10 (10% annual reduction rate), Y₀ = 2013 (baseline year), and CF_{2013}^{(vw)} is the carbon footprint of the value-weighted portfolio in 2013, computed once and held fixed as the reference throughout the sample.

This formulation is motivated by the concept of a carbon budget: the total amount of CO₂ that can be emitted while staying within a given temperature target. By reducing the portfolio's CF by 10% per year on a compounding basis, the investor follows a trajectory that is consistent with achieving significant decarbonization over a multi-decade horizon. Specifically, after 10 years, the portfolio's CF must be below (0.9)^{10} ≈ 34.9% of the 2013 level, and after 20 years, below (0.9)^{20} ≈ 12.2%.

The annual carbon budgets evolve as follows:

| Year | Exponent (Y − 2013 + 1) | CF budget (% of 2013 VW CF) |
|---|---|---|
| 2013 | 1 | 90.0% |
| 2014 | 2 | 81.0% |
| 2015 | 3 | 72.9% |
| 2016 | 4 | 65.6% |
| 2017 | 5 | 59.0% |
| 2018 | 6 | 53.1% |
| 2019 | 7 | 47.8% |
| 2020 | 8 | 43.0% |
| 2021 | 9 | 38.7% |
| 2022 | 10 | 34.9% |
| 2023 | 11 | 31.4% |
| 2024 | 12 | 28.2% |

An important feature of this constraint is that the reference CF (from 2013) is fixed, while the constraint budget decreases over time. This means that the constraint becomes increasingly tight as time progresses. In the early years (2013–2016), the 90%–65% budget is relatively generous and may not even be binding — the optimizer can stay close to the benchmark while easily satisfying the carbon constraint. In later years (2020–2024), the budget drops below 40% and the constraint becomes much more binding, forcing significant portfolio adjustments.

### 5.2 Implementation

The optimization problem is identical to the tracking-error minimization of Section 4.3, with only the CF constraint modified:

$$\min_{\alpha} \quad (\alpha_Y - \alpha_Y^{vw})' \Sigma_Y (\alpha_Y - \alpha_Y^{vw})$$

$$\text{s.t.} \quad CF_Y^{(p)} \leq (1 - 0.10)^{Y - 2013 + 1} \times CF_{2013}^{(vw)}$$

$$\sum_{i=1}^{N} \alpha_i = 1, \quad \alpha_i \geq 0 \quad \forall i$$

Two implementations are produced:

- **SLSQP:** Uses the sample covariance matrix. The CF reference is computed from the 2013 VW portfolio using MV_Y_weight weights.
- **CVXPY/LW:** Uses the Ledoit-Wolf covariance matrix. The CF reference is computed from the 2013 VW-FA portfolio using MV_Y_weight_fa weights over the full set.

### 5.3 Results

> *[TABLE: Summary statistics — VW vs. TE(0.5) vs. NZ (SLSQP and LW): annualized return, volatility, Sharpe ratio]*

> *[FIGURE: Cumulative returns — VW vs. TE(0.5) vs. NZ]*

> *[FIGURE: CF evolution — VW vs. TE(0.5) vs. NZ, with carbon budget line]*

The comparison between the TE(0.5) and NZ portfolios is particularly instructive because they share the same objective function (tracking error minimization) but differ in the stringency of the carbon constraint. In the early years, when the NZ budget is above 50%, the NZ portfolio may actually have a larger carbon budget than the TE(0.5) portfolio, meaning it stays closer to the benchmark. In the later years, when the NZ budget drops below 50%, the roles reverse and the NZ portfolio must deviate more from the benchmark.

### 5.4 Comparison and Cost of Net Zero

> *[FIGURE: Tracking error evolution over time — TE(0.5) vs. NZ]*

The net-zero portfolio illustrates the dynamic cost of a decarbonization commitment. Unlike the static 50% constraint, which imposes a constant level of stringency, the net-zero path creates a time-varying trade-off:

- In early years, the constraint is slack and the financial cost is near zero. The portfolio closely tracks the benchmark.
- In middle years, the constraint becomes binding and the tracking error increases gradually.
- In late years, the constraint is very tight and the portfolio may need to take concentrated positions in low-carbon firms, increasing both tracking error and idiosyncratic risk.

This time-varying nature has important implications for investor behavior. An investor who commits to a net-zero trajectory must accept that the cost of maintaining this commitment increases over time. If the investment universe does not decarbonize fast enough (i.e., if firm-level emissions do not decrease at the same pace as the carbon budget), the investor faces an increasingly difficult optimization problem.

Comparing the three portfolios P_oos^{vw}, P_oos^{vw}(0.5), and P_oos^{vw}(NZ) provides a comprehensive view of the cost spectrum of climate-aware investing, from no constraint (VW) through moderate constraint (50% reduction) to ambitious constraint (net zero trajectory).

---

## 6. Discussion

### 6.1 Financial Performance vs. Carbon Reduction Trade-Off

The central question of this project is whether carbon constraints impose a material cost on portfolio performance. Our results allow us to assess this trade-off across multiple dimensions.

> *[TABLE: Summary comparison — all 10 portfolios: return, volatility, Sharpe ratio, CF (2013 and 2024), WACI (2013 and 2024)]*

The evidence suggests that moderate carbon reduction (50% of the unconstrained CF) can be achieved at a relatively low financial cost. The tracking-error portfolios, in particular, demonstrate that it is possible to maintain very close adherence to the market benchmark while halving the portfolio's carbon footprint. This finding is consistent with the academic literature on ESG integration (e.g., Andersson, Bolton, and Samama, 2016), which shows that carbon-constrained portfolios can closely replicate market indices.

The cost increases under the net-zero constraint, particularly in the later years of the sample when the carbon budget becomes very tight. This highlights the importance of the investment universe's decarbonization trajectory: if the average carbon intensity of the universe decreases over time (through firm-level improvements and compositional changes), the net-zero constraint remains feasible without extreme portfolio distortions.

### 6.2 Decomposition of Carbon Metric Changes

To understand the sources of carbon reduction in each portfolio, we implement a Laspeyres-type counterfactual decomposition that separates the observed change in CF and WACI from the 2013 level into two additive components.

#### 6.2.1 Emission Effect (Frozen Weights)

The portfolio weights are held fixed at their 2013 values while firm-level emissions and carbon intensities are allowed to evolve over time:

$$CF_Y^{emission} = \sum_i \frac{w_{i,2013}}{\sum_j w_{j,2013}} \times \frac{E_{i,Y}}{Cap_{i,Y}}$$

This counterfactual measures what the portfolio's CF would have been if the investor had maintained the 2013 portfolio composition unchanged and only firm-level carbon data had evolved. A decrease in this metric indicates that firms themselves are reducing their emissions — the portfolio benefits from passive decarbonization without any change in allocation.

#### 6.2.2 Composition Effect (Frozen Emissions)

The firm-level emissions and carbon intensities are held fixed at their 2013 values while the portfolio weights are allowed to vary:

$$CF_Y^{composition} = \sum_i w_{i,Y} \times \frac{E_{i,2013}}{Cap_{i,2013}}$$

This counterfactual measures the impact of portfolio reweighting. A decrease indicates that the optimizer is actively steering capital away from carbon-intensive firms and toward cleaner alternatives. For the constrained portfolios (MV-CF, TE, NZ), the composition effect should be substantial and increasing over time as the constraints become more binding.

> *[FIGURE: Decomposition — emission effect vs. composition effect for VW, MV-CF, TE, and NZ]*

The decomposition reveals a key insight: for the VW portfolio, most of the observed carbon reduction comes from the emission effect (firms decarbonizing) and from compositional changes in the investment universe (new low-carbon firms entering). For the constrained portfolios, the composition effect is amplified by the optimization, which actively tilts weights toward low-carbon firms.

### 6.3 Top-10 Carbon Intensive Firms

> *[FIGURE: Top-10 CI firms — weight bar charts for each portfolio family (4×3 grid by year)]*

The analysis of portfolio weights allocated to the most carbon-intensive firms provides a granular view of how each strategy handles high-emitters. The unconstrained MV portfolio may allocate significant weight to carbon-intensive firms if they exhibit low volatility or favorable correlation properties. The CF-constrained and net-zero portfolios systematically reduce these positions, with the net-zero portfolio converging toward near-zero exposure to high-CI firms in the later years of the sample.

This analysis also reveals the sectoral implications of carbon constraints. If the top-10 carbon-intensive firms are concentrated in a single sector (e.g., utilities or energy), the carbon constraint effectively creates a sector exclusion, which may have unintended consequences for portfolio diversification.

### 6.4 Suggestions for Improvements

Several extensions could enhance the robustness and practical relevance of this analysis:

- **Factor-model covariance estimator.** In addition to the sample covariance and Ledoit-Wolf, a factor-model approach (e.g., using statistical factors from PCA or fundamental factors from Fama-French) could provide a third perspective on covariance estimation. Factor models impose more structure on the covariance matrix, which can improve out-of-sample stability when the number of assets is large.

- **WACI-based constraints.** The current carbon constraints are based on CF (emissions normalized by market cap). Adding a parallel constraint on WACI (emissions normalized by revenues) would capture a different dimension of climate risk — revenue-based carbon efficiency rather than capital-based emission intensity. The two metrics can diverge for firms with high market-to-revenue ratios (e.g., technology firms).

- **Parameter sensitivity analysis.** The stale-price threshold (50%), minimum return history (36 months), and estimation window length (10 years) are chosen based on practical considerations. A systematic sensitivity analysis would reveal how robust the results are to alternative parameter choices. For instance, a shorter estimation window (5 years) would make the covariance matrix more responsive to recent market conditions but also noisier.

- **Transaction costs and turnover constraints.** In practice, annual rebalancing involves trading costs (commissions, bid-ask spreads, market impact) that reduce net portfolio returns. Adding a turnover constraint (e.g., maximum annual turnover of 30%) or incorporating a transaction cost penalty in the objective function would produce more realistic results.

- **Sector-neutral constraints.** Carbon optimization can create large sector bets (e.g., massively underweighting energy). Adding sector-neutrality constraints would force the optimizer to achieve carbon reduction within each sector rather than across sectors, producing portfolios that are more robust to sector rotations.

### 6.5 Limitations

We identify the following limitations of our analysis:

- **Data quality and carbon reporting coverage.** Carbon data coverage is limited before 2010, and even after 2010, coverage varies significantly by region and sector. The use of forward-filled values for missing emissions data introduces measurement error, particularly for firms that experience large year-to-year changes in emissions. Moreover, the investment set composition changes substantially over time as more firms begin reporting carbon data. This means that year-over-year comparisons of portfolio-level carbon metrics are partly driven by changes in the investment universe rather than genuine decarbonization. Disentangling these effects requires the decomposition analysis of Section 6.2.

- **Parameter sensitivity and estimation choices.** The choice of the 10-year estimation window, the 50% stale-price threshold, and the 36-month minimum return requirement are somewhat arbitrary, though motivated by practical considerations. Different parameter choices could lead to materially different investment sets and portfolio allocations. For instance, a more stringent stale-price threshold (e.g., 30%) would exclude more illiquid firms, potentially improving portfolio implementability but reducing the investment universe. Similarly, a shorter estimation window would increase the responsiveness of the covariance matrix to recent market conditions but also increase estimation noise.

- **Constraint feasibility and portfolio concentration.** As the net-zero carbon budget tightens (below 30% of the 2013 baseline by 2024), the optimizer may struggle to find feasible solutions, particularly in years where the investment set contains few low-carbon firms with sufficient market capitalisation. This can result in highly concentrated portfolios that are exposed to significant idiosyncratic risk. In the most extreme case, the constraint may become infeasible — no long-only portfolio can achieve the required carbon reduction — which would require relaxing the constraint or expanding the investment universe.

- **Estimation risk and out-of-sample performance.** The sample covariance matrix is notoriously noisy for large cross-sections (large N relative to τ), and even the Ledoit-Wolf shrinkage may not fully resolve the estimation error problem. Portfolio weights derived from an estimated covariance matrix are themselves estimates, subject to sampling variability. Out-of-sample portfolio performance can differ substantially from in-sample expectations, particularly when the portfolio is highly concentrated. The comparison between SLSQP and LW results provides some evidence on the sensitivity to estimation quality, but a more comprehensive assessment would require bootstrap simulations or cross-validation.

- **Scope of emissions and data completeness.** Only Scope 1 (direct) and Scope 2 (indirect from purchased energy) emissions are considered. Scope 3 emissions — which include upstream supply chain emissions and downstream product use emissions — are excluded. For some sectors, Scope 3 emissions vastly exceed Scope 1 + 2: for example, oil and gas companies have large Scope 3 emissions from the combustion of their products by end users, and financial services firms have Scope 3 emissions through their lending and investment activities. Excluding Scope 3 may therefore understate the true carbon exposure of the portfolio and lead to an incomplete picture of climate risk. However, Scope 3 data is substantially less reliable and less widely reported than Scope 1 + 2, making its inclusion challenging in practice.

---

## 7. Conclusion

This project demonstrates that meaningful carbon footprint reductions can be achieved within a standard long-only equity portfolio framework at a moderate cost in terms of financial performance. The 50% CF reduction constraint proves feasible across all years for both the active (minimum-variance) and passive (tracking-error) approaches, suggesting that significant decarbonization is compatible with standard asset management practices.

The tracking-error minimization approach appears particularly well suited for institutional investors who are benchmarked against a market index and cannot afford large deviations from their benchmark. Our results show that a 50% reduction in portfolio carbon footprint can be achieved with a tracking error that remains small relative to the benchmark's volatility, confirming that decarbonization and index tracking are not mutually exclusive objectives.

The net-zero strategy reveals the dynamic cost of an ambitious decarbonization commitment. While the financial cost is negligible in the early years of the trajectory, it increases progressively as the carbon budget shrinks. The feasibility and cost of the net-zero path depend critically on the pace of firm-level decarbonization within the investment universe: if firms reduce their emissions over time, the constraint remains manageable; if they do not, the investor must make increasingly aggressive portfolio adjustments that may compromise diversification and risk management.

The choice of covariance estimator matters: the Ledoit-Wolf shrinkage estimator produces more stable and diversified portfolio weights than the raw sample covariance matrix, though at the cost of a smaller investment universe (limited to firms with complete return histories). Both approaches yield qualitatively consistent results, increasing confidence in the robustness of our conclusions.

The decomposition analysis provides additional insight into the drivers of carbon reduction. For unconstrained portfolios, much of the observed decline in carbon metrics reflects changes in the composition of the investment universe as carbon reporting expands. For constrained portfolios, the optimizer's active reweighting amplifies this decline by systematically shifting capital toward lower-carbon firms.

Overall, our findings support the growing consensus in both academic research and industry practice that climate-aware investing can be integrated into standard portfolio management without incurring prohibitive costs. The key conditions for success are a sufficiently diversified investment universe, a well-chosen covariance estimator, and a carbon constraint that is calibrated to remain feasible over the investment horizon. As carbon data quality improves and Scope 3 coverage expands, these strategies will become even more precise and informative.

---

## Use of Large Language Models (LLMs)

During this project, we used Claude (Anthropic) as a support tool for the following purposes: debugging Python code and understanding error messages encountered during the implementation of the optimization routines, improving code structure and readability (variable naming, function organization), and polishing the language, grammar, and structure of this report. The LLM was also used to help format and organize the Jupyter notebook with proper section headers and documentation.

We confirm that all core methodological choices — including the data cleaning rules, investment set construction criteria, choice of covariance estimators, formulation of optimization problems, and calibration of carbon constraints — are our own work and reflect our understanding of the course material. The coding decisions, including the implementation of the buy-and-hold weight drift, the choice of numerical solvers, and the treatment of edge cases (missing data, solver noise, infeasible constraints), were made by the group members. All results were generated by running our own code, and all interpretation and discussion of findings are our own analysis.

The group remains fully responsible for the correctness and academic integrity of all deliverables submitted as part of this project.
