# 🐯 mindhunter

Extensions for DataFrames to make statistical and analysis operations much, *much* more comfortable and convenient. Turns your `DataFrame` into a `StatFrame`, composing Mindhunter's new features *over* it, supercharging its capabilities without sacrificing compatibility. 

---

## 📦 Installation

### 🗃️ From the repo:
You need `uv` to build the module.

- Clone the repository
- `chmod +x ./build.sh`
- `./build.sh`
  - It will clear cache, build, install and test the module.


## 🧪 Testing
Mindhunter implements a fairly rudimentary setup for testing. It will look inside `tests` for any fixtures or tests inside files starting with `test_`. It uses `pytest` and `faker` to create a randomised dataset to test upon.

So far, coverage goes to the extent of making sure a `StatFrame` can be created and data can be obtained. More testing is being developed and it's coming soon.


## 📝 Features

### 📋 Meet `StatFrame` and the crew

- Your new `StatFrame` can be used now with Mindhunter's new **Analyzers, Plotters and Toolkits:**
  - `DistributionAnalyzer`: adds normal distribution utilities directly on top of the `DataFrame`.
  - `HypothesisAnalyzer`: adds hypothesis testing, binomial and related functionality.
  - `AnalyticalTools`: provides access to `scipy.stats` methods to generate and convert several values over a given `StatFrame`.
  - `StatPlotter`: adds ready-to-go plotting capabilities for many common values, like z-scores, Coefficient of Variation, Normal Distribution, and others; using `seaborn` and `matplotlib.pyplot`.
  - `StatVisualizer`: provides easy access to build common graphs and visualizations, returning ready-to-go graphs just by passing lists or a `StatFrame`.

### 💾 Quick stats and cached values
- `StatFrame` also holds a cache of the most commonly-used values and variables, providing easy access to the values of not just a column, but of a whole set. It caches:
- **Central Tendency:**
  - mean
  - median
  - mode
- **Spread/Variability:**
  - std (standard deviation)
  - variance
  - range
  - iqr (inter-quantile range)
  - mad (median absolute deviation)
- **Distribution Shape:**
  - skewness
  - kurtosis
- **Data Quality:**
  - count
  - missing_count
  - missing_pct
- **Extreme Values:**
  - min
  - max
  - q1
  - q3
- **Key Ratios:**
  - cv (coefficient of variation)
  - sem (standard error of mean)

### 🧹 Auto-cleanup:
- Mindhunter can also **automatically cleans column names, drops NaN and duplicates** of datasets. It also provides methods to **locate, analyze and remove zero-values** from your dataset.

---

## ℹ️ But, why?

I've been studying data analysis and, over the months, I've been collecting a bunch of little methods and scripts to do my homework. It then went to the point it was a 800+ line cell on each Jupyter Notebook. It became a *bit* too much. 

### 🏗️ How does it work on the inside:

In short: it uses basic OOP **composition**, against all advise, to pass the `StatFrame` as an argument. That class holds the `DataFrame` itself, and all operations are done through the `StatFrame` directly to the DF. All operations act directly on the source, and calling `update()` will re-trigger the caching process.

### 🔮 So, what's the future?

This library will be updated fairly regularly, as I start collecting and tidying up more and more little tools, and taking more advantage of the internal mechanisms. I am *much* more of a developer than a data analyst, so I need much more help knowing what the community *needs* for me to keep on improving the library. If you have any issue, suggestion or comment, feel free to create a new issue!