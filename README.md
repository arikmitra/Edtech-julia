# Instructor Ranking and Classification System: Advanced Data Engineering & Generative Augmentation Pipeline

[![Julia Version](https://img.shields.io/badge/Julia-1.12+-9558B2?style=for-the-badge&logo=julia)](https://julialang.org/)
[![Framework](https://img.shields.io/badge/Flux.jl-Deep_Learning-green?style=for-the-badge)](https://fluxml.ai/)
[![Data Tool](https://img.shields.io/badge/DataFrames.jl-Data_Engineering-blue?style=for-the-badge)](https://dataframes.juliadata.org/stable/)
[![CSV Tool](https://img.shields.io/badge/CSV.jl-Data_I%2FO-lightgrey?style=for-the-badge&logo=julia)](https://csv.juliadata.org/stable/)
[![ML Framework](https://img.shields.io/badge/MLJ.jl-Machine_Learning-orange?style=for-the-badge&logo=julia)](https://alan-turing-institute.github.io/MLJ.jl/dev/)
[![Plotting Engine](https://img.shields.io/badge/Plots.jl-Visualization-yellowgreen?style=for-the-badge&logo=julia)](https://docs.juliaplots.org/stable/)
[![Heavy Rendering](https://img.shields.io/badge/CairoMakie.jl-Graphics-red?style=for-the-badge&logo=julia)](https://makie.juliaplots.org/stable/)
[![Statistical Core](https://img.shields.io/badge/StatsPlots.jl-Statistical_Plots-blueviolet?style=for-the-badge&logo=julia)](https://github.com/JuliaPlots/StatsPlots.jl)
[![Profiling Asset](https://img.shields.io/badge/BenchmarkTools.jl-Profiling-9558B2?style=for-the-badge&logo=julia)](https://github.com/JuliaCI/BenchmarkTools.jl)

## 1. Project Motivation & Pipeline Goals
In production machine learning pipelines, training deep classifiers or ranking architectures on small datasets often causes overfitting, poor generalization, and mode collapse. This project establishes an end-to-end data engineering and mathematical validation harness in Julia to ingest a baseline telemetry log of **2,000 rows** and systematically scale it up to a high-fidelity dataset of **50,000 records**.

Instead of applying uniform oversampling or unconstrained randomization—which damages the underlying multi-variable joint dependencies—this framework runs an 8-way benchmark suite across statistical, spatial, and deep generative models. The synthetic continuous features are then rigorously validated and mapped back into a structural, relational schema (`batch_id`, `instructor_id`, `course_id`) that accurately mimics the scaling realities of a massive digital education platform.

---

## 2. Repository Architecture & Codebase Layout
### Module Breakdown
* **`setup.jl`**: Configures and provisions a localized project environment using Julia’s built-in package manager, guaranteeing complete runtime reproducibility.
* **`Instructor_analysis.jl`**: Performs initial exploratory data analysis (EDA), maps continuous features, constructs a deterministic performance indicator, and partitions observations into low/medium/high tiers via quantile distributions.
* **`evaluation.jl`**: A decoupled validation suite that ingests synthetic and original matrices to calculate distribution similarity across variance, correlation, and topography boundaries.
* **`Data_augmenting.jl`**: The execution engine that transforms the baseline data, coordinates the 8 generative models, tracks performance metrics, chooses the optimal candidate matrix, and structures the final 50,000-row platform schema.

---

## 3. Data Processing & Analytical Foundations (`Instructor_analysis.jl`)

The ground-truth dataset (`instructor_effectiveness_dataset_2000_rows.csv`) tracks 9 continuous telemetry metrics detailing course engagement and performance:
1. `completion_rate` 
2. `avg_score_improvement`
3. `avg_quiz_score`
4. `dropout_rate`
5. `avg_watch_time`
6. `assignment_submission_rate`
7. `forum_activity_rate`
8. `avg_feedback_score`
9. `feedback_response_rate`

### Feature Engineering & Tiering
Before running any data synthesis, `Instructor_analysis.jl` establishes a definitive performance benchmark. It processes these features to build a comprehensive, multi-factorial metric:
* Inverse weights are applied to negative indicators (such as `dropout_rate`).
* The values are blended into a unified, deterministic **Instructor Effectiveness Score**.
* It divides the instructor pool into **Low**, **Medium**, and **High** proficiency tiers using explicit quantile distribution breaks.
* High-volume visualizations (heatmaps, histograms, and violin plots) are compiled using `StatsPlots` and `CairoMakie` to inspect multi-modal density variations across courses.

---

## 4. The 8-Way Generative Augmentation Suite

Implemented natively in `Data_augmenting.jl`, the pipeline evaluates eight distinct mathematical approaches to synthesis:

### 1. Gaussian Jittering
Applies targeted additive Gaussian noise to the standardized data matrix. It acts as an anchor baseline by introducing localized variance without altering point-wise locations:
$$\tilde{\mathbf{x}} = \mathbf{x} + \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

### 2. Gaussian Copula
Separates the selection of marginal distributions from the joint dependency structure. It maps each continuous feature into standard uniform space via a probability integral transform, extracts the underlying linear correlation structure, samples from a multivariate normal distribution, and maps the vectors back to the original marginal spaces.

### 3. Kernel Density Estimation (KDE)
Bypasses restrictive parametric assumptions using Direct Multivariate Sampling. The original observations function as empirical cluster centers, and synthetic points are generated by adding random noise scaled by a selective smoothing bandwidth parameter ($h = 0.03$).

### 4. SMOTE (Synthetic Minority Over-sampling Technique)
Preserves local geometry through spatial feature interpolation. For each uniformly sampled observation, it maps its neighborhood using a spatial $k$-nearest neighbors tree ($k=5$) and extracts a random candidate point to draw a new synthetic vector:
$$\mathbf{x}_{new} = \mathbf{x}_{base} + \lambda (\mathbf{x}_{neighbor} - \mathbf{x}_{base}), \quad \lambda \sim \mathcal{U}(0,1)$$

### 5. ADASYN (Adaptive Synthetic Sampling)
Implements an advanced spatial approach by analyzing local neighborhood densities via a $k$-d tree. It assigns higher generation weights to points situated in sparser, highly isolated clusters. This shifts the synthesizer's focus toward regions that are structurally complex or harder for standard algorithms to resolve.

### 6. CTGAN (Conditional Tabular GAN)
An adversarial architecture implemented in `Flux.jl` tailored for tabular layouts using $k$-means mode-specific normalization to resolve multi-modal distributions. The Generator and Discriminator then optimize concurrently using conditional vectors to prevent mode collapse.

### 7. TVAE (Tabular Variational Autoencoder)
An autoencoders framework optimized in `Flux.jl`. An Encoder network maps the continuous input vectors into lower-dimensional latent distribution parameters (Mean $\mu$, Log-Variance $\log(\sigma^2)$). A Decoder network reconstructs the inputs from samples drawn using the reparameterization trick. The architecture balances reconstruction fidelity and latent spacing using a composite loss function:
$$\mathcal{L}_{TVAE} = \text{MSE}(\mathbf{x}, \hat{\mathbf{x}}) + \alpha \cdot D_{KL}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, \mathbf{I}))$$

### 8. TabDDPM (Tabular Denoising Diffusion Probabilistic Model)
A state-of-the-art parameterized generative framework. A **Forward Process** systematically destroys data structure by introducing linear Gaussian noise across a 20-step schedule. A time-conditioned Multi-Layer Perceptron (MLP) learns a **Reverse Process** to iteratively remove noise from a pure Gaussian starting state, outputting clean, realistic tabular patterns.

---

## 5. Multi-Dimensional Metric Evaluation Stack (`evaluation.jl`)

To verify that the synthetic samples match the properties of the original data, `evaluation.jl` runs every output matrix through three strict mathematical validation tests:

### 1. Shape Variance Discrepancy (`Std_Diff_MAE`)
Tracks the preservation of column-wise dispersion by measuring the Mean Absolute Error between the standard deviations of the synthetic and ground-truth features:
$$\text{MAE}_{\sigma} = \frac{1}{M}\sum_{j=1}^{M} |\sigma_{orig}^{(j)} - \sigma_{new}^{(j)}|$$

### 2. Structural Correlation Error (`Correlation_Error`)
Measures structural deformation across all 9 variables simultaneously. It evaluates the Frobenius Norm of the difference between the original and synthetic Pearson correlation matrices:
$$\text{Error}_{corr} = \| \mathbf{R}_{orig} - \mathbf{R}_{new} \|_F = \sqrt{\sum_{i=1}^{M}\sum_{j=1}^{M} (R_{orig, ij} - R_{new, ij})^2}$$

### 3. Distribution Topography Shift (`Avg_Wasserstein`)
Computes the average 1D Earth Mover's Distance (1st Wasserstein Distance) across all continuous variables. By integrating the absolute area difference between the Empirical Cumulative Distribution Functions (ECDFs), it checks that multi-modal curves and asymmetric boundaries match perfectly:
$$\mathcal{W}_1(P^{(j)}, Q^{(j)}) = \int_{-\infty}^{\infty} |F_{orig}^{(j)}(x) - F_{new}^{(j)}(x)| dx$$

---

## 6. Relational Schema Engineering & Post-Assembly

After validating the features, the pipeline transforms the continuous matrix into an educational schema. It applies real-world platform rules to generate structural ID columns that avoid artificial uniform distributions:

### Naming Conventions & Schema Structure
* **`batch_id`**: Prefixed as `B_XXXX` (e.g., `B_0001` to `B_50000`), serving as a sequential tracker.
* **`instructor_id`**: Prefixed as `I_XXX` up to the user target of **2,500 unique instructors**.
* **`course_id`**: Prefixed as `C_XX` up to the user target of **100 unique courses**.

### Behavioral Simulation Logic
1. **The Pareto Principle (80/20 Rule) for Instructors**: Instructors are sampled using a Zipf-like power decay distribution ($1 / x^{0.85}$). This accurately simulates a real-world platform where a small pool of active power-instructors author the majority of running course cohorts.
2. **Dense Mapping Portfolios**: To capture true domain-specific constraints, every `course_id` is assigned a restricted portfolio of 5 to 15 eligible instructors. Rows are generated within these portfolios to build realistic, dense cross-mappings and clear many-to-many relationships.

---

## 7. Execution Guide & Deployment Flow

### Step 1: Provision the Local Environment
Initialize the environment and download all required analytical and deep learning packages by running the setup utility from your terminal:
```bash
julia setup.jl

julia Instructor_analysis.jl

julia Data_augmenting.jl
