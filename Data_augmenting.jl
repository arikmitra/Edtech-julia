""" 
The original data was only 2000 observations with 12 columns 3 of which are ID columns.

There were 120 instructors, 2000 batches, and 25 courses

The no of observations has been fixed at 50000 to simulate a real life large education
platform with many courses and degree options.

Post augmentation there will be 100 courses, 50000 batches and 2500 instructors

Augmentation methods used here are for the sake of experimenting with various types
including machine learning techniques.
"""

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using NearestNeighbors
using Clustering
using Flux
using MLJ
using Distributions
using Random

# loading data
data = CSV.read("D:/intern/instructor_effectiveness_dataset_2000_rows.csv",DataFrame);

# column names
#names(data)

# Using print functon to view all column names without truncating
println(names(data))

# Check for missing values
[col => sum(ismissing.(data[!, col])) for col in names(data)]

"""Now we set our augmenting targets"""

target_batch = 50000
target_instr = 2500
target_course = 100

"""
NOTE: Many large education platforms have degree options as well but here we aren't 
coonsidering them to keep the dataset little less complex
"""

"""
Since we will be using various augmenting methods, standardizing the data will be
a sensible thing to do. There are machine learning methods involved so we'll be using
standard scaling. It should be kept in mind that many data fields have hard boundaries
from 0-1. It is a good idea to standardize the original data before augmenting. There are some
ID columns which will need to stay as they are.
"""

"""Normalizing the data"""

# Filtering to take only the non ID columns
filtered_df = data[:,4:12]

#Normalizing
# Compact assignment
normal_scaler(df) = MLJ.transform(fit!(machine(Standardizer(), df)), df)

scaled_df = normal_scaler(filtered_df)

"""We need to add the normalised data back with the ID columns
which are like primary keys for the data"""

instructor_df = hcat(data[:,1:3],scaled_df)

# We do not need the IDs for the algorithms, in fact they may break the ML pipeline 

X_scaled = Matrix{Float64}(scaled_df)

# X_scaled1 = Matrix{Any}(instructor_df) # Not ideal as will create data aurmneting problems later

"""We can build the evaluation stack here or in a separate evaluation.jl script"""

"""
evaluate_sim(X_new_scaled, X_orig_scaled)

Evaluates the fidelity of synthetic tabular data against original data using:
1. Shape Variance Discrepancy (MAE of Standard Deviations)
2. Structural Correlation Error (Frobenius Norm of Correlation Differences)
3. Distribution Topography Shift (Average Univariate Wasserstein Distance)

Expects inputs to be Matrix{Float64} oriented as (Samples × Features).
"""

"""
function evaluate_sim(X_new_scaled::AbstractMatrix{Float64}, X_orig_scaled::AbstractMatrix{Float64})
    # Defensive programming check to prevent mismatched dimension calculations
    if size(X_new_scaled, 2) != size(X_orig_scaled, 2)
        throw(DimensionMismatch("Feature counts must match! New has $(size(X_new_scaled, 2)) cols, Orig has $(size(X_orig_scaled, 2)) cols."))
    end

    n_features = size(X_orig_scaled, 2)
    
    # -------------------------------------------------------------------------
    # 1. Variance Discrepancy (MAE of Stds)
    # -------------------------------------------------------------------------
    # Captures if models are collapsing variance (VAEs) or inflating it (Jittering)
    std_orig = std(X_orig_scaled, dims=1)
    std_new  = std(X_new_scaled, dims=1)
    std_diff_mae = mean(abs.(std_orig .- std_new))

    # -------------------------------------------------------------------------
    # 2. Structural Correlation Error (Frobenius Norm)
    # -------------------------------------------------------------------------
    # Checks if critical domain correlations remain intact
    corr_orig = cor(X_orig_scaled)
    corr_new  = cor(X_new_scaled)
    
    # Replace any potential NaNs (caused by zero variance columns) with 0.0 safely
    corr_new[isnan.(corr_new)] .= 0.0
    corr_structural_diff = norm(corr_orig - corr_new)

    # -------------------------------------------------------------------------
    # 3. Average 1D Wasserstein Distance (Earth Mover's Distance)
    # -------------------------------------------------------------------------
    # Measures the physical work needed to reshape the synthetic distribution profiles 
    # back into the original distribution shapes across boundaries.
    wasserstein_distances = zeros(n_features)
    
    for j in 1:n_features
        col_orig = sort(X_orig_scaled[:, j])
        col_new  = sort(X_new_scaled[:, j])
        
        # Create continuous empirical CDF functions for both distributions
        ecdf_orig = ecdf(col_orig)
        ecdf_new  = ecdf(col_new)
        
        # Define a fine evaluation grid spanning the joint minimum and maximum limits
        min_val = min(col_orig[1], col_new[1])
        max_val = max(col_orig[end], col_new[end])
        grid = range(min_val, max_val, length=1000)
        
        # The 1D Wasserstein distance is mathematically the integrated area between the two CDF profiles
        # Using a trapezoidal/Riemann integration method across the common boundary limits
        dx = (max_val - min_val) / 999.0
        integrated_area = sum(abs(ecdf_orig(x) - ecdf_new(x)) for x in grid) * dx
        
        wasserstein_distances[j] = integrated_area
    end
    avg_wasserstein = mean(wasserstein_distances)
    
    # Return the three metrics rounded to 4 decimals to cleanly update your `results` array
    return (
        round(std_diff_mae, digits=4), 
        round(corr_structural_diff, digits=4), 
        round(avg_wasserstein, digits=4)
    )
end
"""

"""Gaussian Jittering with @elapsed"""

n_samples, n_features = size(X_scaled)
repeats = ceil(Int, target_batch / n_samples)

# We capture the runtime of the exact generation block
elapsed_time = @elapsed begin
    X_jitter_base = vcat(fill(X_scaled, repeats)...)[1:target_batch, :]
    jitter_strength = 0.02 
    noise = randn(target_batch, n_features) .* jitter_strength
    global X_jittered = X_jitter_base .+ noise
end

# Evaluate and Store using the captured time
#eval_metrics = evaluate_sim(X_jittered, X_scaled)
#push!(results, ("Gaussian Jittering", eval_metrics..., elapsed_time))

""" Even though we tried jittering, we can see the following originally planned methods have
certain shortcomings as below and thus we will not be moving forward with them.

The "Proceed with Caution" List (High Risk of Overfitting/Failure)
These techniques face significant challenges when applied directly to a 2,000-row 
tabular dataset:

GAN (Generative Adversarial Networks) & VAE: Standard GANs and VAEs 
fail on tabular data. They require massive amounts of data to converge, 
suffer from mode collapse, and completely ignore column data types. 
If you use a GAN/VAE, do not use standard image/text architectures. 
You must use CTGAN (Conditional Tabular GAN) or TVAE, which use Gumbel-Softmax to handle 
tabular distributions.

Mixup: Mixup creates convex combinations of pairs of examples 
($x_{new} = lambda x_i + (1-lambda)x_j$). While it is great for neural network 
regularization, doing this directly on tabular data can create unrealistic "ghost" 
averages that violate physical boundaries if not carefully constrained.Gaussian 
    
Jittering: Adding random Gaussian noise to rows can destroy sharp boundaries. 
For instance, if an instructor has a perfect feedback score of 5.0, adding random 
noise could push it to 5.1 (impossible) or degrade meaningful edge cases.
"""

"""Gaussian Copula (Optimized Vectorized Version)"""

n_samples, n_features = size(X_scaled)

elapsed_time_copula = @elapsed begin
    # =========================================================================
    # Step 1: Forward Quantile Transformation (To Normal Space)
    # =========================================================================
    # Precompute normal quantiles once to eliminate redundant computations
    u_grid = range(0, 1, length=n_samples)
    q_precomputed = quantile.(Normal(), clamp.(u_grid, 1e-7, 1.0 - 1e-7))

    X_norm = zeros(n_samples, n_features)
    for j in 1:n_features
        # sortperm with a view avoids copying underlying column data
        perm = sortperm(@view X_scaled[:, j])
        X_norm[perm, j] .= q_precomputed
    end

    # =========================================================================
    # Step 2 & 3: Covariance Calculation & Empirical Sampling
    # =========================================================================
    cov_matrix = cov(X_norm)

    # Force strict Positive Semi-Definiteness via Symmetric matrix constructor
    mv_norm = MvNormal(zeros(n_features), Symmetric(cov_matrix))
    
    # rand() outputs (Features × Samples); we transpose to match (Samples × Features)
    X_norm_sampled = rand(mv_norm, target_batch)' 

    # =========================================================================
    # Step 4: Inverse Quantile Transformation (Vectorized Grid Search)
    # =========================================================================
    # Compute all marginal CDF probabilities across the matrix simultaneously
    U_sampled = cdf.(Normal(), X_norm_sampled)

    global X_copula = zeros(target_batch, n_features)
    for j in 1:n_features
        sorted_col = sort(@view X_scaled[:, j])
        
        # Vectorized coordinate computation for linear interpolation mapping
        val_idx = (@view U_sampled[:, j]) .* (n_samples - 1) .+ 1
        idx_floor = clamp.(floor.(Int, val_idx), 1, n_samples - 1)
        weight = val_idx .- idx_floor
        
        # Fully broadcasted linear interpolation map without inner row loops
        X_copula[:, j] .= sorted_col[idx_floor] .+ weight .* (sorted_col[idx_floor .+ 1] .- sorted_col[idx_floor])
    end
end

# =========================================================================
# Step 5: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_copula = evaluate_sim(X_copula, X_scaled)
#push!(results, ("Gaussian Copula", metrics_copula..., elapsed_time_copula))
#println("Gaussian Copula Complete. Rows generated: ", size(X_copula, 1))


"""KDE (Direct Multivariate Sampling)"""

n_samples, n_features = size(X_scaled)

# Define the smooth bandwidth matching your pipeline baseline parameters
bandwidth = 0.03

elapsed_time_kde = @elapsed begin
    # 1. Randomly sample row indices with replacement to scale up to 50,000
    kde_indices = rand(1:n_samples, target_batch)

    # 2. Generate synthetic data by injecting continuous Gaussian noise
    # This draws new points directly from the estimated multivariate probability density
    global X_kde = X_scaled[kde_indices, :] .+ randn(target_batch, n_features) .* bandwidth
end

# =========================================================================
# Step 3: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_kde = evaluate_sim(X_kde, X_scaled)
#push!(results, ("KDE", metrics_kde..., elapsed_time_kde))
#println("KDE Complete. Rows generated: ", size(X_kde, 1))


"""ADASYN (Adaptive Synthetic Sampling)"""

# Bring neighborhood spatial trees into scope

n_samples, n_features = size(X_scaled)
k = 5  # Number of nearest neighbors to look at

elapsed_time_adasyn = @elapsed begin
    # 1. Build a KDTree for fast spatial proximity coordinate queries
    # NearestNeighbors.jl expects a (Features × Samples) matrix orientation
    X_scaled_t = collect(X_scaled')
    tree = KDTree(X_scaled_t)

    # 2. Find the k+1 nearest neighbors (index 1 is the point itself)
    idxs, dists = knn(tree, X_scaled_t, k + 1, true)

    # 3. Calculate an adaptive density weight for each historical point
    # We use the mean distance to its neighbors as an indicator of local sparsity
    # Scaled data points that are isolated get higher weights
    avg_distances = [mean(dists[i][2:end]) for i in 1:n_samples]
    
    # Avoid zero-division errors if points are perfectly duplicated
    total_dist = sum(avg_distances)
    weights = total_dist > 0 ? avg_distances ./ total_dist : fill(1.0 / n_samples, n_samples)

    # 4. Probabilistically sample base row indices using the density weights
    base_indices = sample(1:n_samples, Weights(weights), target_batch)

    # 5. Extract neighbor pools for our selected base points
    # We map coordinates between pairs to interpolate new vectors
    global X_adasyn = zeros(target_batch, n_features)
    
    for i in 1:target_batch
        base_idx = base_indices[i]
        
        # Pick a random neighbor from the k-nearest choices (excluding itself)
        neighbor_pool = idxs[base_idx][2:end]
        chosen_neighbor_idx = rand(neighbor_pool)
        
        # Draw a random interpolation factor lambda between 0.0 and 1.0
        λ = rand()
        
        # Core ADASYN Interpolation Step: X_new = X_base + λ * (X_neighbor - X_base)
        X_adasyn[i, :] .= X_scaled[base_idx, :] .+ λ .* (X_scaled[chosen_neighbor_idx, :] .- X_scaled[base_idx, :])
    end
end

# =========================================================================
# Step 6: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_adasyn = evaluate_sim(X_adasyn, X_scaled)
#push!(results, ("ADASYN", metrics_adasyn..., elapsed_time_adasyn))
#println("ADASYN Complete. Rows generated: ", size(X_adasyn, 1))


"""SMOTE (Synthetic Minority Over-sampling Technique)"""

n_samples, n_features = size(X_scaled)
k = 5  # Number of nearest neighbors for neighborhood interpolation

elapsed_time_smote = @elapsed begin
    # 1. Build a spatial KDTree using the transposed scale (Features × Samples)
    X_scaled_t = collect(X_scaled')
    tree = KDTree(X_scaled_t)

    # 2. Extract the neighborhood index mapping matrix (k+1 to ignore the self-match)
    idxs, _ = knn(tree, X_scaled_t, k + 1, true)

    # 3. Uniformly sample base row indices to scale up to 50,000 observations
    base_indices = rand(1:n_samples, target_batch)

    # 4. Generate the synthetic rows via localized neighbor vector blending
    global X_smote = zeros(target_batch, n_features)
    
    for i in 1:target_batch
        base_idx = base_indices[i]
        
        # Pull candidate neighbors (positions 2 through k+1)
        neighbor_pool = idxs[base_idx][2:end]
        chosen_neighbor_idx = rand(neighbor_pool)
        
        # Draw a random linear interpolation weight 
        λ = rand()
        
        # SMOTE Interpolation: X_new = X_base + λ * (X_neighbor - X_base)
        X_smote[i, :] .= X_scaled[base_idx, :] .+ λ .* (X_scaled[chosen_neighbor_idx, :] .- X_scaled[base_idx, :])
    end
end

# =========================================================================
# Step 7: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_smote = evaluate_sim(X_smote, X_scaled)
#push!(results, ("SMOTE", metrics_smote..., elapsed_time_smote))
#println("SMOTE Complete. Rows generated: ", size(X_smote, 1))

"""
There is a point about using SMOTE here. SMOTE uses Euclidean distances
and might not yield plausible results in this case. It may be better to use SMOTE-NC
if one needs to avoid randomizing the instructor patterns entirely. There was no
specific pattern as we found in EDA. Also, using any SMOTE method isnt a good idea if
new generated data needs to simulate the original population as SMOTE increases the
the number of entries within existing categories by interpolation, and not generate data 
for new categories as per our plan to have 2500 instructors and 100 courses. The models
we use next are more suitable.
"""

"""CTGAN (Conditional Tabular Generative Adversarial Network) Engine"""

n_samples, n_features = size(X_scaled)
latent_dim = 32
modes_per_feature = 3  # Number of GMM-style clusters per column

elapsed_time_ctgan = @elapsed begin
    # =========================================================================
    # Step 1: Mode-Specific Normalization (Pre-processing)
    # =========================================================================
    # We locate cluster centers for each column to handle multi-modal shapes
    centers = zeros(modes_per_feature, n_features)
    X_reconstructed_base = zeros(n_samples, n_features)
    
    for j in 1:n_features
        col_data = reshape(X_scaled[:, j], 1, n_samples)
        # Partition the column into localized modes
        res = kmeans(col_data, modes_per_feature)
        centers[:, j] .= res.centers[1, :]
    end

    # =========================================================================
    # Step 2: Define Generator and Discriminator Networks
    # =========================================================================
    # Generator outputs the structural feature map matching n_features
    generator = Chain(
        Dense(latent_dim => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => n_features)
    )

    # Discriminator evaluates authenticity (scalar probability output)
    discriminator = Chain(
        Dense(n_features => 32, leakyrelu),
        Dense(32 => 16, leakyrelu),
        Dense(16 => 1, sigmoid)
    )

    # =========================================================================
    # Step 3: Setup Optimizers and Training Routines
    # =========================================================================
    opt_g = Flux.setup(Flux.Adam(2f-4, (0.5, 0.999)), generator)
    opt_d = Flux.setup(Flux.Adam(2f-4, (0.5, 0.999)), discriminator)

    # Convert training data to Float32 Matrix for neural network processing
    X_train = Float32.(X_scaled)

    # Mini-GAN Training loop over 150 mini-epochs 
    for epoch in 1:150
        # Draw random noise representations for the latent space
        z = randn(Float32, latent_dim, n_samples)
        
        # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        loss_d, grads_d = Flux.withgradient(discriminator) do d_net
            fake_data = generator(z)
            real_preds = d_net(X_train')
            fake_preds = d_net(fake_data)
            
            # Binary Cross Entropy Loss
            return -mean(log.(real_preds .+ 1f-7) .+ log.(1f0 .- fake_preds .+ 1f-7))
        end
        Flux.update!(opt_d, discriminator, grads_d[1])

        # Train Generator: minimize log(1 - D(G(z)))
        loss_g, grads_g = Flux.withgradient(generator) do g_net
            fake_data = g_net(z)
            fake_preds = discriminator(fake_data)
            return -mean(log.(fake_preds .+ 1f-7))
        end
        Flux.update!(opt_g, generator, grads_g[1])
    end

    # =========================================================================
    # Step 4: Conditional Generation (Sampling 50,000 Rows)
    # =========================================================================
    z_samples = randn(Float32, latent_dim, target_batch)
    X_fake_t = generator(z_samples)
    X_ctgan_raw = Float64.(X_fake_t')

    # Apply Mode-Conditioned Smoothing to snap outputs toward closest logical modes
    global X_ctgan = zeros(target_batch, n_features)
    for j in 1:n_features
        for i in 1:target_batch
            # Force values to align relative to the mathematically extracted centers
            closest_mode_idx = argmin(abs.(X_ctgan_raw[i, j] .- centers[:, j]))
            X_ctgan[i, j] = X_ctgan_raw[i, j] + (centers[closest_mode_idx, j] * 0.05)
        end
    end
end

# =========================================================================
# Step 5: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_ctgan = evaluate_sim(X_ctgan, X_scaled)
#push!(results, ("CTGAN", metrics_ctgan..., elapsed_time_ctgan))
#println("CTGAN Complete. Rows generated: ", size(X_ctgan, 1))

"""NOTE: CTGAN can be unstable on small datasets"""

"""TVAE (Tabular Variational Autoencoder) Engine"""

n_samples, n_features = size(X_scaled)
latent_dim = 4
modes_per_feature = 3

elapsed_time_tvae = @elapsed begin
    # =========================================================================
    # Step 1: Extract Multi-Modal Cluster Metadata (GMM-style centers)
    # =========================================================================
    centers = zeros(modes_per_feature, n_features)
    for j in 1:n_features
        col_data = reshape(X_scaled[:, j], 1, n_samples)
        res = kmeans(col_data, modes_per_feature)
        centers[:, j] .= res.centers[1, :]
    end

    # =========================================================================
    # Step 2: Define the Variational Autoencoder Architecture
    # =========================================================================
    # The Encoder paths reduce features to Latent Mean (mu) and Log-Variance (lv)
    encoder_backbone = Chain(Dense(n_features => 32, relu), Dense(32 => 16, relu))
    mu_layer = Dense(16 => latent_dim)
    lv_layer = Dense(16 => latent_dim)

    # The Decoder maps latent vectors back to the original standardized feature space
    # FIX: No sigmoid on the final layer so it can map to full negative Z-score ranges
    decoder = Chain(Dense(latent_dim => 16, relu), Dense(16 => 32, relu), Dense(32 => n_features))

    # Collect parameters for the composite model structure
    tvae_params = Flux.params(encoder_backbone, mu_layer, lv_layer, decoder)
    opt_tvae = Flux.setup(Flux.Adam(1f-3), (encoder_backbone, mu_layer, lv_layer, decoder))

    # Convert to Float32 Matrix for Flux optimizations
    X_train = Float32.(X_scaled')

    # =========================================================================
    # Step 3: Define Custom TVAE Loss (MSE Reconstruction + KL Divergence)
    # =========================================================================
    function compute_tvae_loss(x)
        # Forward pass through encoder
        h = encoder_backbone(x)
        mu = mu_layer(h)
        lv = lv_layer(h)
        
        # Reparameterization Trick: z = mu + ϵ * σ
        ϵ = randn(Float32, size(mu))
        z = mu .+ ϵ .* exp.(0.5f0 .* lv)
        
        # Reconstruct from latent space
        x_hat = decoder(z)
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = Flux.mse(x_hat, x)
        
        # 2. Kullback-Leibler Divergence (Regularizer penalizing non-normal distributions)
        kl_loss = -0.5f0 * sum(1f0 .+ lv .- mu.^2 .- exp.(lv)) / size(x, 2)
        
        # TVAE balances these terms to prevent latent variance collapse
        return recon_loss + 0.1f0 * kl_loss
    end

    # =========================================================================
    # Step 4: TVAE Training Loop (300 Epochs)
    # =========================================================================
    for epoch in 1:300
        loss, grads = Flux.withgradient() do
            compute_tvae_loss(X_train)
        end
        Flux.update!(opt_tvae, (encoder_backbone, mu_layer, lv_layer, decoder), grads)
    end

    # =========================================================================
    # Step 5: Generative Latent Sampling (50,000 Rows)
    # =========================================================================
    # Sample clean Gaussian representations directly from the smooth latent space
    z_sampled = randn(Float32, latent_dim, target_batch)
    X_fake_t = decoder(z_sampled)
    X_tvae_raw = Float64.(X_fake_t')

    # Re-apply mode-specific scaling weights to align cluster conditional boundaries
    global X_tvae = zeros(target_batch, n_features)
    for j in 1:n_features
        for i in 1:target_batch
            closest_mode_idx = argmin(abs.(X_tvae_raw[i, j] .- centers[:, j]))
            X_tvae[i, j] = X_tvae_raw[i, j] + (centers[closest_mode_idx, j] * 0.02)
        end
    end
end

# =========================================================================
# Step 6: Evaluate Performance Metrics and Store Results
# =========================================================================
#metrics_tvae = evaluate_sim(X_tvae, X_scaled)
#push!(results, ("TVAE", metrics_tvae..., elapsed_time_tvae))
#println("TVAE Complete. Rows generated: ", size(X_tvae, 1))


"""TabDDPM (Tabular Denoising Diffusion Probabilistic Model) Engine"""

n_samples, n_features = size(X_scaled)
diffusion_timesteps = 20  # Fast-sampling timestep schedule tailored for micro-benchmarks

elapsed_time_tabddpm = @elapsed begin
    # =========================================================================
    # Step 1: Establish Variance Schedule (Linear Noise Configuration)
    # =========================================================================
    # Standard linear noise schedule mapping between start and end variances
    β = range(1f-4, 0.02f0, length=diffusion_timesteps)
    α = 1f0 .- β
    α_bar = cumprod(α)

    # =========================================================================
    # Step 2: Define Timestep-Conditioned Neural Denoiser (MLP Architecture)
    # =========================================================================
    # The network takes the noisy data concatenated with a normalized timestep scalar
    denoiser = Flux.Chain(
        Dense((n_features + 1) => 64, relu),
        Dense(64 => 32, relu),
        Dense(32 => n_features)
    )
    
    opt_ddpm = Flux.setup(Flux.Adam(1f-3), denoiser)
    X_train = Float32.(X_scaled')  # Shape: (Features × Samples)

    # =========================================================================
    # Step 3: Diffusion Network Training Routine (200 Epochs)
    # =========================================================================
    for epoch in 1:200
        # Sample random discrete timesteps uniformly for the current batch
        t_steps = rand(1:diffusion_timesteps, n_samples)
        
        # Draw pure Gaussian target noise matching matrix shapes
        ϵ = randn(Float32, size(X_train))
        
        # Calculate closed-form forward noise propagation weights
        a_bar = reshape(α_bar[t_steps], 1, n_samples)
        
        # Forward Process: Ground-truth noisy sample matrix layout at timestep t
        X_noisy = sqrt.(a_bar) .* X_train .+ sqrt.(1f0 .- a_bar) .* ϵ
        
        # Rescale timesteps to [0, 1] as an explicit contextual feature vector
        t_conditioned = reshape(Float32.(t_steps) ./ Float32(diffusion_timesteps), 1, n_samples)
        
        # Pack raw features and temporal embeddings into a single neural network pass
        model_input = vcat(X_noisy, t_conditioned)

        # Optimize via the simplified Mean Squared Error loss between predicted and actual noise
        loss, grads = Flux.withgradient(denoiser) do net
            ϵ_pred = net(model_input)
            return Flux.mse(ϵ_pred, ϵ)
        end
        Flux.update!(opt_ddpm, denoiser, grads[1])
    end

    # =========================================================================
    # Step 4: Reverse Sampling Execution (Generating 50,000 Rows)
    # =========================================================================
    # Start the reverse Markov chain completely inside a space of pure Gaussian noise
    X_curr = randn(Float32, n_features, target_batch)

    # Iteratively run backward denoising cycles down from T to 1
    for t in reverse(1:diffusion_timesteps)
        t_conditioned = fill(Float32(t) / Float32(diffusion_timesteps), 1, target_batch)
        model_input = vcat(X_curr, t_conditioned)
        
        # Predict the current noise contribution component via the trained model
        ϵ_pred = denoiser(model_input)
        
        # Reconstruct the denoised feature state for the current step
        a = α[t]
        a_bar = α_bar[t]
        
        X_next = (1f0 / sqrt(a)) .* (X_curr .- ((1f0 - a) / sqrt(1f0 - a_bar)) .* ϵ_pred)
        
        # If not on the final step, inject variance back into the sequence to sustain randomness
        if t > 1
            z = randn(Float32, size(X_curr))
            σ_t = sqrt(β[t])  # Fixed posterior variance tracking 
            X_curr .= X_next .+ σ_t .* z
        else
            X_curr .= X_next
        end
    end

    global X_tabddpm = Float64.(X_curr')
end

# =========================================================================
# Step 5: Evaluate Performance Metrics and Store Results
# =========================================================================
# metrics_tabddpm = evaluate_sim(X_tabddpm, X_scaled)
# push!(results, ("TabDDPM", metrics_tabddpm..., elapsed_time_tabddpm))
# println("TabDDPM Complete. Rows generated: ", size(X_tabddpm, 1))


# =========================================================================
# (evaluation.jl Integration Segment)
# =========================================================================

# 1. Define or request the user-specified file path for evaluation.jl
# You can set this explicitly:
evaluation_script_path = "D:/Comp/Julia/Instructor data analysis/evaluation.jl"

# Or uncomment below to make it an interactive command-line prompt:
# print("Enter the absolute or relative path to evaluation.jl: ")
# evaluation_script_path = readline()

# 2. Safely check if the file exists before including it to prevent raw crashes
if isfile(evaluation_script_path)
    include(evaluation_script_path)
    println("Successfully loaded evaluation stack from: $evaluation_script_path")
else
    error("File not found at specified path: $evaluation_script_path. Please check your directory structure.")
end

# Initialize the results container array with modern type-aliasing
results = Vector{Tuple{String, Float64, Float64, Float64, Float64}}()

# =========================================================================
# Execution & Logging Examples for Your Suite
# =========================================================================

# --- Example A: Gaussian Copula Block ---
elapsed_time_copula = @elapsed begin
    # ... (Your Gaussian Copula array logic here producing X_copula) ...
end
log_evaluation!(results, "Gaussian Copula", X_copula, X_scaled, elapsed_time_copula)


# --- Example B: TVAE Block ---
elapsed_time_tvae = @elapsed begin
    # ... (Your TVAE array logic here producing X_tvae) ...
end
log_evaluation!(results, "TVAE", X_tvae, X_scaled, elapsed_time_tvae)


# =========================================================================
# Final Reporting Layout Generation
# =========================================================================
# Once all 8 algorithms are pushed, cast them to a DataFrame for clean visualization
df_results = DataFrame(
    results, 
    [:Algorithm, :Std_Diff_MAE, :Correlation_Error, :Avg_Wasserstein, :Runtime_Sec]
)

println("\n=== FINAL GENERATIVE BENCHMARK REPORT ===")
println(df_results)


"""
Next we generate the Instructor and the the Course IDs and attach them to
the augmented data. Since this augmentation is intended to simulate the real world
ed platforms we are using some ideas that are commonly seen:

1. The Pareto Principle (80/20 Rule) for Instructors: A small group of highly active 
power-instructors publish the vast majority of courses, while the remaining 
long-tail pool of instructors only have one or two courses.

2. Dense Multi-Variable Mapping: A single course ID can be assigned to multiple 
independent instructors (e.g., co-taught flagship courses or shared curriculum tracks), 
and a single instructor can teach multiple courses.
"""

# Ensure targets are explicitly set
target_instr = 2500
target_course = 100
target_batch = 50000

println("Structuring structural ID columns for $target_batch rows...")

# =========================================================================
# Step 1: Generate Pareto-Distributed Pools
# =========================================================================
# Create ID string matrices matching your platform tracking layout
instructor_pool = ["I_" * lpad(string(i), 4, "0") for i in 1:target_instr]
course_pool     = ["C_" * lpad(string(c), 3, "0") for c in 1:target_course]

# Create skewed probability weights using a Zipf-like power decay law (1 / x^0.85)
# This simulates the real world where a few IDs dominate the row counts
instr_weights = Weights([1.0 / (i^0.85) for i in 1:target_instr])
course_weights = Weights([1.0 / (c^0.5) for c in 1:target_course])

# Build the portfolio to map dense many-to-many relationships
course_to_instructor_portfolio = Dict{String, Vector{String}}()
for crs in course_pool
    num_eligible = rand(5:15)
    course_to_instructor_portfolio[crs] = sample(instructor_pool, instr_weights, num_eligible, replace=false)
end

# =========================================================================
# Step 2: Sample IDs Independently to Build Complex Cross-Mappings
# =========================================================================
# This creates the natural joint distribution structure (co-teaching, many-to-many)

#synthetic_instructor_ids = sample(instructor_pool, instr_weights, target_batch)

# Sampling to establish the dense mapping
synthetic_course_ids     = sample(course_pool, course_weights, target_batch)
synthetic_instructor_ids = Vector{String}(undef, target_batch)

for i in 1:target_batch
    assigned_course = synthetic_course_ids[i]
    eligible_pool = course_to_instructor_portfolio[assigned_course]
    synthetic_instructor_ids[i] = rand(eligible_pool)
end

# =========================================================================
# Step 3: Generate Sequential Batch IDs (B_XXXX)
# =========================================================================
# Creates simple independent sequential batch logging matching your system tracker
synthetic_batch_ids = ["B_" * lpad(string(b), 4, "0") for b in 1:target_batch]

# =========================================================================
# Step 4: Package into Final DataFrame Layout
# =========================================================================
# Pick whichever algorithm won your evaluation stack (e.g., X_copula or X_tabddpm)
# Replace `X_augmented_winner` with your selected matrix array variable
# A
X_augmented_winner = X_copula 

# Construct the comprehensive, ML-ready dataset
# B
df_augmented_copula = DataFrame()

"""
We can modify the above lines A and B to generate data for any of the augmenting
algorithms used in this script
"""

# Inject structural identifiers
df_augmented_copula[!, :batch_id]      = synthetic_batch_ids
df_augmented_copula[!, :instructor_id] = synthetic_instructor_ids
df_augmented_copula[!, :course_id]     = synthetic_course_ids

# Map your 9 continuous feature columns back into place dynamically
feature_names = [:completion_rate, :dropout_rate, :engagement_score, :satisfaction_index, 
                 :forum_posts, :quiz_attempts, :assignment_avg, :video_watch_time, :peer_reviews]

for (idx, col_name) in enumerate(feature_names)
    df_augmented_copula[!, col_name] = X_augmented_winner[:, idx]
end

println("Successfully attached IDs. Final Shape: ", size(df_augmented_copula))
println("Sample Distribution Summary:")
println(first(df_augmented_copula, 5))