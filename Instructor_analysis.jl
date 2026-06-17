"""If there are any packages not installed, following
 is an example of how to install it"""
 # import Pkg; Pkg.add("CSV")

 """Loading necessary libraries"""

using CSV
using DataFrames
using Statistics
using LinearAlgebra

"""
NOTE:
It is a good idea to avoid Plotly as it causes Interface and 
version issues. Compatiblity is a big problem. Also compiling 
takes unusually long upto several minutes (maybe even 20)
"""

# loading data
data = CSV.read("D:/intern/instructor_effectiveness_dataset_2000_rows.csv",DataFrame);

# column names
println(names(data))

# see some rows of the data
describe(data)

# Shows column name and its element type
println([(name, eltype(col)) for (name, col) in pairs(eachcol(data))])

#use this function from DataFrames to see specified no of rows
#first(data,20)

# check for missing values column wise
[col => sum(ismissing.(data[!, col])) for col in names(data)]

# there seems to be no missing values in any of the columns

#taking the columns required for plotting
plot_df = data[:,4:12]

"""For visuals we begin with simple 2D plots first"""

"""First looking at the individual distributions of the features. For this we need
to first standardize the columns as they have different value ranges"""

using MLJ

# Compact assignment syntax
normal_scaler(df) = MLJ.transform(fit!(machine(Standardizer(), df)), df)

"""
# function form
function NormalScaler(df)
    mach = machine(Standardizer(), df) |> fit!
    return MLJ.transform(mach, df)
end
"""

plotdf_scaled = normal_scaler(plot_df)


"""Building distribution visuals"""

using Plots

function plot_all_histograms(df::DataFrame)
    # Get names of columns where the element type is a subtype of Number
    numeric_cols = [n for n in names(df) if eltype(df[!, n]) <: Number]
    
    if isempty(numeric_cols)
        println("No numeric columns found to plot.")
        return nothing
    end

    plot_list = []
    
    for col in numeric_cols
        # dropmissing is used to handle NAs which would otherwise break the histogram
        data = filter(!isnan, collect(skipmissing(df[!, col])))
        
        p = histogram(data, 
                      title = "Histogram of $col", 
                      xlabel = col, 
                      ylabel = "Frequency", 
                      legend = false)
        push!(plot_list, p)
    end
    
    # Arrange in a grid; ncols=2 makes it easier to read than one long strip
    return plot(plot_list..., layout = (length(plot_list), 1), size = (1200, 300 * length(plot_list)))
end

plot_all_histograms(plotdf_scaled)

"""
the distributions show us that most of the features do not
have a normal distribution with deviations. Many of the
features are skewed
"""

"""
We now check for correlations to get an idea of associations
between one or more features/data fields.
"""

using CairoMakie

# Note that using the CairoMakie library takes a long executio time

function plot_correlations_makie(df::DataFrame; figure_size=(800, 800))
    # 1. Filter for numeric columns
    numeric_cols = [name for name in names(df) if eltype(df[!, name]) <: Number]
    n = length(numeric_cols)
    
    if n == 0
        println("No numeric columns found!")
        return nothing
    end

    # 2. Calculate Correlation Matrix
    # Matrix(df[:, numeric_cols]) converts the dataframe subset to a matrix for cor()
    corr_matrix = cor(Matrix(df[:, numeric_cols]))

    # 3. Create the Figure
    fig = CairoMakie.Figure(size = figure_size)
    
    # We use yreversed = true so that the first variable appears at the TOP,
    # making it read like a standard correlation table.
    ax = CairoMakie.Axis(fig[1, 1], 
              title = "Correlation Matrix",
              xticks = (1:n, numeric_cols),
              yticks = (1:n, numeric_cols),
              xticklabelrotation = π/4,
              yreversed = true) 

    # 4. Plot the Heatmap
    # Use the matrix directly. Makie maps matrix[i, j] to x=i, y=j.
    hm = CairoMakie.heatmap!(ax, 1:n, 1:n, corr_matrix, 
                  colorrange = (-1, 1), 
                  colormap = :RdBu)

    # 5. Add text annotations
    for i in 1:n, j in 1:n
        val = corr_matrix[i, j]
        # Logic for high contrast text: use white on dark colors
        txt_color = abs(val) > 0.5 ? :white : :black
        
        CairoMakie.text!(ax, string(round(val, digits=2)),
              position = (i, j),
              align = (:center, :center),
              color = txt_color,
              fontsize = 14)
    end

    # 6. Add a Colorbar
    CairoMakie.Colorbar(fig[1, 2], hm, label = "Correlation Coefficient")

    return fig
end


"""
There are tools to time executions line by line. We can use the
BenchmarkTools for getting the time for execution just to
see how it works
"""

using BenchmarkTools
# correlation function call
@btime plot_correlations_makie(plotdf_scaled)


"""
correlations don't tell us much except high negative correlation 
between dropout and completion rate which is obvious. 
It could be that correlations cannot be above a certain
value which is specific to this ed tech domain or this use case
"""

"""
Let's explore the data a bit more before going for any 
more plotting
"""

"""
Gathering observations about the data.

1. The ID columns hold simply the uinque identifiers for the course,
instructor and batch.

2. The rate columns have the highest value of 1 with other values 
having decimals without any blatant pattern suggesting they 
could be percentages though the totals from which such 
computation was done cannot be surmised.

3. The forum activity rate is consistently low across all batches, 
courses and instructors which is true in real life course 
platforms.

4. The score columns are scores out of different total scores and need to be
standardized.
"""

"""
We need to add the normalised data back with the ID columns
which are like primary keys for the data
"""
@btime instructor_df = hcat(data[:,1:3],plotdf_scaled)

# Now we further visualize
using StatsPlots

# Comparing courses

function plot_course_comparison_stats(df)
    # 1. Generate the violin plot independently
    p1 = @df df StatsPlots.violin(
        cols(:course_id), cols(:completion_rate), 
        group=cols(:course_id), 
        title="Completion Rate Variance",
        legend=false, 
        marker=(:circle, 3, 0.3)
    )
    
    # 2. Generate the boxplot independently 
    p2 = @df df StatsPlots.boxplot(
        cols(:course_id), cols(:completion_rate), 
        group=cols(:course_id),
        title="Completion Rate Boxplot",
        legend=false,
        fillalpha=0.5
    )
    
    # 3. Combine both plots into a 2-row, 1-column vertical grid layout
    return StatsPlots.plot(p1, p2, layout=(2, 1), size=(600, 800))
end

# Run the benchmark
@btime plot_course_comparison_stats(instructor_df)


"""once again no apparent linear pattern based on OLS"""

"""Now we define a measure for instructor effectiveness and visualize 
based on the measure"""

using StatsBase

function define_effectiveness!(df::DataFrame)
    # Check if df is empty or missing
    if isempty(df)
        println("Load data first!")
        return df
    end

    # List of metrics to scale
    metrics_cols = [
        :completion_rate, :avg_score_improvement, :avg_quiz_score,
        :avg_watch_time, :assignment_submission_rate, :forum_activity_rate,
        :avg_feedback_score, :feedback_response_rate
    ]

    # 1. Scaling individual metrics to 0-1 (Min-Max Scaling)
    # We create a temporary matrix of scaled values
    scaled_df = copy(df[:, metrics_cols])
    for col in names(scaled_df)
        vals = scaled_df[!, col]
        min_v, max_v = extrema(vals)
        # Avoid division by zero if all values are the same
        if max_v > min_v
            scaled_df[!, col] = (vals .- min_v) ./ (max_v - min_v)
        else
            scaled_df[!, col] .= 0.0
        end
    end

    # 2. Calculate Pillars
    # Outcomes: indices 1, 2, 3 (0, 1, 2 in Python)
    outcomes = (scaled_df[:, 1] + scaled_df[:, 2] + scaled_df[:, 3]) ./ 3
    
    # Engagement: indices 4, 5, 6 (3, 4, 5 in Python)
    engagement = (scaled_df[:, 4] + scaled_df[:, 5] + scaled_df[:, 6]) ./ 3
    
    # Feedback: indices 7, 8 (6, 7 in Python)
    feedback = (scaled_df[:, 7] + scaled_df[:, 8]) ./ 2

    # 3. Effectiveness Score Formula
    df.effectiveness_score = (0.4 .* outcomes) + (0.3 .* engagement) + (0.3 .* feedback)

    # 4. Tiering using quantiles (qcut equivalent)
    # n-quantiles splits data into n equal-sized groups
    qs = quantile(df.effectiveness_score, [0, 1/3, 2/3, 1.0])
    
    df.tier = map(df.effectiveness_score) do score
        if score <= qs[2]
            "Low"
        elseif score <= qs[3]
            "Medium"
        else
            "High"
        end
    end

    return df
end

effectiveness_df = define_effectiveness!(instructor_df)

# At this stage we can view the effectiveness column showing the intended measure

"""Next we aggregate the data by the effectiveness tier made by using various measures"""

function aggregate_to_instructor_tier!(df::DataFrame)
    # 1. Define the aggregation
    # We use 'mean' for scores/rates and 'nrow' (count) for batch_id
    instructor_df = combine(groupby(df, :instructor_id),
        :completion_rate             => mean => :completion_rate,
        :avg_score_improvement       => mean => :avg_score_improvement,
        :avg_quiz_score              => mean => :avg_quiz_score,
        :dropout_rate                => mean => :dropout_rate,
        :avg_watch_time              => mean => :avg_watch_time,
        :assignment_submission_rate  => mean => :assignment_submission_rate,
        :forum_activity_rate         => mean => :forum_activity_rate,
        :avg_feedback_score          => mean => :avg_feedback_score,
        :feedback_response_rate      => mean => :feedback_response_rate,
        :effectiveness_score         => mean => :effectiveness_score,
        :batch_id                    => length => :batch_count
    )

    # 2. Recalculate Tier at Instructor Level using quantiles
    qs = quantile(instructor_df.effectiveness_score, [0, 1/3, 2/3, 1.0])
    
    instructor_df.tier = map(instructor_df.effectiveness_score) do score
        if score <= qs[2]
            "Low"
        elseif score <= qs[3]
            "Medium"
        else
            "High"
        end
    end

    # 3. Export to CSV
#    CSV.write("instructor_level_data.csv", instructor_df)

    return instructor_df
end

aggregate_df = aggregate_to_instructor_tier!(instructor_df)

"""
The plots dont tell us much except some
basic and not very important features.
We can try some other things like interpreting the
distributions and find top instructors by tier
"""

"""Top instructors by tier"""

using DataFrames

# 1. Group by tier
# 2. Sort each group by effectiveness_score (descending)
# 3. Take the first 10 rows of each group
top_instructors_by_tier = combine(groupby(instructor_df, :tier)) do sdf
    # Sort the sub-dataframe (sdf) by score in reverse (rev=true)
    sorted_sdf = sort(sdf, :effectiveness_score, rev=true)    
    # Return the first 10 rows for each tier
    return first(sorted_sdf, 10)
end

# Select specific columns to make the output readable
final_list = top_instructors_by_tier[:, [:tier, :instructor_id, :effectiveness_score]]

# Display the result
println(final_list)

# 1. Ensure the data is sorted for the plot
# We'll sort by tier and then score so the bars look organized
plot_data = sort(top_instructors_by_tier, [:tier, :effectiveness_score], rev=[false, true])

# 2. Create the Bar Plot
# We use 'group' to automatically color-code by Tier
groupedbar(
    plot_data.instructor_id, 
    plot_data.effectiveness_score,
    group = plot_data.tier,
    title = "Top 10 Instructors per Tier",
    xlabel = "Instructor ID",
    ylabel = "Effectiveness Score",
    xrotation = 45,               # Tilt IDs so they don't overlap
    size = (1200, 800),            # Make it wide enough for 30 bars
    legend = :outertopright,
    color = [:red :green :orange], # Matches Low, High, Medium alphabetically
    alpha = 0.8
)

"""
Seeing that we cannot infer much about the data as it is, we'll
now attempt to augment the data to see if any interesting patterns come up.
One thing is that we don't want to add any external data, so we'll use
data generating methods based on the available 2000 observations.    
We'll target augmenting upto 50000 observations, proportionately
increasing the instructors and the courses to keep it realistic.
"""
"""The augmenting procedures will be coded into Data_augmenting.jl script file"""