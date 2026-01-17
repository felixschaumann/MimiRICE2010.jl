using CSV, DataFrames
using PythonPlot

# 1 "USA"
# 2 "Russia"
# 3 "other Asia"
# 4 "other high income"
# 5 "Middle East"
# 6 "Latin America"
# 7 "Japan"
# 8 "India"
# 9 "Eurasia"
# 10 "EU"
# 11 "China"
# 12 "Africa"

# region_names = Dict(:x1 => "USA", :x2 => "Russia", :x3 => "other Asia", :x4 => "other high income", :x5 => "Middle East", :x6 => "Latin America", :x7 => "Japan", :x8 => "India", :x9 => "Eurasia", :x10 => "EU", :x11 => "China", :x12 => "Africa")
# region_labels = ["USA", "Russia", "Other \n Asia", "Other \n High \n Income", "Middle \n East", "Latin \n America", "Japan", "India", "Eurasia", "EU", "China", "Africa"]
region_names = Dict(:x1 => "rich", :x2 => "poor")
region_labels = ["rich", "poor"]

#%%

remove_negishi = false  # set to true if the runs being analysed were done without Negishi weights

# Load the data.
ut_emissions = CSV.read("./results/MyResults/rice_utilitarian"*(remove_negishi ? "_no_negishi" : "")*"/Emissions.csv", DataFrame)
# ad_ut_emissions = CSV.read("./results/MyResults/ad_rice_utilitarian/Emissions.csv", DataFrame)
var_ad_ut_emissions = CSV.read("./results/MyResults/var_ad_rice_utilitarian"*(remove_negishi ? "_no_negishi" : "")*"/Emissions.csv", DataFrame)
stock_ad_ut_emissions = CSV.read("./results/MyResults/stock_ad_rice_utilitarian"*(remove_negishi ? "_no_negishi" : "")*"/Emissions.csv", DataFrame)

#%%
# cumulate all emissions

rename!(ut_emissions, region_names)
# rename!(ad_ut_emissions, region_names)
rename!(var_ad_ut_emissions, region_names)
rename!(stock_ad_ut_emissions, region_names)

cum_em = DataFrame(Region=names(ut_emissions), cum_em=[sum(ut_emissions[:, col]) for col in names(ut_emissions)])
# ad_cum_em = DataFrame(Region=names(ad_ut_emissions), ad_cum_em=[sum(ad_ut_emissions[:, col]) for col in names(ad_ut_emissions)])
var_ad_cum_em = DataFrame(Region=names(var_ad_ut_emissions), var_ad_cum_em=[sum(var_ad_ut_emissions[:, col]) for col in names(var_ad_ut_emissions)])
stock_ad_cum_em = DataFrame(Region=names(stock_ad_ut_emissions), stock_ad_cum_em=[sum(stock_ad_ut_emissions[:, col]) for col in names(stock_ad_ut_emissions)])

# total emissions
baseline_cum_em_total = 10 .* sum([sum(ut_emissions[:, col]) for col in names(ut_emissions)])
var_ad_cum_em_total = 10 .* sum([sum(var_ad_ut_emissions[:, col]) for col in names(var_ad_ut_emissions)])
stock_ad_cum_em_total = 10 .* sum([sum(stock_ad_ut_emissions[:, col]) for col in names(stock_ad_ut_emissions)])

# Combine the dataframes
combined_emissions = innerjoin(stock_ad_cum_em, var_ad_cum_em, cum_em, on=:Region)

# Calculate normalized shares per country
combined_emissions[!, :stock_ad_cum_em_share] = combined_emissions.stock_ad_cum_em ./ sum(combined_emissions.stock_ad_cum_em)
combined_emissions[!, :var_ad_cum_em_share] = combined_emissions.var_ad_cum_em ./ sum(combined_emissions.var_ad_cum_em)
combined_emissions[!, :cum_em_share] = combined_emissions.cum_em ./ sum(combined_emissions.cum_em)

# Calculate differences between scenarios
combined_emissions[!, :stock_ad_Change] = (combined_emissions.stock_ad_cum_em_share .- combined_emissions.cum_em_share) ./ combined_emissions.cum_em_share .* 100
combined_emissions[!, :var_ad_Change] = (combined_emissions.var_ad_cum_em_share .- combined_emissions.cum_em_share) ./ combined_emissions.cum_em_share .* 100

#%% make bar chart of percentage change in emissions share

stock_ad = true
constrained = true

fig, ax = PythonPlot.subplots(1, 1, figsize=(8, 4), sharex=true)

ETHBlue = (33/255, 92/255, 175/255)

ax.bar(0:1, stock_ad ? combined_emissions.stock_ad_Change : combined_emissions.var_ad_Change, color=ETHBlue, clip_on=false)

ax.set_title("$(stock_ad ? "Stock & flow adaptation" : "Flow adaptation") as a decision variable - $(constrained ? "constrained" : "unconstrained") climate investment", weight="bold")
ax.set_ylabel("Change in cumulative emissions share by region (%)")
ax.set_xlabel("RICE2010 Region")
ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)
ax.set_ylim(-3, 3)

# add horizontal line at 0
ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

ax.set_ylim(-51, 51)

ax.spines["bottom"].set_visible(false)
ax.spines["bottom"].set_position(("outward", -20))

vals = [stock_ad ? combined_emissions.stock_ad_Change : combined_emissions.var_ad_Change][1]
for j in 1:length(vals)
    y = vals[j]
    # Offset: above for positive, below for negative
    offset = y ≥ 0 ? 0.15 : -0.25
    va = y ≥ 0 ? "bottom" : "top"
    ax.text(j-1, y + offset, (y ≥ 0 ? "+" : "") * string(round(y; digits=2))*"%", ha="center", va=va, fontsize=8, color=ETHBlue)
end

ax.set_xticks(0:1, region_labels)

fig.tight_layout()
fig.savefig("../cum_em_change_$(stock_ad ? "stock_ad" : "var_ad")_$(constrained ? "constrained" : "unconstrained").png", bbox_inches="tight")
fig

#%%

fig, ax = PythonPlot.subplots(1, 1, figsize=(6, 4))

# plot time series of emissions for ut_emissions and stock_ad_ut_emissions for both regions
time = collect(2015:10:2605)[1:20]
ax.plot(time, (ut_emissions.rich .* 10)[1:20], label="Utilitarian - rich", color="blue")
ax.plot(time, (ut_emissions.poor .* 10)[1:20], label="Utilitarian - poor", color="orange")
ax.plot(time, (stock_ad_ut_emissions.rich .* 10)[1:20], label="Constrained mitigation - rich", color="blue", linestyle="--")
ax.plot(time, (stock_ad_ut_emissions.poor .* 10)[1:20], label="Constrained mitigation - poor", color="orange", linestyle="--")

fig