using CSV, DataFrames
using PythonPlot

# --- Configuration ---
region_names = Dict(:x1 => "rich", :x2 => "poor")
region_labels = ["Rich", "Poor"]
base_path = "./results/Decomposition_v2/full_model"

ETHBlue = "#215CB0"
ETHRed = "#B7352D"

# --- Load data ---
config_names = ["config1", "config2f", "config3f", "config4f"]
config_labels = [
    "C1: Mitigation only",
    "C2f: + Flow adaptation",
    "C3f: + Cost cap (0.8%)",
    "C4f: + Cap + Carbon budget"
]

emissions = Dict{String, DataFrame}()
for cfg in config_names
    df = CSV.read(joinpath(base_path, cfg, "Emissions.csv"), DataFrame)
    rename!(df, region_names)
    emissions[cfg] = df
end

# --- Compute cumulative emissions (sum × 10 for decade steps → GtC) ---
function cum_emissions(df)
    rich = sum(df.rich) * 10
    poor = sum(df.poor) * 10
    total = rich + poor
    share_rich = rich / total * 100
    share_poor = poor / total * 100
    return (; rich, poor, total, share_rich, share_poor)
end

cum = Dict(cfg => cum_emissions(emissions[cfg]) for cfg in config_names)

# Print summary
println("=" ^ 60)
println("Cumulative emissions summary (GtC)")
println("=" ^ 60)
for cfg in config_names
    c = cum[cfg]
    println("$cfg: rich=$(round(c.rich; digits=1)), poor=$(round(c.poor; digits=1)), " *
            "total=$(round(c.total; digits=1)), share_rich=$(round(c.share_rich; digits=2))%")
end
println()

# --- Bar chart function ---
# mode: :pp for percentage-point change in share, :pct for relative % change in share
function make_bar_chart(baseline_cfg, comparison_cfg, title_text, filename;
                        ylim_val=3.0, mode=:pp)
    b = cum[baseline_cfg]
    c = cum[comparison_cfg]

    if mode == :pct
        # Relative percentage change in share: (new - old) / old × 100
        Δ_rich = (c.share_rich - b.share_rich) / b.share_rich * 100
        Δ_poor = (c.share_poor - b.share_poor) / b.share_poor * 100
        unit = "%"
        ylabel = "Change in cumulative emissions share (%)"
    else
        # Percentage-point change in share
        Δ_rich = c.share_rich - b.share_rich
        Δ_poor = c.share_poor - b.share_poor
        unit = " pp"
        ylabel = "Change in cumulative emissions share (pp)"
    end
    Δ = [Δ_rich, Δ_poor]

    fig, ax = PythonPlot.subplots(1, 1, figsize=(8, 4))

    ax.bar(0, Δ_rich, color=ETHBlue, clip_on=false, width=0.6)
    ax.bar(1, Δ_poor, color=ETHBlue, clip_on=false, width=0.6)

    ax.set_title(title_text, weight="bold", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylim(-ylim_val, ylim_val)
    ax.spines["bottom"].set_visible(false)
    ax.spines["bottom"].set_position(("outward", -20))

    # Value labels
    for (j, y) in enumerate(Δ)
        offset = y ≥ 0 ? 0.08 : -0.08
        va = y ≥ 0 ? "bottom" : "top"
        label = (y ≥ 0 ? "+" : "") * string(round(y; digits=2)) * unit
        ax.text(j-1, y + offset, label, ha="center", va=va, fontsize=10,
                color=ETHBlue, weight="bold")
    end

    ax.set_xticks(0:1, region_labels)

    # Total emissions annotation
    total_b = round(Int, b.total)
    total_c = round(Int, c.total)
    annotation = "Total: $total_b → $total_c GtC"
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray",
            bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "wheat", "alpha" => 0.5))

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=200)
    println("Saved: $filename")
    display(fig)
end

# --- Generate 3 charts ---

# 1. Flow adaptation vs baseline (separability ≈ zero change)
make_bar_chart("config1", "config2f",
    "Flow adaptation does not change mitigation burdens",
    "./Figures/cum_em_change_flow_ad.png";
    ylim_val=1.0)

# 2. Cost constraint vs baseline (no carbon budget)
make_bar_chart("config1", "config3f",
    "Investment constraints shift mitigation toward rich region",
    "./Figures/cum_em_change_constrained_no_budget.png";
    ylim_val=2.0)

# 3. Cost constraint + carbon budget vs baseline
make_bar_chart("config1", "config4f",
    "Constrained investment + carbon budget",
    "./Figures/cum_em_change_constrained_budget.png";
    ylim_val=2.0, mode=:pct)

println("\nDone. All charts saved to ./Figures/")

#%%
########################################################################################################
# Legacy analysis: MyResults folder (non-multistart single runs)
#
# Reads from results/MyResults/{rice_utilitarian, var_ad_rice_utilitarian, stock_ad_rice_utilitarian}
# Computes relative (%) change in cumulative emissions shares + time series plot
########################################################################################################


function run_legacy_analysis(; remove_negishi=false, stock_ad=true, constrained=true)
    suffix = remove_negishi ? "_no_negishi" : ""

    ut_emissions = CSV.read("./results/MyResults/rice_utilitarian$(suffix)/Emissions.csv", DataFrame)
    var_ad_ut_emissions = CSV.read("./results/MyResults/var_ad_rice_utilitarian$(suffix)/Emissions.csv", DataFrame)
    stock_ad_ut_emissions = CSV.read("./results/MyResults/stock_ad_rice_utilitarian$(suffix)/Emissions.csv", DataFrame)

    legacy_region_names = Dict(:x1 => "rich", :x2 => "poor")
    rename!(ut_emissions, legacy_region_names)
    rename!(var_ad_ut_emissions, legacy_region_names)
    rename!(stock_ad_ut_emissions, legacy_region_names)

    # Cumulate emissions
    cum_em = DataFrame(Region=names(ut_emissions), cum_em=[sum(ut_emissions[:, col]) for col in names(ut_emissions)])
    var_ad_cum_em = DataFrame(Region=names(var_ad_ut_emissions), var_ad_cum_em=[sum(var_ad_ut_emissions[:, col]) for col in names(var_ad_ut_emissions)])
    stock_ad_cum_em = DataFrame(Region=names(stock_ad_ut_emissions), stock_ad_cum_em=[sum(stock_ad_ut_emissions[:, col]) for col in names(stock_ad_ut_emissions)])

    # Total emissions
    baseline_cum_em_total = 10 .* sum([sum(ut_emissions[:, col]) for col in names(ut_emissions)])
    var_ad_cum_em_total = 10 .* sum([sum(var_ad_ut_emissions[:, col]) for col in names(var_ad_ut_emissions)])
    stock_ad_cum_em_total = 10 .* sum([sum(stock_ad_ut_emissions[:, col]) for col in names(stock_ad_ut_emissions)])

    # Combine and compute shares
    combined_emissions = innerjoin(stock_ad_cum_em, var_ad_cum_em, cum_em, on=:Region)
    combined_emissions[!, :stock_ad_cum_em_share] = combined_emissions.stock_ad_cum_em ./ sum(combined_emissions.stock_ad_cum_em)
    combined_emissions[!, :var_ad_cum_em_share] = combined_emissions.var_ad_cum_em ./ sum(combined_emissions.var_ad_cum_em)
    combined_emissions[!, :cum_em_share] = combined_emissions.cum_em ./ sum(combined_emissions.cum_em)

    # Relative change in share (%)
    combined_emissions[!, :stock_ad_Change] = (combined_emissions.stock_ad_cum_em_share .- combined_emissions.cum_em_share) ./ combined_emissions.cum_em_share .* 100
    combined_emissions[!, :var_ad_Change] = (combined_emissions.var_ad_cum_em_share .- combined_emissions.cum_em_share) ./ combined_emissions.cum_em_share .* 100

    # Bar chart
    vals = stock_ad ? combined_emissions.stock_ad_Change : combined_emissions.var_ad_Change
    legacy_labels = ["rich", "poor"]

    fig, ax = PythonPlot.subplots(1, 1, figsize=(8, 4))
    ax.bar(0:1, vals, color=(33/255, 92/255, 175/255), clip_on=false)
    ax.set_title("$(stock_ad ? "Stock & flow adaptation" : "Flow adaptation") as a decision variable - $(constrained ? "constrained" : "unconstrained") climate investment", weight="bold")
    ax.set_ylabel("Change in cumulative emissions share by region (%)")
    ax.set_xlabel("RICE2010 Region")
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-11, 11)
    ax.spines["bottom"].set_visible(false)
    ax.spines["bottom"].set_position(("outward", -20))
    for j in 1:length(vals)
        y = vals[j]
        offset = y ≥ 0 ? 0.15 : -0.25
        va = y ≥ 0 ? "bottom" : "top"
        ax.text(j-1, y + offset, (y ≥ 0 ? "+" : "") * string(round(y; digits=2))*"%", ha="center", va=va, fontsize=8, color=(33/255, 92/255, 175/255))
    end
    ax.set_xticks(0:1, legacy_labels)
    fig.tight_layout()
    fig.savefig("./Figures/legacy_cum_em_change_$(stock_ad ? "stock_ad" : "var_ad")_$(constrained ? "constrained" : "unconstrained").png", bbox_inches="tight", dpi=200)
    println("Saved legacy bar chart")
    display(fig)

    # Time series plot
    fig2, ax2 = PythonPlot.subplots(1, 1, figsize=(6, 4))
    time = collect(2015:10:2605)[1:20]
    ax2.plot(time, (ut_emissions.rich .* 10)[1:20], label="No adaptation - rich", color="blue")
    ax2.plot(time, (ut_emissions.poor .* 10)[1:20], label="No adaptation - poor", color="orange")
    ax2.plot(time, (stock_ad_ut_emissions.rich .* 10)[1:20], label="Constrained investment - rich", color="blue", linestyle="--")
    ax2.plot(time, (stock_ad_ut_emissions.poor .* 10)[1:20], label="Constrained investment - poor", color="orange", linestyle="--")
    ax2.legend()
    ax2.set_ylabel("Emissions (GtC/decade)")
    ax2.set_xlabel("Year")
    fig2.tight_layout()
    fig2.savefig("./Figures/legacy_emissions_timeseries.png", bbox_inches="tight", dpi=200)
    println("Saved legacy time series")
    display(fig2)

    return combined_emissions
end

# Uncomment to run legacy analysis:
# run_legacy_analysis(remove_negishi=false, stock_ad=true, constrained=true)
