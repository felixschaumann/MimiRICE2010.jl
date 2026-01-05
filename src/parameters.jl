function getrice2010parameters(filename)
    p = Dict{Symbol,Any}()

    T = 60
    # p[:timesteps] = 1:T # Time periods (5 years per period); not currently used in constructrice
    rice_regions = ["US", "EU", "Japan", "Russia", "Eurasia", "China", "India", "MidEast", "Africa", "LatAm", "OHI", "OthAsia"]
    regions = ["rich", "poor"]

    # Define aggregation mapping
    region_mapping = Dict(
        "rich" => ["US", "EU", "Japan", "Russia", "OHI"],
        "poor" => ["Eurasia", "China", "India", "MidEast", "Africa", "LatAm", "OthAsia"]
    )
    
    # Open RICE_2010 Excel File to Read Parameters
    f = readxlsx(filename)

    # Time Step
    # p[:tstep] = 10 # Years per Period; not currently used in constructrice
    # p[:dt] =  10 # Time step parameter for model equations; not currently used in constructrice

    # If optimal control
    ifopt = true # Indicator where optimized is 1 and base is 0

    # Preferences
    # p[:elasmu] =  getparam_single(f, "B18:B18", rice_regions) # Elasticity of MU of consumption
    p[:elasmu] =  aggregate_regions(getparam_single(f, "B18:B18", rice_regions), region_mapping, rice_regions, mean) # Elasticity of MU of consumption
    # p[:prstp] =  getparam_single(f, "B15:B15", regions) # Rate of Social Time Preference; not currently used in constructrice

    # Population and technology

    # Capital elasticity in production function
    p[:gama] = 0.300
    # p[:dk]  = getparam_single(f, "B8:B8", regions) # Depreciation rate on capital (per year)
    p[:dk]  = aggregate_regions(getparam_single(f, "B8:B8", rice_regions), region_mapping, rice_regions, mean) # Depreciation rate on capital (per year)
    # p[:k0] = getparam_single(f, "B11:B11", regions) #Initial capital
    p[:k0] = aggregate_regions(getparam_single(f, "B11:B11", rice_regions), region_mapping, rice_regions, sum) #Initial capital
    # p[:miu0] = getparam_single(f, "B103:B103", regions) # Initial emissions control rate for base case 2010; not currently used in constructrice
    # p[:miubase] = getparam_timeseries(f, "B103:BI103", regions, T) # Optimized emission control rate results from RICE2010 (base case); duplicate line to p[:MIU]

    # Carbon cycle

    # Initial Conditions
    p[:mat0] =  787.0 # Initial Concentration in atmosphere 2010 (GtC)
    p[:mat1] =  829.0
    p[:mu0] = 1600. # Initial Concentration in upper strata 2010 (GtC)
    p[:ml0] =  10010. # Initial Concentration in lower strata 2010 (GtC)

    # Carbon cycle transition matrix

    # Flow paramaters
    p[:b12] = 12.0/100 # Carbon cycle transition matrix atmosphere to shallow ocean
    p[:b23] = 0.5/100 # Carbon cycle transition matrix shallow to deep ocean

    # Parameters for long-run consistency of carbon cycle
    p[:b11] = 88.0/100 # Carbon cycle transition matrix atmosphere to atmosphere
    p[:b21] = 4.704/100 # Carbon cycle transition matrix biosphere/shallow oceans to atmosphere
    p[:b22] = 94.796/100 # Carbon cycle transition matrix shallow ocean to shallow oceans
    p[:b32] = 0.075/100 # Carbon cycle transition matrix deep ocean to shallow ocean
    p[:b33] = 99.925/100 # Carbon cycle transition matrix deep ocean to deep oceans

    # Climate model parameters
    p[:t2xco2] = 3.2 # Equilibrium temp impact (oC per doubling CO2)
    fex0 = 0.83 # 2010 forcings of non-CO2 GHG (Wm-2)
    fex1 = 0.3 # 2100 forcings of non-CO2 GHG (Wm-2)
    p[:tocean0] = .0068 #  Initial lower stratum temp change (C from 1900)
    p[:tatm0] = 0.83 # Initial atmospheric temp change 2005 (C from 1900)
    p[:tatm1] = 0.98 # Initial atmospheric temp change 2015 (C from 1900)

    # Transient TSC Correction ("Speed of Adjustment Parameter")
    p[:c1] = 0.208 # Climate equation coefficient for upper level
    p[:c3] = 0.310 # Transfer coefficient upper to lower stratum
    p[:c4] = 0.05 # Transfer coefficient for lower level
    p[:fco22x] = 3.8 # Forcings of equilibrium CO2 doubling (Wm-2)

    # Climate damage parameters
    p[:a1] = aggregate_regions(getparam_single(f, "B24:B24", rice_regions), region_mapping, rice_regions, mean) # Damage intercept
    p[:a2] = aggregate_regions(getparam_single(f, "B25:B25", rice_regions), region_mapping, rice_regions, mean) # Damage quadratic term
    p[:a3] = aggregate_regions(getparam_single(f, "B26:B26", rice_regions), region_mapping, rice_regions, mean) # Damage exponent

    # Welfare Weights
    alpha0 = transpose(f["Data"]["B359:BI370"]) # Read in alpha
    # !!! NOT PROPERLY AGGREGATED (UNWEIGHTED AVERAGE) !!!
    p[:alpha] = aggregate_regions(convert(Array{Float64}, alpha0), region_mapping, rice_regions, mean) # Convert to type used by Mimi

    # Abatement cost
    # Exponent of control cost function
    p[:expcost2] = aggregate_regions(getparam_single(f, "B38:B38", rice_regions), region_mapping, rice_regions, mean)

    # Availability of fossil fuels
    # Maximum cumulative extraction fossil fuels (GtC)
    # p[:fosslim] = 6000; not currently used in constructrice.

    # Scaling parameters
    # Multiplicative scaling coefficient
    # p[:scale1] = getparam_single(f, "B52:B52", regions)
    p[:scale1] = aggregate_regions(getparam_single(f, "B52:B52", rice_regions), region_mapping, rice_regions, mean)

    # !!! NOT PROPERLY AGGREGATED (UNWEIGHTED AVERAGE) !!!
    # Additive scaling coefficient (combines two additive scaling coefficients from RICE for calculating utility with welfare weights)
    scale2 = Array{Float64}(undef, length(rice_regions))
    for (i,r) in enumerate(rice_regions)
        data = f[r]["B53:C53"]
        scale2[i] = data[1] - data[2]
    end
    p[:scale2] = aggregate_regions(scale2, region_mapping, rice_regions, mean)

    # p[:savebase] = getparam_timeseries(f, "B97:BI97", regions, T) # Optimized savings rate in base case for RICE2010; not currently used in constructrice
    # p[:optlrsav] = getparam_single(f, "BI97:BI97", regions) # Optimized savings rate in base case for RICE2010 for last period (fraction of gross output); not currently used in constructrice
    p[:l] = aggregate_regions(getparam_timeseries(f, "B56:BI56", rice_regions, T), region_mapping, rice_regions, sum) # Level of population and labor
    p[:al] = aggregate_regions(getparam_timeseries(f, "B20:BI20", rice_regions, T), region_mapping, rice_regions, mean) # Level of total factor productivity
    p[:sigma] = aggregate_regions(getparam_timeseries(f, "B40:BI40", rice_regions, T), region_mapping, rice_regions, mean) # CO2-equivalent-emissions output ratio
    p[:pbacktime] = aggregate_regions(getparam_timeseries(f, "B36:BI36", rice_regions, T), region_mapping, rice_regions, mean) # Backstop price
    p[:cost1] = aggregate_regions(getparam_timeseries(f, "B31:BI31", rice_regions, T), region_mapping, rice_regions, mean) # Adjusted cost for backstop
    regtree = aggregate_regions(getparam_timeseries(f, "B43:BI43", rice_regions, T), region_mapping, rice_regions, sum) # Regional Emissions from Land Use Change
    p[:rr] = aggregate_regions(getparam_timeseries(f, "B17:BI17", rice_regions, T), region_mapping, rice_regions, mean) # Social Time Preference Factor

    # Global Emissions from Land Use Change (Sum of regional emissions for land use change in RICE model)
    etree = Array{Float64}(undef, T)
    for i = 1:T
        etree[i] = sum(regtree[i,:])
    end
    p[:etree] = etree

    # Exogenous forcing for other greenhouse gases
    forcoth =  Array{Float64}(undef, 60)
    data = f["Global"]["B21:BI21"]
    for i=1:T
        forcoth[i] = data[i]
    end
    p[:forcoth] = forcoth

    # Fraction of emissions in control regime
    p[:partfract] = ones(60, length(regions))

    # Savings Rate (base case RICE2010)
    p[:S] = aggregate_regions(getparam_timeseries(f, "B97:BI97", rice_regions, T), region_mapping, rice_regions, mean)

    # MIU (base case RICE2010)
    p[:MIU] = aggregate_regions(getparam_timeseries(f, "B103:BI103", rice_regions, T), region_mapping, rice_regions, mean)

    # SEA LEVEL RISE PARAMETERS
    p[:slrmultiplier] = aggregate_regions(getparam_single(f, "B49:B49", rice_regions), region_mapping, rice_regions, mean) # Multiplier for SLR
    p[:slrelasticity] = aggregate_regions(getparam_single(f, "C49:C49", rice_regions), region_mapping, rice_regions, mean) # SLR elasticity of substitution
    p[:slrdamlinear] = aggregate_regions(getparam_single(f, "B48:B48", rice_regions), region_mapping, rice_regions, mean) # SLR damage parameter (linear)
    p[:slrdamquadratic] = aggregate_regions(getparam_single(f, "C48:C48", rice_regions), region_mapping, rice_regions, mean) # SLR damage parameter (quadratic)

    # Thermal Expansion
    p[:therm0] = f["SLR"]["B9:B9"][1] # Thermal Expansion initial conditions (SLR per decade)
    p[:thermadj] = f["SLR"]["B8:B8"][1] # Thermal Expansion adjustment rate/calibration (per decade)
    p[:thermeq] = f["SLR"]["B7:B7"][1] # Thermal Expansion equilibrium (m/degree C)

    # Glaciers and Small Ice Caps (GSIC)
    p[:gsictotal] = f["SLR"]["B12:B12"][1] # GSIC total ice (SLR equivalent in meters)
    p[:gsicmelt] = f["SLR"]["B13:B13"][1] # GSIC melt rate (meters/year/degree C)
    p[:gsicexp] = f["SLR"]["B11:B11"][1] # GSIC exponent (assumed)
    # p[:gsiceq] = f["SLR"]["B14:B14"][1] # GSIC equilibrium temperature (degrees C relative to global T of -1 degree C from 2000)

    # Greenland Ice Sheet (GIS)
    p[:gis0] = f["SLR"]["B18:B18"][1] # GIS initial ice volume (meters)
    p[:gismelt0] = f["SLR"]["B17:B17"][1] # GIS initial melt rate (mm per year)
    p[:gismeltabove] = f["SLR"]["B20:B20"][1] # GIS melt rate above threshold (mm/year/degree C)
    p[:gismineq] = f["SLR"]["B19:B19"][1] # GIS minimum equilibrium temperature (degrees C)
    p[:gisexp] = f["SLR"]["B21:B21"][1] # GIS exponent on remaining

    # Antarctic Ice Sheet (AIS)
    p[:aismelt0] = f["SLR"]["B24:B24"][1] # AIS initial melt rate (mm/year)
    p[:aismeltlow] = f["SLR"]["B26:B26"][1] # AIS melt rate lower (mm/year/degrees C) [T < 3 degrees C over Antarctic)
    p[:aismeltup] = f["SLR"]["B27:B27"][1] # AIS melt rate upper (mm/year/degrees C) [T = 8 degrees C over Antarctic]
    p[:aisratio] = f["SLR"]["B29:B29"][1] # AIS ratio t-ant/t-glob
    p[:aisinflection] = f["SLR"]["B28:B28"][1] # AIS inflection point (degrees C)
    p[:aisintercept] = f["SLR"]["B25:B25"][1] # AIS intercept (mm/year)
    p[:aiswais] = f["SLR"]["B31:B31"][1] # AIS total remaining ice volume (m) for WAIS
    p[:aisother] = f["SLR"]["B32:B32"][1] # AIS total remaining ice volume (m) for other AIS

    return p
end
